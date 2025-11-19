from __future__ import annotations

import random
from dataclasses import dataclass, field
from pathlib import Path
from typing import List
import torch
import numpy as np

from datasets import load_dataset
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    BitsAndBytesConfig,
)
from transformers.trainer_utils import set_seed
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
from trl import SFTTrainer, SFTConfig

# ---- Optional: wandb safe import ----
try:
    import wandb  # noqa: F401

    _WANDB_AVAILABLE = True
except Exception:
    _WANDB_AVAILABLE = False


# ----------------------------
# Config
# ----------------------------
@dataclass
class Config:
    # data & io
    data_json: str = ""
    output_root: str = ""
    model_id: str = "tokyotech-llm/Llama-3.1-Swallow-8B-Instruct-v0.3"

    # wandb
    project: str = "llama3-qlora_train_EX_Part_lr1e-3_adamw_torch_fused_warmup0.03"
    run_group: str = "qLoRA-multiseed"
    run_name: str = "exp"  # final run name = f"{run_name}-s{seed}"

    # seeds & schedule
    seeds: List[int] = field(default_factory=lambda: [4, 8, 12, 16])
    epochs: int = 6
    lr: float = 1e-3  # 1e-3, 1e-4, 2e-4, 2e-5, 5e-4, 5e-5
    weight_decay: float = 0.0
    warmup_ratio: float = 0.03  # 0.01, 0.03, 0.05, 0.07, 0.09

    # batches
    per_device_train_batch_size: int = 1
    per_device_eval_batch_size: int = 1
    gradient_accumulation_steps: int = 16

    # sequence
    max_length: int = 8192

    # lora
    lora_r: int = 16
    lora_alpha: int = 32
    lora_dropout: float = 0.05

    # logging/eval/save
    logging_steps: int = 10
    eval_strategy: str = "epoch"  # "no" | "steps" | "epoch"
    eval_steps: int = 200
    save_strategy: str = "epoch"  # "no" | "steps" | "epoch"
    save_steps: int = 200
    eval_split: float = 0.1
    save_total_limit: int = 2
    sample_rows: int = 5

    # backends & optim
    use_flash_attn: str = "auto"  # auto|fa2|sdpa|none
    optim: str = "adamw_torch_fused"  # adamw_torch_fused, adamw_8bit, paged_adamw_8bit
    lr_scheduler_type: str = "cosine"

    # quantization / dtype
    use_4bit: bool = True
    compute_dtype: str = "bfloat16"  # "bfloat16" | "float16"

    # misc
    packing: bool = False  # for SFT packing
    gradient_checkpointing: bool = True
    report_to: str = "wandb"  # "wandb" | "none"


# ----------------------------
# Utils
# ----------------------------
# 设置seed
def seed_all(seed: int) -> None:
    set_seed(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


# dtype 选择 _compute_dtype 确保量化运算稳定、减少溢出
def _compute_dtype(dtype_str: str):
    if dtype_str.lower() in ("bf16", "bfloat16"):
        return torch.bfloat16
    if dtype_str.lower() in ("fp16", "float16", "half"):
        return torch.float16
    return torch.bfloat16


# 注意力后端
def select_attn_backend(model, pref: str) -> str:
    """
    Returns the effective attention backend set on model.config.attn_implementation.
    """
    backend = None
    pref = (pref or "auto").lower()
    try:
        from transformers.utils import is_flash_attn_2_available  # type: ignore

        fa2_ok = is_flash_attn_2_available()
    except Exception:
        fa2_ok = False

    sdpa_ok = (
        hasattr(torch.backends, "cuda")
        and getattr(torch.backends.cuda, "sdp_kernel", None) is not None
    )

    if pref in ("fa2", "flash", "flash_attention_2"):
        backend = "flash_attention_2" if fa2_ok else ("sdpa" if sdpa_ok else "eager")
    elif pref == "sdpa":
        backend = "sdpa" if sdpa_ok else "eager"
    elif pref == "none":
        backend = "eager"
    else:  # auto
        backend = "flash_attention_2" if fa2_ok else ("sdpa" if sdpa_ok else "eager")

    try:
        if getattr(model.config, "attn_implementation", None) != backend:
            model.config.attn_implementation = backend
    except Exception:
        pass
    return backend


# 分词器
def build_tokenizer(model_id: str, max_len: int):
    tok = AutoTokenizer.from_pretrained(model_id, use_fast=True)
    if tok.pad_token is None and hasattr(tok, "eos_token"):
        tok.pad_token = tok.eos_token
    tok.padding_side = "right"
    tok.model_max_length = max_len
    return tok


# RoPE/序列守护
def rope_and_seq_guard(model, tokenizer, max_seq_len: int) -> int:
    max_pos = getattr(model.config, "max_position_embeddings", None)
    eff_seq_len = min(int(max_seq_len), int(max_pos)) if max_pos else int(max_seq_len)
    if getattr(model.config, "rope_scaling", None) and eff_seq_len <= (
        max_pos or eff_seq_len
    ):
        model.config.rope_scaling = None
    tokenizer.model_max_length = eff_seq_len
    return eff_seq_len


# 模型加载
def load_model(model_id: str, use_4bit: bool, compute_dtype: torch.dtype):
    quant_cfg = None
    torch_dtype = None
    if use_4bit:
        quant_cfg = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_use_double_quant=True,
            bnb_4bit_compute_dtype=compute_dtype,
        )
    else:
        torch_dtype = compute_dtype

    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        device_map="auto",
        quantization_config=quant_cfg,
        torch_dtype=torch_dtype,
    )
    return model


# 设置聊天模板
def make_formatting_func(tok):
    def _fmt(batch):
        msgs_batch = batch["messages"]
        if isinstance(msgs_batch, dict) or (
            isinstance(msgs_batch, list)
            and msgs_batch
            and isinstance(msgs_batch[0], dict)
        ):
            msgs_batch = [msgs_batch]
        outputs = []
        for msgs in msgs_batch:
            outputs.append(
                tok.apply_chat_template(
                    msgs, tokenize=False, add_generation_prompt=False
                )
            )
        return outputs

    return _fmt


# 设置W&B 样本回调
class WandbSamplesCallback:
    """
    After the last epoch, log a few raw formatted samples.
    Why: light sanity-checking in UI.
    """

    def __init__(self, project_on: bool, sample_rows: int):
        self.on = project_on
        self.k = max(0, int(sample_rows))

    def log_samples(self, dataset, formatting_func, step: int, epoch: float):
        if not (self.on and _WANDB_AVAILABLE) or self.k == 0:
            return
        try:
            tbl = wandb.Table(columns=["idx", "text"])
            idxs = list(range(min(self.k, len(dataset))))
            sample = (
                dataset.select(idxs)
                if hasattr(dataset, "select")
                else dataset[: self.k]
            )
            formatted = formatting_func(sample)
            for i, t in enumerate(formatted):
                tbl.add_data(idxs[i], t[:2048])
            wandb.log({"samples": tbl, "epoch": epoch}, step=step)
        except Exception:
            pass  # ignore logging error


# SFT 配置版本兼容
def make_sft_config_version_safe(
    cfg: Config, eff_seq_len: int, out_dir: Path
) -> SFTConfig:
    """
    Robust runtime fallback:
    - Prefer new key `eval_strategy`.
    - If TypeError complains about it, swap to `evaluation_strategy` and retry.
    - If it complains about `evaluation_strategy`, swap back to `eval_strategy`.
    """
    common = dict(
        output_dir=str(out_dir),
        num_train_epochs=cfg.epochs,
        per_device_train_batch_size=cfg.per_device_train_batch_size,
        per_device_eval_batch_size=cfg.per_device_eval_batch_size,
        gradient_accumulation_steps=cfg.gradient_accumulation_steps,
        learning_rate=cfg.lr,
        weight_decay=cfg.weight_decay,
        max_steps=-1,
        logging_steps=cfg.logging_steps,
        eval_steps=cfg.eval_steps,
        save_strategy=cfg.save_strategy,
        save_steps=cfg.save_steps,
        lr_scheduler_type=cfg.lr_scheduler_type,
        warmup_ratio=cfg.warmup_ratio,
        bf16=True,
        gradient_checkpointing=cfg.gradient_checkpointing,
        packing=cfg.packing,
        max_seq_length=eff_seq_len,
        save_total_limit=cfg.save_total_limit,
        optim=cfg.optim,
        report_to=(
            [cfg.report_to] if (cfg.report_to and cfg.report_to != "none") else []
        ),
        run_name=cfg.run_name,  # seed will be appended later
        push_to_hub=False,
    )

    # 1st attempt: new name
    kwargs = dict(common, eval_strategy=cfg.eval_strategy)
    try:
        return SFTConfig(**kwargs)
    except TypeError as e1:
        msg = str(e1)
        # If it failed due to eval_strategy being unexpected, try old name
        if "eval_strategy" in msg and "unexpected" in msg:
            kwargs_old = dict(common, evaluation_strategy=cfg.eval_strategy)
            try:
                return SFTConfig(**kwargs_old)
            except TypeError as e2:
                # If even old name is unexpected, re-raise first error for clarity
                raise e1
        # If it complained about evaluation_strategy (some wrappers rephrase), try new name explicitly
        if "evaluation_strategy" in msg and "unexpected" in msg:
            kwargs_new = dict(common, eval_strategy=cfg.eval_strategy)
            return SFTConfig(**kwargs_new)
        # Unknown TypeError: re-raise
        raise


# 按照seed进行train
def train_one_seed(cfg: Config, seed: int) -> Path:
    seed_all(seed)

    out_dir = Path(cfg.output_root) / f"{cfg.run_name}-s{seed}"
    out_dir.mkdir(parents=True, exist_ok=True)

    tok = build_tokenizer(cfg.model_id, cfg.max_length)
    model = load_model(cfg.model_id, cfg.use_4bit, _compute_dtype(cfg.compute_dtype))

    backend = select_attn_backend(model, cfg.use_flash_attn)
    print(f"[info] attn backend: {backend}")

    eff_seq_len = rope_and_seq_guard(model, tok, cfg.max_length)

    model.config.use_cache = False  # why: gradient checkpointing
    if cfg.use_4bit:
        model = prepare_model_for_kbit_training(
            model, use_gradient_checkpointing=cfg.gradient_checkpointing
        )
    elif cfg.gradient_checkpointing:
        model.gradient_checkpointing_enable()

    lora_cfg = LoraConfig(
        r=cfg.lora_r,
        lora_alpha=cfg.lora_alpha,
        lora_dropout=cfg.lora_dropout,
        bias="none",
        task_type="CAUSAL_LM",
        target_modules=[
            "q_proj",
            "k_proj",
            "v_proj",
            "o_proj",
            "gate_proj",
            "up_proj",
            "down_proj",
        ],
    )
    model = get_peft_model(model, lora_cfg)
    model.enable_input_require_grads()

    data_path = Path(cfg.data_json)
    if not data_path.exists():
        raise FileNotFoundError(f"未找到数据: {data_path}")

    raw = load_dataset("json", data_files=str(data_path))["train"]
    if cfg.eval_split and 0.0 < cfg.eval_split < 1.0:
        split = raw.train_test_split(test_size=cfg.eval_split, seed=seed)
        ds_train, ds_eval = split["train"], split["test"]
    else:
        ds_train, ds_eval = raw, None

    fmt_func = make_formatting_func(tok)

    sft_args = make_sft_config_version_safe(cfg, eff_seq_len, out_dir)
    sft_args.run_name = f"{cfg.run_name}-s{seed}"

    if _WANDB_AVAILABLE and cfg.report_to == "wandb":
        wandb.init(
            project=cfg.project,
            group=cfg.run_group,
            name=f"{cfg.run_name}-s{seed}",
            config={
                "seed": seed,
                "model_id": cfg.model_id,
                "epochs": cfg.epochs,
                "lr": cfg.lr,
                "weight_decay": cfg.weight_decay,
                "warmup_ratio": cfg.warmup_ratio,
                "max_length": eff_seq_len,
                "optim": cfg.optim,
                "attn_impl": backend,
                "use_4bit": cfg.use_4bit,
            },
            dir=str(out_dir),
        )

    trainer = SFTTrainer(
        model=model,
        tokenizer=tok,
        train_dataset=ds_train,
        eval_dataset=ds_eval,
        args=sft_args,
        formatting_func=fmt_func,
        dataset_text_field=None,
    )

    samples_cb = WandbSamplesCallback(
        project_on=(_WANDB_AVAILABLE and cfg.report_to == "wandb"),
        sample_rows=cfg.sample_rows,
    )

    trainer.train()

    samples_cb.log_samples(
        dataset=ds_eval if ds_eval is not None else ds_train,
        formatting_func=fmt_func,
        step=int(trainer.state.global_step or 0),
        epoch=float(trainer.state.epoch or 0.0),
    )

    trainer.save_model(str(out_dir))
    tok.save_pretrained(str(out_dir))

    if _WANDB_AVAILABLE and cfg.report_to == "wandb":
        wandb.finish()

    print(f"[OK] 模型/LoRA 已保存: {out_dir.resolve()}")
    return out_dir


def train_many(cfg: Config) -> None:
    for s in cfg.seeds:
        print(f"\n===== Seed {s} =====")
        train_one_seed(cfg, s)


if __name__ == "__main__":
    cfg = Config()
    train_many(cfg)
