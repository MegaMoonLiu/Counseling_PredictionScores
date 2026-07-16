# Explanation-Guided Prediction of Multi-Dimensional Feedback Scores in Psychological Counseling Dialogues

<p align="center">
  <a href="README.md"><img alt="日本語" height="20" src="https://img.shields.io/badge/日本語-CDCFD4"></a>&nbsp;
  <a href="README_zh.md"><img alt="简体中文" height="20" src="https://img.shields.io/badge/简体中文-BCDCF7"></a>&nbsp;
</p>

# 研究背景
全球约有八分之一的人患有精神疾病「WHO, 2022」
然而，心理咨询师面临严重匮乏，其培养工作迫在眉睫：
- 现实中的咨询师难以获得反馈
    - 咨询不顺利 → 来访者不再复诊
    - 咨询很顺利 → 问题得到解决，来访者同样不再复诊

因此，迫切需要一种自动且系统化的反馈支持机制。

当前，对话型 AI 正在社会中普及，且大语言模型（LLM）在心理咨询领域的潜力也备受瞩目。

研究表明，GPT-4 生成的回复在合适性与共情性上，已获得了与人类新手咨询师同等的评价「Inaba+2024」

## 实际环境中的挑战
- 反馈机制的缺失
  - 难以客观评估并改善基于 LLM 的心理咨询质量  

尽管 LLM 具备生成拟人化对话和情感支持的能力

但其预测过程并不透明 

  - 无法得知为何会计算出特定的反馈评分
  - 评分依据也模糊不清，导致难以理解模型的决策，进而难以提高其信誉度

# 研究目的
- 先前研究
收集了带有评估标签的咨询对话数据集“KokoroChat”，并构建了评分预测模型「Qi+ 2025」
- 现有课题
  - 仅给出评分无法让人得知“为何会得到该评价”，导致难以针对性地改善咨询质量
  - 预测精度本身仍有待提升
- 本研究的特征与优势
 	- 明确评分依据：使咨询师更易于理解评估结果并进行改进
 	- 引入思维链（CoT, Chain-of-Thought）推理：有望进一步提升评分预测的准确性

# 提案方法：解释引导型评分预测
## 先生成评估理由说明，再进行评分预测的方法
![先生成评估理由说明，再进行评分预测的方法](/Asset/Approach.png)
- Step 1
向 GPT-5 输入咨询对话历史以及 20 个维度的来访者评估评分
生成各评分所对应的理由说明文本
- Step 2
利用Step1中构建的、带有说明文本的数据
以 CoT（思维链） 的形式对 LLM 进行微调
  - 输入	对话历史
  - 输出  按照“理由->评分”的顺序，对 20 个评估维度进行预测。
 
# 实验
-  实验 1：自动评估实验
  - 目的：验证评估说明文本是否有助于提高评分预测的精度
  - 评估指标：使用与 KokoroChat 相同的评估指标[Qi+2025]
    - ACC 与真实评分的完全一致率
    - ACCsoft 允许 $\pm 1$ 分误差的宽容一致率
    - MAE 预测值与真实值之间的绝对误差平均值
## 结果
|   | Accuracy(↑) | Soft Accuracy(↑) | MAE(↓) |
| ------------- | ------------- | ------------- | ------------- |
| Llama-3.1   | 0.287  |  0.7253  |  1.0540  |
| Paper  | 0.3535  | 0.8364  | 0.8283  |
| Overall (CoT)   | 0.346  |  0.815  |  0.874  |
-  CoT　Ours
-  Paper [Qi+2025] Llama-3.1 Fine-tuning

###  ACC
![ACC](/Asset/CoT_vs_Paper_ACC.png)
-  Model在**D11 相談開始の円滑さ** 中、预测精度最高
-  在**D6 一緒に考えながら取り組めた** 中、预测精度最低

###  ACCsoft & MAE
![ACCsoft](/Asset/CoT_vs_Paper_ACCsoft.png)
![MAE](/Asset/CoT_vs_Paper_MAE.png)
```
ACCsoft (↑): D6 | |Δ|=6.01 | Δ(CoT-Paper)=-6.01 | CoT=77.7 | Paper=83.71
MAE (↓): D6 | |Δ|=0.1464 | Δ(CoT-Paper)=0.1464 | CoT=0.971 | Paper=0.8246
 
=== Max gap dimension overall (sum of abs gaps across 3 metrics) ===
D6: sum(|Δ|)=12.1964
per-metric
|Δ|ACC(↑)=6.039999999999999
|Δ|ACCsoft(↑)=6.009999999999991
|Δ|MAE(↓)=0.14639999999999997
```

###  预测实例分析

-  例①：与正解完全一致

| Field | Content |
| ------------- | ------------- |
| Label | 新しい気づきや体験があった |
| Comment |「感謝される」「楽しいと感じる」要素の言語化や、安定とやりがいの両立という視点の提示は新鮮でした。加えて、価値観の棚卸しや強みの再確認、職種の仮説立て（医療関連・対人支援・事務・コミュニケーション中心など）まで踏み込むと、さらに発見が深まります。 |
| Output score | 3 |
| Reference | 3 |

EGSP 的预测评分与真实评分一致

模型能够将评分的依据进行恰当的文本化表达

模型模拟了真实的评估流程


- 例②：包含轻微偏差的解读
在部分评估维度中，预测评分与人工标注（真实标签）之间存在 1 分的偏差

| Field | Content |
| ------------- | ------------- |
| Label | やりとりのリズムがあっていた |
| Comment | 短い相槌や連投が続き、質問が重なる場面があり[1]、流れが途切れがちでした。[1] メッセージ 1 要点で、区切りごとに要約→確認→次の質問の順で進めるとリズムが整います。 |
| Output score | 2 |
| Reference | 3 |
| Label | 居心地のよいやりとりだった |
| Comment | 安心して話せる雰囲気がありました。提案の前に「この方向で進めても大丈夫？」[2]と都度合意を取ると、さらに居心地が良くなります。 |
| Output score | 4 |
| Reference | 3 |

- [1] 对咨询行为具有较深的理解
- [2] 评估依据主要基于显性的语言表达，可能未能充分捕获诸如「居心地のよいやりとりだった」等偏主观的反馈

##  Case Study
### 预测分数 vs 真实评分
![output_scoreVSreference](/Asset/output_scoreVSreference.png)
- 当真实评分为「0」时、EGSP 一次也未输出「0」
- 模型大多预测为「2〜4」之间的中等评分

### 预测分数= 0

| Field | Content |
| ------------- | ------------- |
| Label | 問題の明確化 |
| Comment | 「趣味がないことへの罪悪感はないが、周囲との会話で困る」という構図が明確になりました。さらに、会話の場面（街コン、履歴書、日常）ごとに課題を切り分けると、より明瞭になります。 |
| Output score | 4 |
| Reference | 0 |

-  模型虽然识别到了问题所在，但给出的评分偏高
-  实际上由于问题澄清不够充分，真实评分应为 0 分
-  但模型关注到了部分积极的发言，从而生成了过度乐观/正向的评估说明


# 今后的工作
- [ ] 实验2: 人工评估实验
  - ⽬的: 验证评估说明文本是否能有效促进咨询师的理解并辅助其改善咨询质量
  - 评估条件
    - 仅展示评分
    - 展示评分 $+$ 理由说明文本
    - 受试者对上述两种条件进行主观评估

# 	関連研究
- [Can Large Language Models be Used to Provide Psychological Counselling? An Analysis of GPT-4-Generated Responses Using Role-play Dialogues](https://arxiv.org/abs/2402.12738)  
评估了在相同情境下人类咨询师的回复与 GPT-4 生成回复的合适性
- [Understanding Client Reactions in Online Mental Health Counseling](https://aclanthology.org/2023.acl-long.577/)  
采用真实的在线咨询记录，并附带了部分来访者的评估反馈
- [KokoroChat: A Japanese Psychological Counseling Dialogue Dataset Collected via Role-Playing by Trained Counselors.](https://aclanthology.org/2025.acl-long.608/)  
人工收集的日语心理咨询数据集
- [ESCoT: Towards Interpretable Emotional Support Dialogue Systems](https://aclanthology.org/2024.acl-long.723/)  
解释驱动型评估框架

# 	研究业绩
##  日本国内学会（无同行评审）
- **Yueliang Liu**, Zhiyang Qi, Michimasa Inaba: [Explanation-Guided Prediction of Multi-Dimensional Feedback Scores in Psychological Counseling Dialogues.](https://aclanthology.org/2024.acl-long.723/)
 第105回言語・音声理解と対話処理研究会(第16回対話システムシンポジウム), 人工知能学会研究会資料言語・音声理解と対話処理研究会, Vol.105, pp.7-11, 2025.
