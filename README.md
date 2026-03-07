<img width="336" height="62" alt="image" src="https://github.com/user-attachments/assets/5c0c24d0-edd1-46f9-adb0-73145e4953b5" /># Explanation-Guided Prediction of Multi-Dimensional Feedback Scores in Psychological Counseling Dialogues

# 背景
世界では約8⼈に1⼈が精神疾患を抱えている 「WHO, 2022」
しかし，カウンセラーは深刻に不⾜しており，養成が急務
- 現実のカウンセラーはフィードバックを得にくい
- 相談がうまくいかない → クライアントが再訪しない
- 相談がうまくいく → 問題が解決し再訪しない
よって，⾃動的・体系的なフィードバック⽀援の仕組みが必要

現在対話型AIが社会的に普及してきている
そして、LLMは心理カウンセリングの分野でその可能性に注目されている  
GPT-4によって生成された応答は適切性や共感性において人間の新人カウンセラーと同等の評価を受けった「Inaba+2024」

## 実際の環境では
- フィードバックの仕組みが不足している 。
  - LLMを用いた心理カウンセリングの品質を客観的に評価し改善することが困難  

LLMは人間らしい対話や感情的なサポートを生成する能力があるが  
その予測プロセスは不透明
  - なぜ特定のフィードバックスコアが算出されたのか
  - そして、根拠も不明瞭	モデルの判断を理解し、信頼性を高めることが難しい

# 目的
- 先⾏研究では，評価付きカウンセリング対話
KokoroChat を収集し，スコア予測モデルを構築「Qi+ 2025」
- 課題
  - スコアだけでは，「なぜその評価になったのか」が分からず，改善が難しい
  - 精度⾃体も⼗分ではない
- 特徴・利点
 	- スコアの根拠を明⽰し，カウンセラーが結果を理解・改善しやすくなる
 	- CoT（Chain-of-Thought）形式の推論により，スコア予測の精度向上が期待できる

# 提案アプローチ: 説明誘導型スコア予測
## 評価理由の説明⽂を⽣成した上でスコアを予測する⼿法
![評価理由の説明⽂を⽣成した上でスコアを予測する⼿法](/Asset/Approach.png)
- Step 1
GPT-5に対し，カウンセリング対話履歴と20項⽬のクライアント
評価スコアを⼊⼒• 各スコアに対応する理由説明⽂を⽣成
- Step 2
Step1で構築した説明⽂付き
データを⽤い，CoT形式でLLMを学習
  - ⼊⼒	対話履歴
  - 出⼒  「理由 -> スコア」の順で，20項⽬について予測
 
# 実験
-  実験1: ⾃動評価実験
  - ⽬的: 説明⽂がスコア予測精度の向上に寄与するかを検証
  - 評価指標: KokoroChatと同⼀の指標を使⽤ [Qi+2025]
    - ACC 正解スコアとの⼀致率
    - ACCsoft ±1点差までを許容する柔軟な⼀致率
    - MAE 平均絶対誤差
## 結果
|   | Accuracy(↑) | Soft Accuracy(↑) | MAE(↓) |
| ------------- | ------------- | ------------- | ------------- |
| Llama-3.1   | 0.287  |  0.7253  |  1.0540  |
| Paper  | 0.3535  | 0.8364  | 0.8283  |
| Overall (CoT)   | 0.346  |  0.815  |  0.874  |
-  CoT　ours
-  Paper [Qi+2025] Llama-3.1 Fine-tuning

###  ACC
![ACC](/Asset/CoT_vs_Paper_ACC.png)
-  Modelは**D11 相談開始の円滑さ** に対して、予測精度が一番高い
-  **D6 一緒に考えながら取り組めた** に対して、予測精度が一番低い

###  ACCsoft & MAE
![ACCsoft](/Asset/CoT_vs_Paper_ACCsoft.png)
![MAE](/Asset/CoT_vs_Paper_MAE.png)
```
ACCsoft (↑): D6 | |Δ|=6.01 | Δ(CoT-Paper)=-6.01 | CoT=77.7 | Paper=83.71
MAE (↓): D6 | |Δ|=0.1464 | Δ(CoT-Paper)=0.1464 | CoT=0.971 | Paper=0.8246
 
=== Max gap dimension overall (sum of abs gaps across 3 metrics) ===
D6: sum(|Δ|)=12.1964
per-metric |Δ|={'ACC (↑)': 6.039999999999999, 'ACCsoft (↑)': 6.009999999999991, 'MAE (↓)': 0.14639999999999997}
```

###  予測例分析

-  例①：正解との完全一致

| Field | Content |
| ------------- | ------------- |
| Label | 新しい気づきや体験があった |
| Comment |「感謝される」「楽しいと感じる」要素の言語化や、安定とやりがいの両立という視点の提示は新鮮でした。加えて、価値観の棚卸しや強みの再確認、職種の仮説立て（医療関連・対人支援・事務・コミュニケーション中心など）まで踏み込むと、さらに発見が深まります。 |
| Output score | 3 |
| Reference | 3 |

EGSP の予測スコアが正解スコアと一致
モデルはスコアの根拠を適切に言語化でき
実際の評価プロセスをシミュレートしている


- 例②：わずかなズレを含む解釈
一部項目で予測スコアと人手ラベルに1点のズレあり

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

- [1] カウンセリング行動の理解は深い
- [2] 評価根拠は主に明示的な言語表現に基づいており、「居心地のよいやりとりだった」主観的なフィードバックには十分に捉えきれていない可能性がある

##  Case Study
### 予測スコア vs 正解スコア
![output_scoreVSreference](/Asset/output_scoreVSreference.png)
- 正解スコアが「0」の場合、EGSPは一度も「0」を出力しなかった
- 多くが「2〜4」の中間スコアに予測

### 予測スコア= 0

| Field | Content |
| ------------- | ------------- |
| Label | 問題の明確化 |
| Comment | 「趣味がないことへの罪悪感はないが、周囲との会話で困る」という構図が明確になりました。さらに、会話の場面（街コン、履歴書、日常）ごとに課題を切り分けると、より明瞭になります。 |
| Output score | 4 |
| Reference | 0 |

-  課題を把握しているが、スコアが高い
-  実際には問題の明確化が不十分だったため、正解は0点
-  しかしモデルは一部のポジティブな発言に注目し、過度に好意的な説明を生成


# これからやること
- [ ] 実験2: ⼈間評価実験
  - ⽬的: 説明⽂がカウンセラーの理解促進・改善⽀援に役⽴つかを検証
  - 評価条件
    - スコアのみ提⽰
    - スコア ＋ 理由説明⽂を提⽰
    - 被験者は2条件で主観的評価を実施

# 	関連研究
- [Can Large Language Models be Used to Provide Psychological Counselling? An Analysis of GPT-4-Generated Responses Using Role-play Dialogues](https://arxiv.org/abs/2402.12738)  
同一状況下での人間カウンセラーの応答とGPT-4が生成した応答の適切性を評価した
- [Understanding Client Reactions in Online Mental Health Counseling](https://aclanthology.org/2023.acl-long.577/)  
実際のオンライン相談記録を用いて、部分的なクライアント評価が付与されている
- [KokoroChat: A Japanese Psychological Counseling Dialogue Dataset Collected via Role-Playing by Trained Counselors.](https://aclanthology.org/2025.acl-long.608/)  
人手収集による日本語心理相談データセット
- [ESCoT: Towards Interpretable Emotional Support Dialogue Systems](https://aclanthology.org/2024.acl-long.723/)  
説明駆動型評価フレームワーク

# 	研究実績
##  国内学会（査読なし）
- **Yueliang Liu**, Zhiyang Qi, Michimasa Inaba: [Explanation-Guided Prediction of Multi-Dimensional Feedback Scores in Psychological Counseling Dialogues.](https://aclanthology.org/2024.acl-long.723/)
 第105回言語・音声理解と対話処理研究会(第16回対話システムシンポジウム), 人工知能学会研究会資料言語・音声理解と対話処理研究会, Vol.105, pp.7-11, 2025.
