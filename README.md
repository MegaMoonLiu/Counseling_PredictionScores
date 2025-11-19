# Explanation-Guided Prediction of Multi-Dimensional Feedback Scores in Psychological Counseling Dialogues

# 研究背景
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

# 研究の目的
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

 # これからやること
- [ ] 実験1: ⾃動評価実験
  - ⽬的: 説明⽂がスコア予測精度の向上に寄与するかを検証
  - 評価指標: KokoroChatと同⼀の指標を使⽤ [Qi+2025]
    - ACC 正解スコアとの⼀致率
    - ACCsoft ±1点差までを許容する柔軟な⼀致率
    - MAE 平均絶対誤差
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
