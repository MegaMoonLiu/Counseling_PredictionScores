# Explanation-Guided Prediction of Multi-Dimensional Feedback Scores in Psychological Counseling Dialogues

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
| Overall (CoT)   | 0.346  |  0.815  |  0.874  |
| Paper  | 0.3535  | 0.8364  | 0.8283  |
-  CoT　ours
-  Paper [Qi+2025]

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

###  Case Study
- Case item_6
```
 "evaluation_items_6": {
       "label": "一緒に考えながら取り組めた",
       "comment": "「一緒に考える」姿勢はありましたが、選択肢の比較検討や優先順位づけは限定的でした。複数案を並べて利点・懸念を一緒に見比べる進め方があると協働感が高まります。",
       "output_score": 3,
       "reference": 2
     },
 "evaluation_items_6": {
       "label": "一緒に考えながら取り組めた",
       "comment": "質問中心で進み、選択肢を並べて一緒に検討する場面が少なかったです。「今この瞬間、できそうなことを一緒に考えても良いですか？」と合意を取りながら進めると協働感が高まります。",
       "output_score": 1,
       "reference": 0
     },
 "evaluation_items_6": {
       "label": "一緒に考えながら取り組めた",
       "comment": "提案はありましたが、選択肢のメリット・デメリットを並べて一緒に検討するプロセスが弱く、共同作業の実感は限定的でした。",
       "output_score": 2,
       "reference": 5
     },
```
-  モデルは中立的なスコアを予測する傾向がある

- Case "reference": 0
```
"evaluation_items_16": {
  "label": "要約",
  "comment": "途中や終盤での要約がほとんどなく、論点が散在しました。「現状」「望む方向」「障壁」「次の一歩」を短くまとめるだけでも見通しが立ちます。",
  "output_score": 2,
  "reference": 0
     },
"evaluation_items_17": {
  "label": "問題の明確化",
  "comment": "「趣味がないことへの罪悪感はないが、周囲との会話で困る」という構図が明確になりました。さらに、会話の場面（街コン、履歴書、日常）ごとに課題を切り分けると、より明瞭になります。",
  "output_score": 4,
  "reference": 0
},
"evaluation_items_18": {
  "label": "この相談での目標の明確化",
  "comment": "「人と話せる程度の趣味を持つ」という方向性は共有できました。短期目標（例：次回までに1つ“話題にできる”候補を挙げる）を一緒に設定できると、より明確になります。",
  "output_score": 4,
  "reference": 0
},
"evaluation_items_19": {
  "label": "次の行動につながる提案",
  "comment": "「興味のある話に焦点を当てる」「質問は核心に近づかないようにする」など、すぐ試せる提案がありました。加えて、話題の切り替えフレーズや、話題が合わない時の返答例をいくつか提示すると、実践性がさらに高まります。",
  "output_score": 4,
  "reference": 0
}
```
-  "output_score": 0の回数はゼロ
- モデルは評価理由が生成したので、中立的なスコアを予測する傾向がある
- 学習データに"reference": 0の比率が低いので、モデルはそれについて、学習できなっかた
- これは正解率が低下している原因かもしれない



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
