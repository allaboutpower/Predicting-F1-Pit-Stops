# 🏎️ F1 Pit Stop Prediction — Ensemble & EDA

Kaggle Playground Series S6E5 競賽解題 Notebook，目標為預測 F1 賽車下一圈是否進站（`PitNextLap`），評估指標為 **ROC-AUC**。

---

## 📁 Project Structure

```
├── f1-pit-stops-ensemble_and_EDA.ipynb   # 主要 Notebook
├── submission.csv                         # 最終預測結果
└── {model_name}_oof.csv / _test.csv      # 各模型 OOF & Test 預測輸出
```

---

## 🔧 Environment & Dependencies

```bash
pip install scikit-learn==1.7.2 xgboost==3.1.3 lightgbm==4.6.0 catboost==1.2.8 pytabkit
```

主要套件：`pandas` / `numpy` / `matplotlib` / `seaborn` / `optuna` / `keras` / `torch`

---

## 📊 Pipeline Overview

### 1. Configuration (`Config`)
- 載入 train / test / 原始外部資料集
- 設定 task type（binary classification）、n_splits=5、metric=ROC-AUC
- 自動偵測 GPU / CPU 環境

### 2. EDA (`EDA`)
- **`data_info()`** — 顯示 train/test head、info、describe、缺失值統計
- **`heatmap()`** — Spearman 相關係數熱力圖
- **`dist_plots()`** — 數值特徵 KDE + BoxPlot（Train vs Test 分布比較）
- **`cat_feature_plots()`** — 類別特徵 top-10 頻率長條圖
- **`target_pie()`** — 目標變數類別比例圓餅圖

### 3. Preprocessing (`Preprocessing`)
- **原始資料 Target Statistics** — 從外部資料集計算每個特徵的目標均值，作為統計特徵
- **Feature Engineering**
  - 比率特徵：`TyreLife / LapNumber`、`LapNumber / RaceProgress`
  - 交乘特徵：`LapTime × Cumulative_Degradation`（含絕對值與比率版本）
  - 組合特徵：`Race_Year_comb`、`Race_Compound_comb`
  - Binning：`RaceProgress`（200 bins）、`LapTime`（7 bins），策略為 quantile
  - 數值轉類別：所有數值欄位取 floor 後 factorize 為 category
- **Frequency Encoding** — 計算所有類別與數值轉類別欄位的出現頻率
- **Target Encoding**（fold 內動態執行）— 於每個 fold 的 training split 對類別特徵進行 TargetEncoder

### 4. Models

採用多模型異質集成（Stacking），包含以下基模型：

| 類型 | 模型 |
|------|------|
| Gradient Boosting | XGBoost（多組超參）、LightGBM（多組超參）、CatBoost（多組超參） |
| Tree-based | HistGradientBoosting、YDF GBT |
| Neural Network | Keras MLP、RealMLP（pytabkit）、TabM-D（pytabkit） |
| Linear | Logistic Regression |

超參數透過 **Optuna** 調整（TPE sampler + Hyperband pruner）。

### 5. Training (`Trainer`)
- **StratifiedKFold（n=5）** 交叉驗證
- 每個 fold 合併外部原始資料到 training set 增加樣本
- 各模型依類型分別處理輸入格式（DMatrix / Pool / Dataset / DataFrame）
- 輸出每個模型的 OOF 預測與 Test 預測並存檔

### 6. Ensemble（Stacking）
- 將各模型 OOF 預測透過 **logit 轉換**後，以 **Logistic Regression** 作為 meta-model 學習加權
- 輸出最終 ensemble OOF score 與 test 預測

### 7. Evaluation Plots
- ROC Curve（One-vs-Rest）
- Confusion Matrix
- Precision-Recall Curve
- Calibration Curve
- 各模型 Score 橫向比較長條圖

### 8. Submission
- 輸出 `submission.csv`
- CDF & KDE 分布圖比對 train OOF 與 test 預測的一致性
- 以下是不同模型輸出分數的差別：
![Model Score](https://github.com/allaboutpower/Predicting-F1-Pit-Stops/raw/main/images/modelScore)
---

## 📈 Metric

**ROC-AUC（binary classification）**

---

## 📝 Notes

- 訓練時 `training=False` 代表直接從已存檔的 OOF/test csv 載入，跳過重新訓練
- CatBoost 與 XGBoost 需 GPU（`task_type="GPU"` / `device="cuda"`）
- Apple Silicon（M1/M2）環境建議關閉 GPU 相關設定並改用 CPU
