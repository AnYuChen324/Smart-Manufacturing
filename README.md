# 智慧製造效能預測系統 

基於機器學習的製造業設備效能分類與預測系統，支援多模型比較、PSO 超參數優化、SHAP 可解釋性分析。

[![Python Version](https://img.shields.io/badge/python-3.8%2B-blue)](https://www.python.org/)
[![scikit-learn](https://img.shields.io/badge/scikit--learn-1.3%2B-orange)](https://scikit-learn.org/)
[![License](https://img.shields.io/badge/license-MIT-green)](LICENSE)

---

##  目錄

- [專案簡介](#專案簡介)
- [功能特色](#功能特色)
- [系統架構](#系統架構)
- [環境需求](#環境需求)
- [模組說明](#模組說明)
- [設定檔說明](#設定檔說明)
- [結果說明](#結果說明)
- [參數調整指南](#參數調整指南)
- [專案結構](#專案結構)

---

##  專案簡介

本專案針對製造業設備運行資料進行分析，預測設備效能的狀態（High/Medium/Low），並提供完整的機器學習管道，從資料預處理到最終模型部署。

### 核心功能

- **多模型自動比較**：Logistic Regression、Random Forest、XGBoost、LightGBM
- **智能超參數優化**：PSO（粒子群優化）支援所有模型
- **完整 EDA 分析**：統計檢定、相關性分析、降維視覺化
- **模型可解釋性**：SHAP 分析、特徵重要性排序
- **時間序列處理**：避免資料洩漏的正確切分方式

### 適用場景

- **製造業設備效能監控** 
- ** 預測性維護（Predictive Maintenance）**
- ** 生產效率優化  **
- ** 異常檢測與預警  **
- ** 工業 4.0 智能製造 ** 

---

##  功能特色

### 1️.智能資料預處理
-  **時間序列特徵衍生**：差分（捕捉瞬時變化）、滾動平均（趨勢平滑）、標準差（波動檢測）
-  **自動特徵編碼**：One-Hot Encoding（類別）+ StandardScaler（數值）
-  **資料洩漏防護**：先切分後編碼，符合機器學習最佳實踐
-  **機器分組處理**：確保時間序列特徵計算正確

### 2️.探索性資料分析（EDA）
-  **常態分布檢定**：Shapiro-Wilk Test(數值特徵在不同類別組之間是否有顯著差異)
-  **非參數檢定**：Kruskal-Wallis H Test（多組比較）
-  **卡方檢定**：檢測類別變數關聯性
-  **相關性分析**：Pearson 相關矩陣 + Heatmap
-  **多重共線性檢測**：VIF（方差膨脹因子）
-  **降維視覺化**：PCA、t-SNE

### 3️.模型訓練與優化
-  **快速模型篩選**：Logistic Regression、Random Forest、XGBoost、LightGBM 4 種模型快速比較，自動選出最佳模型(不使用GridSearch調參數)
-  **通用 PSO 優化**：支援 Logistic Regression、Random Forest、XGBoost、LightGBM
-  **時序交叉驗證**：TimeSeriesSplit 避免未來資料洩漏
-  **類別不平衡處理**：自動樣本權重調整

### 4️.模型評估與視覺化
-  **ROC 曲線**：多分類 Micro/Macro AUC
-  **PRC 曲線**：Precision-Recall + 最佳閾值標註
-  **混淆矩陣**：視覺化分類結果
-  **特徵重要性**：樹模型專用
-  **SHAP 分析**：模型可解釋性（Beeswarm Plot）

---

##  系統架構

```
智慧製造效能預測系統
│
├─ 資料層
│  ├─ data_loading.py        # 資料讀取、排序、資訊
│  └─ data_preprocessing.py  # 特徵工程、編碼、切分
│
├─ 分析層
│  ├─ eda.py                 # 探索性資料分析
│  └─ utils.py               # 統計工具（VIF、卡方檢定）
│
├─ 模型層
│  ├─ model_train.py         # 快速模型比較
│  ├─ pso_optimization.py    # PSO 超參數優化
│  └─ final_model_train.py   # 最終模型訓練
│
├─ 視覺化層
│  └─ visualization.py       # ROC、PRC、SHAP、特徵重要性
│
├─ 設定層
│  └─ config.py              # 統一參數管理
│
└─ 執行層
   └─ main.py                # 主程式進入點
```

---

##  環境需求

### Python 版本
- Python 3.8+

### 核心套件
```
numpy >= 1.21.0
pandas >= 1.3.0
scikit-learn >= 1.0.0
xgboost >= 1.5.0
lightgbm >= 3.3.0
matplotlib >= 3.4.0
seaborn >= 0.11.0
shap >= 0.40.0
pyswarm >= 0.6
scipy >= 1.7.0
statsmodels >= 0.13.0
pandasql >= 0.7.3
```

---

##  模組說明

### 1. `config.py` - 設定檔
統一管理所有路徑、參數、常數，包括：
- 資料路徑設定
- 特徵欄位定義
     ```python
    類別欄位：
    ['Operation_Mode', 'Efficiency_Status', 'Machine_ID']
    數值欄位：
    ['Temperature_C', 'Vibration_Hz', 'Power_Consumption_kW',
     'Network_Latency_ms', 'Packet_Loss_%','Quality_Control_Defect_Rate_%', 
     'Production_Speed_units_per_hr','Predictive_Maintenance_Score', 'Error_Rate_%']
    衍生欄位：
    ['Speed_diff', 'Error_diff',
     'Speed_ma3', 'Error_ma3',
     'Speed_std3', 'Error_std3']
- 模型參數空間
- PSO 設定
- 隨機種子與 CV 折數

### 2. `data_loading.py` - 資料載入
```python
load_manufacture_data()    # 讀取 CSV 並排序
display_basic_info(data)   # 顯示基本統計資訊
```

### 3. `data_preprocessing.py` - 資料預處理
```python
generate_derived_features(df)     # 生成衍生特徵
prepare_model_data(df)             # 分離 X, y
train_test_split_timeseries(X, y) # 依時序排列切分，前80％訓練、後20％測試
encode_features(X_train, X_test)  # 特徵編碼(OneHotEncoder+StandardScaler+LabelEncoder)
```

**重要概念：**
- **差分（diff）**：捕捉瞬時變化 `Production_Speed_diff`
- **滾動平均（rolling mean）**：平滑趨勢 `Speed_ma3`
- **滾動標準差（rolling std）**：檢測波動 `Speed_std3`

### 4. `eda.py` - 探索性資料分析
```python
run_eda(manufacture_data, X, y)  # 執行完整 EDA
```

包含：
- 類別特徵分布（長條圖）
- 常態分布檢定（Shapiro-Wilk）
- Kruskal-Wallis H 檢定
- 卡方檢定（類別關聯性）
- 相關性矩陣（Heatmap）
- VIF 多重共線性檢測
- PCA / t-SNE 降維視覺化

### 5. `model_train.py` - 模型訓練
```python
quick_model_compare(X_train, X_test, y_train, y_test)  # 快速比較 4 種模型
```

支援模型：
- Logistic Regression
- Random Forest
- XGBoost
- LightGBM

### 6. `pso_optimization.py` - PSO 優化
```python
optimize_model_with_pso(model_name, X_train, y_train)  # PSO 優化
auto_potimize_model(model_name, X_train, y_train)      # 自動判斷是否執行
```

**PSO 參數空間：**
| 模型 | 優化參數 |
|------|---------|
| LightGBM | learning_rate, n_estimators, max_depth, num_leaves |
| XGBoost | learning_rate, n_estimators, max_depth, subsample |
| Random Forest | n_estimators, max_depth, min_samples_split, min_samples_leaf |
| Logistic Regression | C, max_iter |

### 7. `final_model_train.py` - 最終模型訓練
```python
train_final_model(model_name, X_train, X_test, y_train, y_test, le, best_params)
```

統一的模型訓練介面，回傳包含：
- 訓練好的模型
- 預測結果與機率
- 評估指標（準確度、F1）
- 特徵名稱

### 8. `visualization.py` - 視覺化
```python
plot_multiple_roc(results, y_test)      # ROC 曲線
plot_prc_curve(results, y_test)         # PRC 曲線
plot_feature_importance(model)          # 特徵重要性
plot_shap_analysis(model, X_train)      # SHAP 分析
plot_confusion_matrix(y_test, y_pred)   # 混淆矩陣
```

### 9. `utils.py` - 工具函式
```python
calculate_vif(X)              # VIF 計算
chi2_test(target, features)   # 卡方檢定
get_sample_weights(y)         # 樣本權重
```

---

## ⚙️ 設定檔說明

### 路徑設定
```python
BASE_PATH = "/your/project/path"
DATA_PATH = os.path.join(BASE_PATH, "manufacturing_AI_dataset.csv")
GRAPH_PATH = os.path.join(BASE_PATH, "graph")
```

### 資料欄位定義
```python
# 數值特徵
NUM_COL = ['Temperature_C', 'Vibration_Hz', 'Power_Consumption_kW', ...]

# 類別特徵
CATEGORY_COL = ['Operation_Mode', 'Efficiency_Status', 'Machine_ID']

# 衍生特徵
DERIVED_COL = ['Speed_diff', 'Error_diff', 'Speed_ma3', ...]
```

### PSO 參數
```python
PSO_CONFIG = {
    'swarmsize': 20,   # 粒子數量
    'maxiter': 10      # 最大迭代次數
}
```

### 其他參數
```python
RANDOM_STATE = 24           # 隨機種子
TRAIN_TEST_SIZE = 0.8       # 訓練測試比例
CV_SPLITS = 3               # 交叉驗證折數
ROLLING_WINDOWS = 3         # 滾動窗口大小
```

---


##  結果說明

### 輸出檔案

執行完成後，會在 `GRAPH_PATH` 目錄生成以下檔案：

#### EDA 圖表
- `Machine_Count_Feature_Distribution.png`
- `Operation_Mode_Count_Feature_Distribution.png`
- `Efficiency_Status_Count_Feature_Distribution.png`
- `{feature}_distribution.png`（每個特徵的分布圖）
- `correlation_heatmap.jpg`
- `Feature_Space_PCA.png`
- `Feature_Space_TSNE.png`

#### 模型評估圖表
- `{model}_Final_ROC_ROC_curve.jpg`
- `{model}_Final_PRC_Precision_&_Recall_vs_Threshold_curve.jpg`
- `{model}_feature_importance.png`
- `{model}_Final_Shap_SHAP_Beeswarm_plot.png`
- `{model}_Final_Shap_SHAP_Feature_Importance.png`
- `{model}_confusion_matrix.png`

### 回傳變數

`main()` 函數回傳一個字典，包含：

```python
results = {
    'X_train': ...,              # 訓練特徵
    'X_test': ...,               # 測試特徵
    'y_train': ...,              # 訓練標籤
    'y_test': ...,               # 測試標籤
    'le': ...,                   # LabelEncoder 物件
    'preprocessor': ...,         # ColumnTransformer 物件
    'final_model': ...,          # 最終模型
    'best_model_name': ...,      # 最佳模型名稱
    'y_pred': ...,               # 預測結果
    'y_prob': ...,               # 預測機率
    'accuracy': ...,             # 準確度
    'f1_macro': ...,             # Macro F1 分數
    'feature_names': ...,        # 特徵名稱列表
    'best_params': ...,          # PSO 優化參數
    'feature_importance_df': ..., # 特徵重要性 DataFrame
    'shap_importance_df': ...    # SHAP 重要性 DataFrame
}
```

---

##  參數調整指南

### 1. 提升模型效能

**增加 PSO 搜尋強度：**
```python
# config.py
PSO_CONFIG = {
    'swarmsize': 30,    # 增加粒子數（預設 20）
    'maxiter': 20       # 增加迭代次數（預設 10）
}
```

**調整訓練測試比例：**
```python
# config.py
TRAIN_TEST_SIZE = 0.7  # 改為 70% 訓練，30% 測試
```

**增加交叉驗證折數：**
```python
# config.py
CV_SPLITS = 5  # 從 3 fold 增加到 5 fold
```

### 2. 加快執行速度

**跳過 EDA：**
```python
# main.py 中註解掉
# run_eda(manufacture_data, X_train, y_train)
```

**關閉 PSO：**
```python
enable_pso=False  # 使用預設參數
```

**減少資料量測試：**
```python
# main.py
manufacture_data = load_manufacture_data().head(1000)  # 只用 1000 筆
```

### 3. 調整特徵工程

**修改滾動窗口大小：**
```python
# config.py
ROLLING_WINDOWS = 5  # 從 3 改為 5
```

**新增自訂特徵：**
```python
# data_preprocessing.py 的 generate_derived_features 函數中
df['Custom_Feature'] = df['Temperature_C'] * df['Vibration_Hz']
```

---

##  專案結構

```
smart-manufacturing/
│
├── config.py                    # 設定檔
├── data_loading.py              # 資料載入
├── data_preprocessing.py        # 資料預處理
├── eda.py                       # 探索性資料分析
├── utils.py                     # 工具函式
├── model_train.py               # 模型訓練
├── pso_optimization.py          # PSO 優化
├── final_model_train.py         # 最終模型訓練
├── visualization.py             # 視覺化
├── main.py                      # 主程式
│
├── manufacturing_AI_dataset.csv # 資料檔案
├── requirements.txt             # 套件需求
├── README.md                    # 說明文件
│
└── graph/                       # 輸出圖表資料夾
    ├── EDA圖表/
    └── 模型評估圖表/
```

---

##  效能指標說明

| 指標 | 說明 | 適用情境 |
|------|------|---------|
| **Accuracy** | 整體準確率 | 類別平衡時使用 |
| **Macro F1** | 各類別 F1 平均 | 重視少數類別表現 |
| **Weighted F1** | 依樣本數加權 | 類別不平衡時使用 |
| **ROC-AUC** | 分類能力 | 閾值調整參考 |
| **PR-AUC** | 不平衡資料 | 少數類別評估 |

---

**⭐ 如果這個專案對您有幫助，請給個星星！**
