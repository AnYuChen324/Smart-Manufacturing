#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Oct 11 10:36:20 2025

@author: anyuchen
"""

"""
配置檔：統一管理所有路徑、參數、常數
"""

import os
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier

#  ==========     路徑設定     ==========
BASE_PATH = "/Users/anyuchen/Desktop/程式語言/智慧製造"
DATA_PATH = os.path.join(BASE_PATH, "manufacturing_AI_dataset.csv")
GRAPH_PATH = os.path.join(BASE_PATH, "graph")

#  確保輸出資料夾存在，不會重複創建
os.makedirs(GRAPH_PATH, exist_ok=True)


#  ==========     資料欄位定義     ==========
# 類別欄位
CATEGORY_COL = ['Operation_Mode', 'Efficiency_Status', 'Machine_ID']

# 數值欄位
NUM_COL = [
    'Temperature_C', 'Vibration_Hz', 'Power_Consumption_kW',
    'Network_Latency_ms', 'Packet_Loss_%',
    'Quality_Control_Defect_Rate_%', 'Production_Speed_units_per_hr',
    'Predictive_Maintenance_Score', 'Error_Rate_%'
]

# 衍生特徵（diff/rolling/std）
DERIVED_COL = [
    'Speed_diff', 'Error_diff',
    'Speed_ma3', 'Error_ma3',
    'Speed_std3', 'Error_std3'
]

# 模型輸入特徵（不包含 Machine_ID）
NUM_COL_ALL = DERIVED_COL + NUM_COL

CATEGORY_COL_ALL = ['Operation_Mode']


#  ==========     模型參數     ==========
#GridSearch CV 參數網格
GRIDSEARCH_PARAMS = {
    'Logistic Regression': {
        'model':LogisticRegression(class_weight='balanced', max_iter=1000, solver='lbfgs'),
        'params':{
            'C': [0.01, 0.1, 1],
            'penalty': ['l2'],
            'solver': ['lbfgs'],
            'class_weight': [None, 'balanced']
        }
    },
    'Random Forest': {
        'model': RandomForestClassifier(),
        'params':{
            'max_depth': [10, 20],
            'min_samples_split': [2, 5],
            'min_samples_leaf': [1, 2],
            'max_features': ['sqrt']
        }
    },
    'XGBoost': {
        'model':None,   # 會在 model_training.py 裡創建,因為它需要n_classes
        'params':{
            'max_depth': [3, 5, 7],
            'learning_rate': [0.01, 0.1, 0.3],
            'n_estimators': [50, 100],
            'subsample': [0.6, 0.8, 1.0],
            'colsample_bytree': [0.6, 0.8, 1.0],
        }
    },
    'LightGBM': {
        'model': None,  # 會在 model_training.py 裡創建,因為它需要n_classes
        'params':{
            'n_estimators': [50, 100],
            'learning_rate': [0.01, 0.05, 0.1]
        }
    }
}

# PSO參數 learning_rate, n_estimators, max_depth, num_leaves
PSO_LOWER_BOUNDS = [0.01, 50, 3, 10]      
PSO_UPPER_BOUNDS = [0.3, 500, 15, 100]

PSO_CONFIG = {
    'swarmsize': 20,
    'maxiter': 10
}


# ==================== 隨機種子 ====================
RANDOM_STATE = 24


# ==================== 訓練測試切分比例 ====================
TRAIN_TEST_SIZE = 0.8  # 80% train, 20% test


# ==================== CV 設定 ====================
CV_SPLITS = 3  # TimeSeriesSplit 的 fold 數


# ==================== 其他參數 ====================
# EDA 繪圖顏色
EARTH_COLORS = ["#6B4226", "#C19A6B", "#8C6A43", "#BFA6A0", "#D2C6B2"]

# 特徵衍生參數
ROLLING_WINDOWS = 3  # rolling mean/std 的窗口大小

# TSNE 採樣大小（降低計算成本）
TSNE_SAMPLE_SIZE = 50000

