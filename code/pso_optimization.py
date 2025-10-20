#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Oct 13 10:12:48 2025

@author: anyuchen

pso_optimization_universal.py：通用 PSO 優化模組
支援所有模型類型的超參數優化

"""

import numpy as np
import pandas as pd
from pyswarm import pso
from sklearn.preprocessing import LabelEncoder
from sklearn.pipeline import Pipeline
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import f1_score
import lightgbm as lgb
from sklearn.metrics import f1_score
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier


from config import (
    PSO_LOWER_BOUNDS, PSO_UPPER_BOUNDS, PSO_CONFIG,
    RANDOM_STATE, CV_SPLITS, NUM_COL_ALL, CATEGORY_COL_ALL
)


# ==================== 定義各模型的參數空間 ====================
PSO_PARAM_SPACES = {
    'LightGBM':{
        'params':['learning_rate', 'n_estimators', 'max_depth', 'num_leaves'],
        'lower_bounds':[0.01, 50, 3, 10],
        'upper_bounds':[0.3, 500, 15, 100],
        'int_params':[1, 2, 3]   # 哪些參數需要轉成整數（索引）
    },
    'XGBoost':{
        'params':['learning_rate', 'n_estimators', 'max_depth', 'subsample'],
        'lower_bounds':[0.01, 50, 3, 0.5],
        'upper_bounds':[0.3, 500, 15, 1.0],
        'int_params':[1, 2]
    },
    'Random Forest': {
        'params': ['n_estimators', 'max_depth', 'min_samples_split', 'min_samples_leaf'],
        'lower_bounds': [50, 5, 2, 1],
        'upper_bounds': [500, 30, 20, 10],
        'int_params': [0, 1, 2, 3]  # 全部都是整數
    },
    'Logistic Regression': {
        'params': ['C', 'max_iter'],
        'lower_bounds': [0.001, 100],
        'upper_bounds': [100, 2000],
        'int_params': [1]  # max_iter 是整數
    }    
}


# ==================== PSO Fitness Function ====================
def universal_pso_fitness(params, model_name, X, y, n_splits=CV_SPLITS):
    """
    LightGBM 的適應度函式（PSO 會最小化這個函式）
    通用的適應度函式，支援所有模型
    
    PSO 的工作原理：
        1. 初始化一群粒子（隨機參數組合）
        2. 每個粒子評估目前位置的適應度
        3. 粒子向最佳位置移動（平衡全局與局域搜尋）
        4. 重複步驟 2-3 直到收斂或達到迭代次數上限
    
    為什麼要寫 fitness function？
        - GridSearchCV 只會嘗試你指定的參數組合
        - PSO / GA 能自動搜尋連續參數空間
        - 但 PSO 是個黑盒優化器，不知道什麼是 cross-validation
        - 所以你必須自己寫 fitness function 來告訴 PSO：
          「給定這組參數，模型表現如何？」
    
    參數：
        params (tuple): PSO 傳入的參數
        model_name (str): 模型名稱
        X (pd.DataFrame): 訓練特徵
        y (np.array): 訓練目標變數
        n_splits (int): CV 折數
    
    回傳：
        float: 負的平均 F1 分數（PSO 最小化）
    """
    
    # 取得參數配置
    param_config = PSO_PARAM_SPACES[model_name]
    param_names = param_config['params']
    int_indices = param_config['int_params']
    
    # 轉換參數(浮點數 -> 整數)
    params_list = list(params)
    for idx in int_indices:
        params_list[idx] = int(params_list[idx])
        
    #建立參數字典
    param_dict = dict(zip(param_names, params_list))
    
    print(f"\n測試參數： {param_dict}")
    
    # 根據模型名稱初始化模型
    if model_name == 'LightGBM':
        model = lgb.LGBMClassifier(
            **param_dict,
            objective = 'multiclass',
            class_weight = 'balanced',
            random_state = RANDOM_STATE,
            verbose = -1
        )
    
    elif model_name == 'XGBoost':
        n_classes = len(np.unique(y))
        model = XGBClassifier(
            **param_dict,
            objective = 'multi:softprob',
            num_class = n_classes,
            eval_metric = 'mlogloss',
            random_state = RANDOM_STATE,
            verbosity = 0
        )
        
    elif model_name == 'Random Forest':
        model = RandomForestClassifier(
            **param_dict,
            random_state = RANDOM_STATE)
        
    elif model_name == 'Logistic Regression':
        model = LogisticRegression(
            **param_dict,
            class_weight = 'balanced',
            random_state = RANDOM_STATE
        )
        
    else:
        raise ValueError(f"不支援的模型：{model_name}")
    
    #時序切分，交叉驗證
    tscv = TimeSeriesSplit(n_splits=n_splits)
    scores = []
    
    for fold, (train_idx, val_idx) in enumerate(tscv.split(X, y),1):
        # train_idx:訓練用的資料索引、val_idx:驗證用的資料索引
        X_tr, X_val=X.iloc[train_idx], X.iloc[val_idx]
        y_tr, y_val=y[train_idx], y[val_idx]
        
        # 模型評估/訓練
        model.fit(X_tr, y_tr)
        
        # 預測
        y_pred = model.predict(X_val)
        score = f1_score(y_val, y_pred, average='macro')
        
        print(f"  Fold {fold} F1-score: {score:.4f}")
        scores.append(score)
        
    mean_score = np.mean(scores)
    print(f"  平均 F1-score: {mean_score:.4f}")

    # PSO 是最小化問題，所以回傳負值
    return -mean_score



# ==================== PSO 最優化主函式 ====================
def optimize_model_with_pso(model_name, X_train, y_train, swarmsize=20, maxiter=10):
    """
    使用 PSO 優化任何模型的超參數
    
    參數：
        model_name (str): 模型名稱
        X_train (pd.DataFrame): 訓練特徵
        y_train (np.array): 訓練目標變數
        swarmsize (int): 粒子群大小
        maxiter (int): 最大迭代次數
    
    回傳：
        dict: 最佳參數字典
    """
    
    if model_name not in PSO_PARAM_SPACES:
        raise ValueError(f"模型 {model_name}尚未支援 PSO 優化")
    
    print("\n" + "="*60)
    print(f"開始 PSO 最優化:{model_name}")
    print("="*60)
    
    # 取得參數空間配置
    param_config = PSO_PARAM_SPACES[model_name]
    param_names = param_config['params']
    lower_bounds = param_config['lower_bounds']
    upper_bounds = param_config['upper_bounds']
    int_indices = param_config['int_params']

    print(f"\n搜尋參數空間...")
    for name, lb, ub in zip(param_names, lower_bounds, upper_bounds):
        print(f"{name}:[{lb},{ub}]")
    print(f"  粒子數：{swarmsize}")
    print(f"  迭代次數：{maxiter}")
  

    # 執行PSO
    best_params, best_score=pso(
        lambda p: universal_pso_fitness(p, model_name, X_train, y_train),
        lower_bounds,
        upper_bounds,
        swarmsize = swarmsize,
        maxiter = maxiter
    )
    
    # 轉換參數(浮點數 -> 整數)
    best_params_list = list(best_params)
    for idx in int_indices:
        best_params_list[idx] = int(best_params_list[idx])
    
    #  將結果包成字典dict
    best_params_dict = dict(zip(param_names, best_params_list))
    
    
    print("\n" + "="*60)
    print("PSO 最優化完成")
    print("="*60)
    print("\n最佳參數:")
    for k, v in best_params_dict.items():
        print(f"  {k}: {v}")
    print(f"\n最佳 Macro F1 分數: {-best_score:.4f}")
    
    return best_params_dict


# ==================== 快速介面：自動判斷是否執行 PSO ====================
def auto_potimize_model(model_name, X_train, y_train, enable_pso=True, **pso_kwargs):
    """
    自動判斷是否對模型執行 PSO 優化
    
    參數：
        model_name (str): 模型名稱
        X_train (pd.DataFrame): 訓練特徵
        y_train (np.array): 訓練目標變數
        enable_pso (bool): 是否啟用 PSO 優化
        **pso_kwargs: PSO 的額外參數（swarmsize, maxiter）
    
    回傳：
        dict or None: 最佳參數字典（如果執行 PSO），否則 None
    """

    if not enable_pso:
        print(f"\n PSO優化已停用，{model_name}使用預設參數")
        return None
    
    if model_name not in PSO_PARAM_SPACES:
        print(f"\n {model_name}尚未支援 PSO 優化，將使用預設參數")
        return None
    
    print(f"\n 執行 PSO 優化：{model_name}")
    
    # 設定預設值
    swarmsize = pso_kwargs.get('swarmsize',20)
    maxiter = pso_kwargs.get('maxiter',10)

    return optimize_model_with_pso(
        model_name, X_train, y_train, 
        swarmsize=swarmsize, 
        maxiter=maxiter
    )


# ==================== 批量優化多個模型 ====================
def optimize_all_models(model_names, X_train, y_train, **pso_kwargs):
    """
    批量優化多個模型
    
    參數：
        model_names (list): 模型名稱列表
        X_train (pd.DataFrame): 訓練特徵
        y_train (np.array): 訓練目標變數
        **pso_kwargs: PSO 參數
    
    回傳：
        dict: {模型名稱: 最佳參數字典}
    """

    results = {}
    
    for model_name in model_names:
        if model_name in PSO_PARAM_SPACES:
            best_params = optimize_model_with_pso(
                model_name, X_train, y_train, **pso_kwargs
            )
            
            results[model_name] = best_params
        else:
            print(f"\n 跳過{model_name} (尚未支援PSO)")
            results[model_name] = None
            
    return results

