#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Oct 17 10:19:25 2025

@author: anyuchen

final_model_train.py：統一的最終模型訓練模組
支援所有模型類型的訓練與評估
"""

import numpy as np
import pandas as pd
from sklearn.metrics import classification_report, accuracy_score, f1_score
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
import lightgbm as lgb

from config import RANDOM_STATE


def train_final_model(model_name, X_train, X_test, y_train, y_test, le, best_params=None):
    """
    統一的最終模型訓練函數
    
    參數：
        model_name (str): 模型名稱 ('LightGBM', 'XGBoost', 'Random Forest', 'Logistic Regression')
        X_train (pd.DataFrame): 訓練特徵（已編碼）
        X_test (pd.DataFrame): 測試特徵（已編碼）
        y_train (np.array): 訓練目標變數（已編碼）
        y_test (np.array): 測試目標變數（已編碼）
        le (LabelEncoder): 標籤編碼器
        best_params (dict, optional): 最佳參數（PSO 或 GridSearch 找到的）
    
    回傳：
        dict: 包含所有訓練結果的字典
            - 'model': 訓練好的模型
            - 'y_pred': 預測結果
            - 'y_prob': 預測機率
            - 'accuracy': 準確度
            - 'f1_macro': Macro F1 分數
            - 'best_params': 使用的參數
            - 'feature_names': 特徵名稱
            
    **為什麼在PSO要初始化，這邊訓練final_model也要初始化？**
            - 因為 PSO 階段的模型都被丟掉了
            - 您需要「重新建立一個模型」
            - 這次是用「完整資料」訓練，給正式使用
            
            **這個模型會保留嗎？**
            - ✅ 會！這是最終要用的模型
            - 使用者會拿這個模型去預測、部署
    """
    
    print("\n" + "="*70)
    print(f"訓練最終模型：{model_name}")
    print("="*70)
    
    
    # ===== 1. 根據模型名稱初始化模型 =====
    if model_name == 'LightGBM':
        # 合併預設參數與優化參數
        model_params = best_params if best_params is not None else {}

        final_model = lgb.LGBMClassifier(
            **model_params,  # 如果是 {}，就用預設值；如果有值，就用優化值
            objective = 'multiclass',
            class_weight = 'balanced',
            random_state = RANDOM_STATE,
            verbose = -1
        )
        
        if best_params is not None:
            print("使用 PSO 優化參數")
            for k, v in best_params.items():
                if k in ['learning_rate', 'n_estimators', 'max_depth', 'num_leaves']:
                    print(f"  {k}:{v}")        
        else:
            print("使用 LightGBM 預設參數")
                   
    elif model_name == 'XGBoost':
        n_classes = len(np.unique(y_train))
        model_params = best_params if best_params is not None else {}
        
        final_model = XGBClassifier(
            **model_params,
            eval_metric = 'mlogloss',
            objective = 'multiLsoftprob',
            num_class = n_classes,
            random_state = RANDOM_STATE,
            verbosity = 0
        )
        print("使用 XGBoost" + ("PSO 優化參數" if best_params else "預設參數"))
        
    elif model_name == 'Random Forest':
        model_params = best_params if best_params is not None else {}
        
        final_model = RandomForestClassifier(
            **model_params,
            random_state = RANDOM_STATE
        )
        print("使用 Random Forest" + (" PSO 優化參數" if best_params else " 預設參數"))
        
    elif model_name == 'Logistic Regression':
        model_params = best_params if best_params is not None else {}
        
        final_model = LogisticRegression(
            **model_params,
            class_weight = 'balanced',
            random_state = RANDOM_STATE
        )
        print("使用 Logistic Regression" + (" PSO 優化參數" if best_params else " 預設參數"))
    
    else:
        raise ValueError(f"不支援的模型：{model_name}")
        
    
    # ===== 2. 訓練模型 =====
    print("\n開始訓練模型...")
    final_model.fit(X_train,y_train)
    print("✓ 訓練完成")
    
    # ===== 3. 模型預測 =====
    print("\n進行預測...")
    y_pred = final_model.predict(X_test)
    y_prob = final_model.predict_proba(X_test)
    
    # ===== 4. 模型評估 =====
    accuracy = accuracy_score(y_test, y_pred)
    f1_macro = f1_score(y_test, y_pred, average = 'macro')
    
    print(f"  準確度：{accuracy:.4f}")
    print(f"  Macro F1:{f1_macro:.4f}")
    
    print(f"\n詳細分類報告：")
    print(classification_report(y_test, y_pred, target_names=le.classes_))
    
    # ===== 5. 取得特徵名稱 =====
    feature_names = X_train.columns.tolist()
    
    # ===== 6. 取得最終使用的參數 =====
    final_params = final_model.get_params()
    
    # ===== 7. 包裝所有結果 =====
    results = {
        'model':final_model,
        'y_pred':y_pred,
        'y_prob':y_prob,
        'y_test':y_test,
        'accuracy':accuracy,
        'f1_macro':f1_macro,
        'best_params':final_params,
        'feature_names':feature_names,
        'model_name':model_name
    }
    
    print("\n" + "="*70)
    print("最終模型訓練完成！")
    print("="*70)
    
    return results
    
    
def check_model_support_feature_importance(model_name):
    """
    檢查模型是否支援特徵重要性分析
   
    參數：
       model_name (str): 模型名稱
   
    回傳：
       bool: True 表示支援，False 表示不支援
    """
    
    tree_models = ['LightGBM', 'XGBoost', 'Random Forest']
    return model_name in tree_models
    
    
def check_model_support_shap(model_name):
    """
    檢查模型是否支援 SHAP 分析
    
    參數：
        model_name (str): 模型名稱
    
    回傳：
        bool: True 表示支援，False 表示不支援
    """
    # SHAP支援的樹模型
    shap_models = ['LightGBM', 'XGBoost', 'Random Forest']
    return model_name in shap_models    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    





