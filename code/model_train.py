#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Oct 12 17:03:32 2025

@author: anyuchen
"""

"""
06_model_training.py：模型訓練與評估（使用 GridSearchCV）
"""

import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder, StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
import lightgbm as lgb
from sklearn.model_selection import GridSearchCV, TimeSeriesSplit
from sklearn.metrics import (
    classification_report, confusion_matrix, accuracy_score,
    roc_auc_score, f1_score, average_precision_score
)
from sklearn.preprocessing import label_binarize
from sklearn.utils.class_weight import compute_sample_weight

from config import (
    NUM_COL_ALL, CATEGORY_COL_ALL, GRIDSEARCH_PARAMS,
    CV_SPLITS, RANDOM_STATE
)
from utils import get_sample_weights


# ==================== 模型訓練主函式(不執行GridSearch,不調參,不驗證) ====================
#  五個模型快速比較（不調參、不交叉驗證）
#  輸入：X_train, X_test, y_train, y_test(假設這些資料都已經標準化或編碼完成)
def quick_model_compare(X_train, X_test, y_train, y_test):
    models = {
        "Logistic Regression": LogisticRegression(max_iter=1000, random_state=42),
        "Random Forest": RandomForestClassifier(random_state=42),
        "XGBoost": XGBClassifier(random_state=42, verbosity=0),
        "LightGBM": LGBMClassifier(random_state=42)
        }
    
    results = []
    
    for name, model in models.items():
        print(f"\n{'='*20} {name} {'='*20}")
        model.fit(X_train,y_train)
        y_pred = model.predict(X_test)
        f1 = f1_score(y_test, y_pred, average='macro')
        print(classification_report(y_test, y_pred, digits=3))
        
        results.append({
            'Model':name, 'F1_macro':f1
            })
        
    df_results = pd.DataFrame(results).sort_values(by="F1_macro", ascending=False)
    print("\n=== 各模型 F1 Macro 分數比較 ===")
    print(df_results)
    
    # 找出最佳模型,第一列就是排序後 F1_macro 最大的模型
    best_model_row = df_results.iloc[0] 
    print("最佳模型：", best_model_row['Model'], '| F1_macro:', best_model_row['F1_macro'])
    
    # 輸出最佳模型
    return df_results, best_model_row



# ==================== 模型訓練主函式 ====================
def model_train_evaluate(X_train, X_test, y_train, y_test, le=None):
    """
    使用 GridSearchCV 訓練多個模型並比較效能
    
    注意：y_train 和 y_test 應該已經被 Label Encoding，直接傳進來即可
         此階段只做模型初選，因此不執行CV驗證
    
    Pipeline 流程：
        1. ColumnTransformer（已在外部做好，這裡不再做）
        2. 分類模型（LR, RF, XGBoost, LightGBM）
        3. GridSearchCV：自動搜尋最佳超參數
    
    參數：
        X_train (pd.DataFrame): 訓練特徵（已編碼）
        X_test (pd.DataFrame): 測試特徵（已編碼）
        y_train (np.array): 訓練目標變數（已 Label Encoding）
        y_test (np.array): 測試目標變數（已 Label Encoding）
        le (LabelEncoder): 已經 fit 過的 LabelEncoder 物件
    
    回傳：
        tuple: (結果字典, LabelEncoder 物件)
    """
    print("\n" + "="*60)
    print("開始模型訓練")
    print("="*60)
    
    # ===== 1. 直接使用已編碼的目標變數 =====
    # y_train 和 y_test 已經在 data_preprocessing.py 中被 Label Encoding
    y_train_enc = y_train
    y_test_enc = y_test
    
    class_labels = np.unique(y_train_enc)
    n_classes = len(class_labels)
    
    print(f"\n✓ 使用已編碼的目標變數")
    print(f"  訓練集: {X_train.shape}")
    print(f"  測試集: {X_test.shape}")
    print(f"  類別數: {n_classes}")
    # 輸出類別對應關係，方便觀察
    if le is not None:
        print(f"  類別對應: {dict(zip(le.classes_, class_labels))}")
    
    # ===== 2. 時間序列交叉驗證 =====
    tscv = TimeSeriesSplit(n_splits=CV_SPLITS)
    
    # ===== 3. 計算樣本權重（處理類別不平衡）=====
    #  建立sample_weight(針對整個訓練集):計算訓練的樣本權重(少數類別樣本數少，權重大，反之亦然)
    train_weights = compute_sample_weight(class_weight='balanced', y=y_train_enc)
    
    # ===== 4.模型訓練 =====
    results = {}
    
    #  建立迴圈跑每個模型
    for model_name, config in GRIDSEARCH_PARAMS.items():
        print(f"\n{'='*60}")
        print(f"訓練 {model_name}...")
        print(f"{'='*60}")
        
        #  XGBoost,LightGBM要額外設定
        if model_name =='XGBoost':
            model = XGBClassifier(
                eval_metric='mlogloss',
                objective='multi:softprob',
                num_class=n_classes,
                random_state=RANDOM_STATE
            )
        elif model_name == 'LightGBM':
            model = lgb.LGBMClassifier(
                #class_weight='balanced',  # 因為XGBoost再多分類需要設定train_weight，就統一都先使用這個，之後PSO要在使用class_weight='balanced'再用
                objective='multiclass',
                num_class=n_classes,
                min_child_samples=10,     # 默認20，改小可以分裂更多
                min_split_gain=0.0,       # 默認0，可調整
                max_depth=-1,             # 可以設定一個正整數限制深度
                n_estimators=100,
                learning_rate=0.05,
                random_state=RANDOM_STATE
            )
        else:
            model = config['model']


        #  GridSearchCV
        grid_search = GridSearchCV(
            model,
            param_grid=config["params"],
            cv=tscv,
            scoring='f1_macro',  # 因為y是多元類別，因此分數的評估要使用'macro','weighted'，而且注重在少數類別，所以要使用recall
            verbose=2,
            n_jobs=1
        )
        
        #  GridSearch訓練
        grid_search.fit(X_train, y_train_enc, sample_weight=train_weights)    #先讓所有可以調整權重的模型先使用手動設定的權重(XGBoost/LightGBM)，要把參數傳到 model 這個 step，格式就是 "stepname__paramname" → model__sample_weight
        
        # 取得最佳模型
        best_model = grid_search.best_estimator_
        
        print(f"\n✓ {model_name} 訓練完成")
        print(f"  最佳參數: {grid_search.best_params_}")
        print(f"  CV 最佳分數: {grid_search.best_score_:.4f}")
        
        # ===== 5.測試集評估 =====
        # shape = (n_samples, n_classes)
        y_prob = best_model.predict_proba(X_test)
        y_pred = best_model.predict(X_test)
        y_test_bin = label_binarize(y_test_enc, classes=class_labels)  # 把多類別標籤轉換成二進制（0/1）矩陣

        # 輸出分類報告
        print(f"\n 分類報告：")
        print(f"==== Classification Report for {model_name} ====")
        print(classification_report(y_test_enc, y_pred, target_names=le.classes_))
    
        # 評分指標
        accuracy = accuracy_score(y_test_enc, y_pred)
        f1_macro = f1_score(y_test_enc, y_pred, average='macro')
        
        print(f"  準確度：{accuracy:.4f}")
        print(f"  Macro F1：{f1_macro:.4f}")
        
        #  將結果存放在result中
        results[model_name] = {
            'model': best_model,
            'y_prob': y_prob,
            'y_pred':y_pred,
            'accuracy':accuracy,
            'f1_macro':f1_macro
            # 'optimal_threshold':optimal_threshold_f1
            }
        
    
    print(f"\n{'='*60}")
    print("所有模型訓練完成")
    print(f"{'='*60}")
    
    return results, le
        
    
# ==================== 模型比較 ====================
def compare_models(results):
    """
    比較所有模型的效能
    
    參數：
        results (dict): 模型訓練結果
    """
    print("\n" + "="*60)
    print("模型效能比較")
    print("="*60)   
    
    comparsion_df = pd.DataFrame([
        {
            'Model':name,
            'Accuracy':info['accuracy'],
            'Macro F1':info['f1_macro']
        }
        for name, info in results.items()
    ]).sort_values('Macro F1', ascending=False)
    
    print(comparsion_df)
    print(f"\n最佳模型： {comparsion_df.iloc[0]['Model']}")
    
    return comparsion_df
    

    
    
    

    
    
    
    
    
    
    