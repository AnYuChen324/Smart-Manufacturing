#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Oct 11 16:16:09 2025

@author: anyuchen
"""

"""
02_data_preprocessing.py：資料清理、特徵衍生、編碼
"""

import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, LabelEncoder, OneHotEncoder
from sklearn.compose import ColumnTransformer
from config import NUM_COL, DERIVED_COL, ROLLING_WINDOWS

def generate_derived_features(manufacture_data):
    """
    生成衍生特徵：差分、滾動平均、標準差
    
    用途：
        - 差分(diff)：捕捉瞬間變化，檢測突發異常
        - 滾動平均(ma)：平滑數據，去除雜訊
        - 標準差(std)：捕捉波動幅度
    
    注意：
        - 因為 diff/rolling 是按時間計算，所以一定要先按機器編號分組
        - 如果不分組，會把上一台機器的最後一筆當作下一台的第一筆，造成錯誤
    
    參數：
        manufacture_data (pd.DataFrame): 原始資料
    
    回傳：
        pd.DataFrame: 包含衍生特徵的資料框
    """
    
    df = manufacture_data.copy()
    print("生成衍生特徵...")
    
    
    #  共有50台相同機型的機台，因此要將不同機台分開算，根據不同機台計算衍生特徵
    #  雖然是相同機型，但因為diff/mean/std是根據時間計算，如果不分組，會把上一台機台的最後一筆當作下一台的前一筆，會造成「跳躍」或不合理的差異。
    #  計算速度差及錯誤率差:可以抓到瞬時變化、突發異常
    
    #  -----  1.diff是計算差分：後值-前值(第一筆為Nan)  -----
    df['Speed_diff'] = df.groupby('Machine_ID')['Production_Speed_units_per_hr'].diff()
    df['Error_diff'] = df.groupby('Machine_ID')['Error_Rate_%'].diff()
    #  缺失值填補：向後填補bfill
    df['Speed_diff'] = df['Speed_diff'].bfill()
    df['Error_diff'] = df['Error_diff'].bfill()
    
    print("  ✓ 差分特徵完成")


    #  -----  2.計算滾動平均：速度短期趨勢及錯誤綠短期趨勢  -----
    #  rolling(3)：窗口大小為3，抓每台機台最近三筆資料，缺點是前兩筆會是Nan,但如果是使用XGBoost,LightGBM可以自動處理Nan
    #  reset_index(0, drop=True) → 因為 rolling 會改變索引，需要把結果對齊回原本 dataframe 的 index。
    df['Speed_ma3'] = df.groupby('Machine_ID')['Production_Speed_units_per_hr'].rolling(ROLLING_WINDOWS).mean().reset_index(0, drop=True)
    df['Error_ma3'] = df.groupby('Machine_ID')['Error_Rate_%'].rolling(ROLLING_WINDOWS).mean().reset_index(0, drop=True)
    #  缺失值填補：向後填補bfill
    df['Speed_ma3'] = df['Speed_ma3'].bfill()
    df['Error_ma3'] = df['Error_ma3'].bfill()
    
    print("  ✓ 滾動平均特徵完成")


    #  -----  3.計算滾動標準差：速度標準差及錯誤率標準差：可以抓到波動大小  -----
    #  rolling(3).std()是利用最近三筆計算標準差(該筆與前兩筆)
    df['Speed_std3'] = df.groupby('Machine_ID')['Production_Speed_units_per_hr'].rolling(ROLLING_WINDOWS).std().reset_index(0, drop=True)
    df['Error_std3'] = df.groupby('Machine_ID')['Error_Rate_%'].rolling(ROLLING_WINDOWS).std().reset_index(0, drop=True)
    #  缺失值填補：向後填補bfill
    df['Speed_std3'] = df['Speed_std3'].bfill()
    df['Error_std3'] = df['Error_std3'].bfill()
    
    print("  ✓ 標準差特徵完成")
    
    return df



def prepare_model_data(df_with_features):
    """
    準備模型訓練資料：分離特徵與目標變數
    （尚未做 encode，等切分後再做）
    
    參數：
        df_with_features (pd.DataFrame): 包含衍生特徵的資料框
    
    回傳：
        tuple: (X, y)
    """
    print("\n準備模型資料...")
    
    # 移除不需要的欄位，將資料照時間排序再切分
    x = df_with_features.drop(columns=['Machine_ID','Production_Speed_units_per_hr','Error_Rate_%', 'Timestamp'], axis=1, errors='ignore')
    y = df_with_features['Efficiency_Status']
    
    print(f"  ✓ 特徵數量: {x.shape[1]}")
    print(f"  ✓ 樣本數量: {x.shape[0]}")
    print(f"  ✓ 目標變數分布:\n{y.value_counts()}")
    
    return x, y
    
    
def train_test_split_timeseries(x, y, train_ratio=0.8):
    """
    時間序列切分：按時間順序分割訓練測試集
    
    為什麼不用 sklearn 的 train_test_split？
        - sklearn 的 split 是隨機的，會破壞時間序列結構
        - 時間序列必須先用歷史資料訓練，再用未來資料測試
        - 否則會造成資料洩漏（data leakage）
    
    參數：
        X (pd.DataFrame): 特徵資料
        y (pd.Series): 目標變數
        train_ratio (float): 訓練集比例
    
    回傳：
        tuple: (X_train, X_test, y_train, y_test)
    """
    
    split_idx = int(len(x) * 0.8)
    X_train, X_test = x.iloc[:split_idx], x.iloc[split_idx:]
    y_train, y_test = y.iloc[:split_idx], y.iloc[split_idx:]
    
    print(f"\n時間序列切分 ({train_ratio*100:.0f}% train, {(1-train_ratio)*100:.0f}% test):")
    print(f"  訓練集: {X_train.shape[0]} 筆")
    print(f"  測試集: {X_test.shape[0]} 筆")
     
    return X_train, X_test, y_train, y_test
    


def encode_features(X_train, X_test, y_train, y_test):
    """
    編碼特徵：One Hot Encoding（類別）+ StandardScaler（數值）
    
    重要概念（避免資料洩漏）：
        - 訓練集：fit_transform（先學習統計特性，再轉換）
        - 測試集：transform（只用訓練集學到的統計特性轉換，不重新學習）
    
    參數：
        X_train, X_test (pd.DataFrame): 訓練/測試特徵
        y_train, y_test (pd.Series): 訓練/測試目標變數
    
    回傳：
        tuple: (X_train_encoded, X_test_encoded, y_train_enc, y_test_enc, scaler, le, preprocessor)
    """
    
    print("\n編碼特徵（避免資料洩漏）...")
    
    # ===== 1.自動篩選存在的欄位 =====
    from config import NUM_COL_ALL, CATEGORY_COL_ALL
    
    num_col = [c for c in NUM_COL_ALL if c in X_train.columns]
    category_col = [c for c in CATEGORY_COL_ALL if c in X_train.columns]
    
    print(f"  數值特徵: {num_col}")
    print(f"  類別特徵: {category_col}")
    
    # ===== 2. 建立 ColumnTransformer（特徵預處理）=====
    # 這個物件會在訓練集上 fit，然後用於轉換測試集
    preprocessor = ColumnTransformer(
        transformers=[
            ('num', StandardScaler(), num_col),
            ('category', OneHotEncoder(drop='first', sparse_output=False, dtype=int), category_col)
        ],
        remainder='drop'     # 不保留其他欄位，避免字串混進來
    )
    
    # ===== 3. 在訓練集上 fit（學習統計特性）=====
    # 重要：只在訓練集上 fit！
    X_train_encoded = preprocessor.fit_transform(X_train)
    print("  ✓ Preprocessor 在訓練集上 fit 完成")
    
    # ===== 4. 用同樣的 preprocessor 轉換測試集 =====
    # 重要：只用 transform，不重新 fit！
    X_test_encoded = preprocessor.transform(X_test)
    print("  ✓ 測試集用訓練集的統計特性轉換完成")
    
    # ===== 5. Label Encoding（目標變數）=====
    le = LabelEncoder()
    y_train_encoded = le.fit_transform(y_train)
    y_test_encoded = le.transform(y_test)
    print("  ✓ Label Encoding 完成")
    print(f"    類別對應: {dict(zip(le.classes_, le.transform(le.classes_)))}")
    
    # ===== 6. 轉換回 DataFrame（為了 SHAP 分析）=====
    feature_names = preprocessor.get_feature_names_out()
    X_train_encoded = pd.DataFrame(X_train_encoded, columns=feature_names)
    X_test_encoded = pd.DataFrame(X_test_encoded, columns=feature_names)
    
    return X_train_encoded, X_test_encoded, y_train_encoded, y_test_encoded, le, preprocessor



