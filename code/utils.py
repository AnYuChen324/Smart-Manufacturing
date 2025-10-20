#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Oct 11 15:57:38 2025

@author: anyuchen
"""
"""
通用工具函式：VIF、Cramer's V、SQL 查詢等
"""

import numpy as np
import pandas as pd
from scipy.stats import chi2_contingency
from statsmodels.stats.outliers_influence import variance_inflation_factor
from pandasql import sqldf

# ==================== SQL 查詢工具 ====================
pysqldf = lambda q: sqldf(q, globals())


# ==================== VIF 計算 ====================
def calculate_vif(x):
    """
    計算 Variance Inflation Factor（方差膨脹因子）
    用途：檢測特徵之間的多重共線性
    
    參數：
        x (pd.DataFrame): 數值特徵資料(標準化後的數值變數,one hot encoding後的類別變數)
    
    回傳：
        pd.DataFrame: 包含特徵名稱與 VIF 值的資料框
    
    VIF 解釋：
        VIF = 1: 沒有共線性
        VIF < 5: 低共線性（一般可接受）
        VIF > 10: 高共線性（需要考慮移除特徵）
    """
    
    vif_data = pd.DataFrame()
    vif_data["Feature"] = x.columns  # 變數名稱
    vif_data['VIF'] = [variance_inflation_factor(
        x.values, i)for i in range(x.shape[1])]
    print(f"=== Variance Inflation Factor ===")
    print(vif_data)
    
    return vif_data

    
# ==================== Cramer's V 計算 ====================
def cramers_v(confusion_matrix):
    """
    計算 Cramer's V 係數（卡方檢定的輔助函式）:兩個類別變數之間的相關性強度
    值域：0 到 1，越接近 1 代表相關性越強
    
    參數：
        confusion_matrix (pd.DataFrame): 列聯表
    
    回傳：
        float: Cramer's V 係數
        
    用途：
        - 衡量類別變數與類別變數之間的相關性
        - 比卡方檢定更不易被樣本大小影響
    """
    from scipy.stats import chi2_contingency
    
    chi2 = chi2_contingency(confusion_matrix)[0]
    n = confusion_matrix.sum().sum()
    min_dim = min(confusion_matrix.shape)-1
    
    return np.sqrt(chi2/(n*min_dim))


# ==================== 卡方檢定 ====================  
def chi2_test(target, features):
    """
    執行卡方檢定：檢測類別變數之間是否有顯著關係
    
    參數：
        target (pd.Series): 目標變數（類別型）-'Efficiency_Status'需為一維資料
        features (pd.DataFrame): 特徵資料（類別型）-所有考量的特徵資料，包含多個特徵
    原理：
        - 卡方統計量越大，p值越小 => 兩變數相關性越強
        - p值 <= 0.05：兩變數有顯著關係
        - p值 > 0.05：兩變數無顯著關係
    回傳：
        pd.DataFrame: 包含檢定結果的資料框
    """
    from scipy.stats import chi2_contingency
    
    chi2_results = []
    
    for feature in features.columns:
        # 建立列連表,需確認輸入數據為一維
        contingency_table = pd.crosstab(target, features[feature])
        
        # 執行卡方檢定
        chi2, p, dof, expected = chi2_contingency(contingency_table)
        
        # 計算Cramer's V
        cramers_v_value = cramers_v(contingency_table)
        
        # 輸出結果為
        result = {
            'Feature': feature,
            'Chi2_stat': chi2,
            'P-value': p,
            'Cramers_V': cramers_v_value,
            'Significance': '顯著性關係' if p <= 0.05 else '不顯著關係'
        }
        chi2_results.append(result)
        
        # 顯示結果
        print(
            f"Feature:{feature} 與 目標變數間的關係為{'顯著性關係' if p <= 0.05 else '不顯著關係'} "
            f"(P-value:{p:.4f}) | Cramer's V:{cramers_v_value:.4f} | {result['Significance']}"
        )
        print(contingency_table)
        print()
        
    return pd.DataFrame(chi2_results)


# ==================== 相關性分析 ====================
def get_correlation_analysis(corr_matrix, feature_name):
    """
    從相關性矩陣中提取特定特徵與其他特徵的相關性
    
    參數：
        corr_matrix (pd.DataFrame): 相關性矩陣
        feature_name (str): 特徵名稱
    
    回傳：
        pd.Series: 該特徵與其他特徵的相關性（按大小排序）
    """
    return corr_matrix[feature_name].sort_values(ascending=False)


# ==================== 樣本權重計算 ====================
def get_sample_weights(y):
    """
    計算樣本權重：用於不平衡類別的訓練
    少數類別的權重會較大，多數類別的權重會較小
    
    參數：
        y (pd.Series): 類別標籤
    
    回傳：
        np.array: 每個樣本的權重
    """
    from sklearn.utils.class_weight import compute_sample_weight
    return compute_sample_weight(class_weight='balanced', y=y)


# ==================== 數據統計摘要 ====================
def summarize_data(df, category_col=None):
    """
    生成資料的統計摘要
    
    參數：
        df (pd.DataFrame): 資料框
        category_col (list): 類別欄位列表
    
    回傳：
        dict: 包含各類別的樣本計數
    """
    summary = {}
    
    if category_col:
        for col in category_col:
            if col in df.columns:
                count = df[col].value_counts()
                summary[col] = count.to_dict()
                print(f"\n=== {col} 分布 ===")
                print(count)
    
    return summary





