#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Oct 11 12:57:24 2025

@author: anyuchen
"""

"""
03_eda.py：探索性資料分析 (EDA)
包含：分布圖、統計檢定、相關性分析、PCA/TSNE 可視化
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import shapiro, kruskal
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.preprocessing import label_binarize
import itertools
from config import GRAPH_PATH, EARTH_COLORS, NUM_COL, CATEGORY_COL, TSNE_SAMPLE_SIZE
from utils import calculate_vif, chi2_test, get_correlation_analysis


# ==================== 繪圖設定 ====================
def setup_plot_style():
    """
    設定全域繪圖風格
    """
    #確保中英文字能正確顯示，font.sans-serif：指定無襯線字體（Sans-serif）系列要用哪個字，'DejaVu Sans' 是 Matplotlib 的預設西文字體之一。
    plt.rcParams['font.sans-serif'] = ['DejaVu Sans']
    #設定 Seaborn 圖表的背景風格，"whitegrid"：表示背景是白底 + 灰色網格線。
    sns.set_style("whitegrid")
    
    
# ==================== 長條圖 ====================
def draw_bar_plot(data, x, y, title='', xlabel='', ylabel='', bar_width=0.8):
    """
    繪製長條圖：查看類別特徵的分佈
    
    參數：
        data(pd.DataFrame):資料框
        x(str): x 軸欄位
        y(str): y 軸欄位
        title(str): 圖表標題
        xlabel(str): x 軸標籤
        ylabel(str): y 軸標籤
        bar_width(float): 長條寬度
    """

    palette = list(itertools.islice(itertools.cycle(EARTH_COLORS), len(data)))

    plt.figure(figsize=(14, 6))
    ax = sns.barplot(data=data, x=x, y=y, palette=palette,
                     legend=False, hue=x, width=bar_width)

    plt.xticks(rotation=0, ha='right')
    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)

    # 加上數值標籤
    for i, bar in enumerate(ax.patches):
        height = bar.get_height()
        ax.text(
            bar.get_x() + bar.get_width()/2,
            height,
            f"{int(height)}",
            ha='center',
            fontsize=8)

    plt.tight_layout()
    plt.savefig(f"{GRAPH_PATH}/{title} Feature Distribution.png", dpi=300, bbox_inches='tight')
    plt.show()

    
    
# ==================== 常態分佈檢定 ====================    
def test_normality(df, features):
    """
    使用 Shapiro-Wilk 檢定檢驗資料是否符合正常分布
    
    原理：
        - 原假設：資料符合正常分布
        - p值 >= 0.05：無法拒絕原假設（可能符合正常分布）
        - p值 < 0.05：拒絕原假設（不符合正常分布）
    
    參數：
        df (pd.DataFrame): 資料框
        features (list): 特徵列表
    """
    #  1.Shapiro-Wilk 檢定常態分佈：先觀察所有特徵的資料分佈,原先用kstest檢定(假設輸入是連續變數,故效果不佳),改用Shapiro-Wilk 檢定
    #   會得到所有數值特徵都不符合常態分佈
    for feature in features:
        # dropna()是為了確保沒有nan值
        data = df[feature].dropna()

        if len(data) >= 3:  # Shapiro-Wilk 最少需要3筆資料
            stat, p = shapiro(data)
            # print(f"Feature:{feature} stat:{stat:.4f} p:{p:.4f}")
            if p >= 0.05:
                print(f"{feature}符合常態分佈 (p={p:.4f})")
            else:
                print(f"{feature}不符合常態分佈 (p={p:.4f})")
        else:
            print(f"{feature} 資料不足，無法進行Shapiro-Wilk 檢定")


# ==================== 直方圖（分布可視化）====================
def plot_distributions(df, features, bar_width=0.8):
    """
    繪製特徵的分布直方圖
    
    參數：
        df (pd.DataFrame): 資料框
        features (list): 特徵列表
    """
    print("\n 繪製分佈直方圖...")
    
    #  可視化數值特徵的常態分佈:長條圖
    for feature in features:
        plt.figure(figsize=(10,5))
        # stat = 'density'將高度標準化為密度，便於與 KDE 曲線比較
        sns.histplot(df[feature], kde=True,
                     stat='density', bins='auto', color="#D3D3D3")
        plt.title(f"Distribution of {feature}")
        plt.xlabel(f"Value of {feature}")
        plt.ylabel('Density')
        # 並將所有圖形存下來
        plt.savefig(f"{GRAPH_PATH}/{feature}_distribution.png", dpi=300)
        plt.show()
        plt.close()  # 加 plt.close() 釋放記憶體，避免多圖時堆積


 
# ==================== Kruskal-Wallis 檢定 ====================   
def kruskal_test(df, target_col, features):
    """
    Kruskal-Wallis H 檢定：比較三組或以上的獨立樣本
    （非參數檢定，不假設資料正常分布）
    
    用途：
        - 多組獨立樣本比較（對應 ANOVA）
        - 當資料不符合正常分布時使用
    
    原理：
        - 原假設：所有組別的中位數相同
        - p值 <= 0.05：拒絕原假設（至少有一組不同）
        - p值 > 0.05：無法拒絕原假設（所有組別相同）
    
    參數：
        df (pd.DataFrame): 資料框
        target_col (str): 目標變數欄位
        features (list): 特徵列表
    
    回傳：
        pd.DataFrame: 包含檢定結果的資料框
    """
    print(f"\n=== Kruskal-Wallis H 檢定 ===")
    
    kruskal_results = []
    
    for feature in features:
        #  依目標變數的每個類別拆分數值特徵
        groups = [df[df[target_col] == cls][feature]
            for cls in df[target_col].unique()]

        # Kruskal-Wallis H Test
        stat, p_value = kruskal(*groups)
        significance = '顯著性差異' if p_value < 0.05 else '無顯著性差異'
        print(f"{feature} : Kruskal-Wallis H statistic = {stat:.4f} , p = {p_value:.4f} ,{significance}")

        kruskal_results.append({
            'Feature': feature,
            'Kruskal_stat': stat,
            'P-value': p_value,
            'Significance':significance
        })
        
    #  將資料轉成表格，視覺上比較好看
    return pd.DataFrame(kruskal_results)

   
# ==================== 相關性分析 ====================    
def analyze_correlations(df, features):
    """
    計算並繪製相關性矩陣（Pearson 相關係數）
    
    用途：
        - 檢測特徵之間的線性相關性
        - 識別可能存在的多重共線性問題
    
    參數：
        df (pd.DataFrame): 資料框
        features (list): 特徵列表
    
    回傳：
        pd.DataFrame: 相關性矩陣
    """
    print("\n分析相關性...")
    
    corr_matrix = df[features].corr()
    
    #繪製 Heatmap 
    plt.figure(figsize=(12, 10))
    sns.heatmap(corr_matrix, annot=True, cmap='YlGnBu', fmt='.2f', linewidth=0.5)
    plt.title('Correlation Heatmap of Numeric Features', fontsize=16)
    plt.savefig(f"{GRAPH_PATH}/correlation_heatmap.jpg", dpi=500)
    plt.show()
    
    return corr_matrix
        
   
# ==================== VIF 多重共線性檢定 ====================
def check_multicollinearity_vif(df, features):
    """
    從units.py中使用calculate_vif
    使用 VIF（方差膨脹因子）檢測多重共線性
    
    VIF 值解釋：
        - VIF = 1：完全沒有共線性
        - 1 < VIF < 5：低到中度共線性（通常可接受）
        - VIF >= 5：高共線性（建議移除或合併特徵）
        - VIF >= 10：嚴重共線性（必須處理）
    
    參數：
        df (pd.DataFrame): 原始資料框
        features (list): 特徵列表
    
    回傳：
        pd.DataFrame: VIF 結果
    """
    
    print("\n 檢查多重共線性(VIF)...")
    vif_df = df[features].copy()
    
    return calculate_vif(vif_df)


# ==================== PCA 可視化 ====================
def draw_pca(x, y, title='PCA Visualization'):
    """
    使用 PCA 降維到 2D 並繪製散點圖
    
    用途：
        - 觀察資料在低維空間中的分布
        - 檢驗是否存在自然的群集（cluster）
    
    參數：
        X (pd.DataFrame): 特徵資料（應為標準化後）
        y (pd.Series): 目標變數
        title (str): 圖表標題
    """
    print("\n執行 PCA...")
    
    reducer = PCA(n_components=2)
    x_reduced = reducer.fit_transform(x)
    
    print(f" 解釋變異數比：{reducer.explained_variance_ratio_}")
    print(f" 累積解釋變異數：{np.sum(reducer.explained_variance_ratio_)}")
    
    # 繪圖資料
    df_plot = pd.DataFrame(data=x_reduced, columns=['PCA1', 'PCA2'])
    df_plot['label'] = y # 在降維資料中創建新欄位label,並填入原本的y

    # 繪圖
    plt.figure(figsize=(10, 6))
    sns.scatterplot(data=df_plot, x='PCA1', y='PCA2', hue='label', palette={
                    0: 'brown', 1: 'blue', 2: 'green'}, alpha=0.6)
    plt.title(f'{title} (PCA)')
    plt.legend(title='Label')
    # 並將所有圖形存下來
    plt.savefig(
        f"{GRAPH_PATH}/{title}_PCA.png", dpi=300)
    plt.grid(True)
    plt.show()
    


#                                                        =====     PCA/TSNE觀察三個類別的ｙ分佈     =====
#  利用PCA/TSNE降成2維，觀察所有資料分布，x需要是標準化後的特徵

#  畫出資料分佈圖
def draw_pca_tsne(x, y, method='PCA', title='Feature Space'):
    # 執行降維
    if method.upper() == 'PCA':
        reducer = PCA(n_components=2)
        x_reduced = reducer.fit_transform(x)
        comp_names = ['PCA1', 'PCA2']
        y_plot = y  # 全部y
    elif method.upper() == 'TSNE':
        # 為了降低計算成本，隨機取50000個樣本
        x_tsne = x.sample(n=50000, random_state=42)
        y_tsne = y.loc[x_tsne.index]  # 抽樣相同的 index
        reducer = TSNE(n_components=2, perplexity=30,
                       random_state=42, max_iter=250)
        x_reduced = reducer.fit_transform(x_tsne)
        comp_names = ['TSNE1', 'TSNE2']
        y_plot = y_tsne
    else:
        raise ValueError("method 必須是 'PCA' 或 'TSNE'")

    # 繪圖資料
    df = pd.DataFrame(data=x_reduced, columns=comp_names)
    df['label'] = y_plot.reset_index(drop=True)  # 在降維資料中創建新欄位label,並填入原本的y

    # 繪圖
    plt.figure(figsize=(10, 6))
    sns.scatterplot(data=df, x=comp_names[0], y=comp_names[1], hue='label', palette={
                    0: 'brown', 1: 'blue', 2: 'green'}, alpha=0.6)
    plt.title(f'{title} ({method.upper()})')
    plt.xlabel(comp_names[0])
    plt.ylabel(comp_names[1])
    plt.legend(title='Label')
    # 並將所有圖形存下來
    plt.savefig(
        f"/Users/anyuchen/Desktop/程式語言/智慧製造/graph/{title} ({method.upper()}).png", dpi=300)
    plt.grid(True)
    plt.show()



# ==================== TSNE 可視化 ====================
def draw_tsne(x, y, sample_size=TSNE_SAMPLE_SIZE, title='TSNE Visualization'):
    """
    使用 TSNE 降維到 2D 並繪製散點圖
    
    用途：
        - 更好地呈現非線性結構
        - 比 PCA 更適合視覺化複雜分布
    
    注意：
        - TSNE 計算成本較高，通常先採樣
        - 結果具有隨機性，每次運行可能不同
    
    參數：
        X (pd.DataFrame): 特徵資料
        y (pd.Series): 目標變數
        sample_size (int): 採樣大小（降低計算成本）
        title (str): 圖表標題
    """
    print(f"\n執行 TSNE（採樣 {sample_size} 筆資料）...")
    
    # 採樣
    if len(x) > sample_size:
        x_tsne = x.sample(n=sample_size, random_state=42)
        y_tsne = y[x_tsne.index]  # 抽樣相同的 index
    else:
        x_tsne = x
        y_tsne = y
    
    # 執行tsne
    reducer = TSNE(n_components=2, perplexity=30, random_state=42, max_iter=250)
    x_reduced = reducer.fit_transform(x_tsne)
    
    # 繪圖資料
    df_plot = pd.DataFrame(data=x_reduced, columns=['TSNE1','TSNE2'])
    df_plot['label'] = y_tsne  # 在降維資料中創建新欄位label,並填入原本的y

    # 繪圖
    plt.figure(figsize=(10, 6))
    sns.scatterplot(data=df_plot, x='TSNE1', y='TSNE2', hue='label', 
                    palette={0: 'brown', 1: 'blue', 2: 'green'}, alpha=0.6)
    plt.title(f'{title} (TSNE)')
    plt.legend(title='Label')
    # 並將所有圖形存下來
    plt.savefig(f"{GRAPH_PATH}/{title}_TSNE.png", dpi=300)
    plt.grid(True)
    plt.show()

 
    
# ==================== 執行完整 EDA ====================
def run_eda(manufacture_data, X, y):
    """
    執行完整的 EDA 流程
    
    參數：
        manufacture_data (pd.DataFrame): 原始資料
        X (pd.DataFrame): 特徵資料
        y (pd.Series): 目標變數
    """
    print("\n" + "="*50)
    print("開始執行 EDA")
    print("="*50)
    
    # 1. 類別特徵分布
    print("\n### 類別特徵分布 ###")
    machine_count = manufacture_data.groupby('Machine_ID').size().reset_index(name='count')
    draw_bar_plot(machine_count, x='Machine_ID', y='count',
                  title='Machine Count', xlabel='machine_id', ylabel='count')
    
    operation_mode_count = manufacture_data.groupby(
        'Operation_Mode')['Machine_ID'].count().reset_index(name='count')
    draw_bar_plot(operation_mode_count, x='Operation_Mode', y='count',
                  title='Operation Mode Count', xlabel='operation_mode', ylabel='count')
    
    efficiency_status_count = manufacture_data.groupby('Efficiency_Status')[
        'Machine_ID'].count().reset_index(name='count')
    draw_bar_plot(efficiency_status_count, x='Efficiency_Status', y='count',
                  title='Efficiency Status Count', xlabel='efficiency_status', ylabel='count')
    
    # 2. 正常分布檢定
    test_normality(manufacture_data, NUM_COL)
    
    # 3. 分布直方圖
    plot_distributions(manufacture_data, NUM_COL)
    
    # 4. Kruskal-Wallis 檢定
    print("\n### Kruskal-Wallis 檢定 ###")
    kruskal_results = kruskal_test(manufacture_data, 'Efficiency_Status', NUM_COL)
    print(kruskal_results)
    
    # 5. 卡方檢定（類別變數：檢測類別變數之間是否有顯著關係）
    # chi2_test(target, features) 來自utils
    print("\n### 卡方檢定 ###")
    chi2_data = manufacture_data[['Operation_Mode', 'Machine_ID']]
    chi2_results = chi2_test(manufacture_data['Efficiency_Status'], chi2_data)
    print(chi2_results)
    
    # 6. 相關性分析
    print("\n### 相關性分析 ###")
    # 需要使用未標準化的數值特徵進行相關性分析
    num_features_original = [col for col in NUM_COL if col in manufacture_data.columns]
    corr_matrix = analyze_correlations(manufacture_data, num_features_original)
    
    # 7. VIF 檢定
    print("\n### VIF 多重共線性檢定 ###")
    check_multicollinearity_vif(manufacture_data, num_features_original)
    
    # 8. PCA 可視化
    print("\n### PCA 可視化 ###")
    draw_pca(X, y, title='Feature Space - PCA')
    
    # 9. TSNE 可視化
    print("\n### TSNE 可視化 ###")
    draw_tsne(X, y, title='Feature Space - TSNE')
    
    print("\n" + "="*50)
    print("EDA 完成")
    print("="*50)

















































   
   
    
   
    
   
    
   
    
   
    
   
    
   
    
   
    
   
    
   
    
   
    
    
    
    
    
    