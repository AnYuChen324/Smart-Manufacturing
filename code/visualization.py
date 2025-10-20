#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Oct 13 00:10:34 2025

@author: anyuchen

08_visualization.py：模型評估視覺化（ROC、PRC、SHAP、特徵重要性）
"""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import shap
from itertools import cycle
from sklearn.metrics import (
    roc_curve, roc_auc_score,
    precision_recall_curve, average_precision_score,
    classification_report, confusion_matrix
)
from sklearn.preprocessing import label_binarize

from config import GRAPH_PATH

# ==================== ROC 曲線 ====================
def plot_multiple_roc(results, y_test, title_suffix=""):
    """
    繪製多分類 ROC 曲線
    
    用途：
        - 評估模型在不同閾值下的分類效能
        - ROC 曲線越接近左上角，模型越好
        - AUC（曲線下面積）：越接近 1 越好
    
    參數：
        results (dict): 模型結果字典
        y_test (np.array): 測試目標變數（編碼後）
        title_suffix (str): 圖表標題後綴
    """

    # 使用matplotlib的色彩循環
    colors = cycle(["blue", "green", 'red', 'purple', 'orange', 'brown'])
    plt.figure(figsize=(10, 8))
    # 設定線條
    line_styles = ['-','--',':','-.']

    # one hot encode for ylabel
    n_classes = np.unique(y_test)
    y_test_bin = label_binarize(y_test, classes=n_classes)

    for (model_name, info), color in zip(results.items(), colors):
        if 'y_prob' not in info:
            print(f"跳過 {model_name} 因為沒有 y_prob ,要先執行模型預測")
            continue

        y_prob = info['y_prob']   # shape: (n_samples, n_classes)

        # 計算各類別micro_average ROC
        for i, cls in enumerate((n_classes)):
            fpr, tpr, _ = roc_curve(y_test_bin[:, i], y_prob[:, i])
            roc_auc = roc_auc_score(y_test_bin[:, i], y_prob[:, i])
            plt.plot(fpr, tpr, color=color,
                     linestyle=line_styles[i%len(line_styles)],
                     label=f"{model_name}' (AUC = {roc_auc:.3f})")

    # 整體 macro / weighted AUC
    try:
        roc_auc_macro = roc_auc_score(y_test_bin, y_prob, average='macro')
        # 整體平均不會畫出線，只會在legend顯示Macro AUC
        plt.plot([], [], ' ', label=f"{model_name} - Macro AUC = {roc_auc_macro:.3f}")
    except:
        pass

    # 添加基線(可有可無)
    plt.plot([0, 1], [0, 1], color='gray', linestyle='--', label="Random Base Line")
    plt.title("Multiclass ROC Curve(Micro-average) - {title_suffix}", fontsize=16)
    plt.xlabel('False Positive Rate(FPR)', fontsize=14)
    plt.ylabel('True Positive Rate(TPR)', fontsize=14)
    plt.legend(loc='lower right', fontsize=10)
    plt.grid(True)
    plt.savefig(f"{GRAPH_PATH}/{title_suffix} ROC curve.jpg", dpi=500)
    plt.tight_layout()
    plt.show()
    
    
# ==================== PRC 曲線 ====================
def plot_prc_curve(results, y_test, title_suffix=""):
    """
    繪製 Precision-Recall 曲線
    
    用途：
        - 特別適合用於不平衡資料集
        - 比 ROC 更能反映少數類別的效能
        - 曲線越靠近右上角越好
    
    參數：
        results (dict): 模型結果字典
        y_test (np.array): 測試目標變數（編碼後）
        title_suffix (str): 圖表標題後綴
    
    plot_prc_curve 是用來繪製 Precision-Recall Curve（PRC），並在每個類別上標出 F1 score 最大值對應的最佳決策閥值。
    適用情境：
    多類別分類問題
    想要觀察不同模型、不同類別的 Precision-Recall 表現
    想用 PRC 找出最佳決策閥值，而不是單純以預設 0.5
    """
    
    # *****  繪製 Precision-Recall 曲線並標出 F1 最大值對應的閥值  *****

    #  畫出precision-recall curv，找出在precision跟recall的平衡點
    plt.figure(figsize=(10, 8))
    colors = cycle(["blue", "green", 'red', 'purple', 'orange', 'brown'])
    # 設定線條
    line_styles = ['-','--',':','-.']

    # one hot encode for ylabel
    n_classes = np.unique(y_test)
    y_test_bin = label_binarize(y_test, classes=n_classes)
    
    # 建立迴圈繪製每個模型 & 類別
    for (model_name, info), color in zip(results.items(), colors):
        if 'y_prob' not in info:
            print(f"跳過 {model_name} 沒有 y_prob,要先執行模型預測")
            continue

        y_prob = info['y_prob']  #results是字典，每個模型都有y_prob(預測機率)
        #optimal_threshold = info['optimal_threshold']

        # 計算每個類別的Precision-Recall曲線
        for i, cls in enumerate(n_classes):
            precision, recall, thresholds = precision_recall_curve(
                y_test_bin[:, i], y_prob[:, i])
            
            # 計算F1 score 最大值及對應的閾值
            f1_scores = 2 * (precision * recall) / (precision + recall + 1e-10)
            best_idx = np.argmax(f1_scores)   #找出最大F1對應的index
            best_threshold = thresholds[best_idx-1] if best_idx > 0 else 0.5  # thresholds長度比precision少1
            best_f1 = f1_scores[best_idx]
            
            # 畫出PRC與最佳點
            pr_auc = average_precision_score(y_test_bin[:, i], y_prob[:, i])   #算出PR AUC線下面積,越接近1代表該類別的precision & recall整理表現愈好
            plt.plot(recall, precision, color=color, linestyle=line_styles[i%len(line_styles)],
                     label=f"{model_name} - Class {cls} PR AUC={pr_auc:.3f})")
            
            # 標示出最大F1的點(最佳閥值)
            plt.scatter(recall[best_idx], precision[best_idx], color=color, marker='*',
                        s=100, label=f"{model_name} Class {cls} Best F1={best_f1:.3f} (thr={best_threshold:.2f})")

        #  整體Macro PR AUC
        pr_auc_macro = average_precision_score(y_test_bin, y_prob, average='macro')
        # 整體平均不會畫出線，只會在legend顯示Macro AUC
        plt.plot([], [], ' ', label=f"{model_name} - Macro PR AUC = {pr_auc_macro:.3f}")

    plt.xlabel('Recall', fontsize=14)
    plt.ylabel('Precision', fontsize=14)
    plt.title(f'Precision - Recall  Curve with Best F1 : {title_suffix}', fontsize=16)
    plt.legend(loc="lower left", fontsize='x-small')
    plt.grid(True)
    plt.savefig(f"{GRAPH_PATH}/{title_suffix} Precision & Recall vs Threshold curve.jpg", dpi=500)
    plt.tight_layout()
    plt.show()

    
    
# ==================== 特徵重要性 ====================
def plot_feature_importance(model, feature_names, title_suffix="", top_n=20):
    """
    繪製特徵重要性柱狀圖
    
    用途：
        - 識別對模型預測最有影響力的特徵
        - 幫助理解模型決策過程
    
    參數：
        model: LightGBM 或 XGBoost 模型
        feature_names (list): 特徵名稱
        title_suffix (str): 圖表標題後綴
        top_n (int): 顯示前 N 個特徵
    """

    # 計算特徵重要性
    importance = model.feature_importances_
    feature_importance_df = pd.DataFrame({
        'feature': feature_names,
        'importance': importance,
    }).sort_values(by='importance', ascending=False)    
    
    # 繪圖
    plt.figure(figsize=(10,8))
    ax = sns.barplot(
             data = feature_importance_df.head(top_n),
             x='importance', y='feature'
         )

    plt.title(f"Top {top_n} Feature Importance - {title_suffix}")
    plt.xlabel('Importance')
    
    # 在柱狀圖上顯示數值
    for container in ax.containers:
        ax.bar_label(container, fmt="%d", label_type='edge', padding=3)
    plt.tight_layout()
    plt.savefig(f"{GRAPH_PATH}/{title_suffix}_feature_importance.png", dpi=300)
    plt.show()
    
    return feature_importance_df
    

# ==================== SHAP 分析 ====================    
def plot_shap_analysis(model, X_train, feature_names, title_suffix=""):
    """
    SHAP (SHapley Additive exPlanations) 分析
    SHAP分析與Beeswarm繪圖，支援多分類LightGBM

    results: dict，存放模型物件與y_prob
    model_name: str，對應results裡的key
    X: pd.DataFrame，模型輸入特徵
    title_suffix: str，圖檔名稱後綴

    LightGBM SHAP 的行為
    二分類 / 單分類
    shap_values → 直接是 2D ndarray，shape = (n_samples, n_features)
    可以直接使用，不用做額外處理

    多分類（>2 類）
    shap_values → 通常是 3D ndarray，shape = (n_classes, n_samples, n_features)
    每個類別都有一組 SHAP 值
    如果你想看整體影響 → 對 n_classes 軸取平均 → (n_samples, n_features)
    如果你想分析單一類別 → 取出對應類別的 slice → (n_samples, n_features)
    
    用途：
        - 解釋每個特徵對預測的貢獻
        - Beeswarm 圖：每個點代表一個樣本
        - 紅色點：特徵值高
        - 藍色點：特徵值低
    
    參數：
        model: LightGBM 模型
        X_train (pd.DataFrame): 訓練特徵資料
        feature_names (list): 特徵名稱
        title_suffix (str): 圖表標題後綴
    """
    
    print("\n計算 SHAP 值...")
    
    #  創建SHAP解釋器，對於隨機森林可以使用TreeExplainer
    explainer = shap.TreeExplainer(model)

    #  取得SHAP值
    shap_values = explainer.shap_values(X_train)
    
    #  多分類處理
    shap_values_array = np.array(shap_values)  # 先把list轉乘ndarray
    
    # 確認轉完ndarray以後是不是(n_samples,n_features,n_classes)
    print("Original shap_values_array.shape:", shap_values_array.shape)
    #  先檢查是否為多類別分類，多分類LightGBM有時回傳會直接是3維ndarray(n_samples,n_features,n_classes)
    # 如果類別為3類(3維陣列n_samples,n_features,n_classes，需要取平均或單獨分析
    if shap_values_array.ndim == 3:
    # if isinstance(shap_values, list):  判斷shap回傳的物件是不是list資料型態
        n_classes = len(shap_values)
        print(f"Defected multi-class SHAP with {n_classes} classes")

        # 取平均(針對類別欄位) → 就強制轉成2D array(因為要先看整體有沒有資料洩漏問題，所以先觀察平均)
        shap_values_used = shap_values_array.mean(axis=-1)    # shape = (n_samples, n_features)
        # 可以改選單一類別，可以單獨看各類別，那每一個類別都是2D
        # shap_values_used = shap_values[1]

    else:
        #  二分類或單一類別，LightGBM 直接回傳 ndarray，直接用即可
        shap_values_used = shap_values_array

    #  顯示shape值
    print(f"shap_values:{shap_values_used}")
    # 檢查SHAP值的形狀
    print(f"shap_values shape:{shap_values_used.shape}")
    # 因為是目標變數是二分類問題，shap_values 是一個包含兩個元素的列表：其中shap_values[0]是負類（類別0）的 SHAP 值，shap_values[1] 是正類（類別1）的
    # 但因為這次使用的是LightGBM他只會為傳ndarray:shape = (n_samples,n_features)，所以不需要特別抓正類1
    # 如果是Random Forest/CatBoost就會是回傳list,需要特別抓正類1
    # shap_values_class_1=shap_values[1]  #只關心正類
    
    #  2.繪製Beeswarmplot
    print("繪製 SHAP Beeswarm 圖...")
    shap.summary_plot(shap_values_used, X_train, show=False)
    plt.savefig(f"{GRAPH_PATH}/{title_suffix}_SHAP_Beeswarm plot.png", dpi=500)
    plt.show()
    
     #  3.印出特徵重要性(SHAP值的平均絕對值)
    importance_df = pd.DataFrame({
         'feature':feature_names,
         'mean_abs_shap':np.abs(shap_values_used).mean(axis = 0)}).sort_values(by='mean_abs_shap',ascending=False)
    print("\n Top 20 重要特徵(by mean|SHAP|):")
    print(importance_df.head(20))
     
    #  特徵重要性繪圖
    plt.figure(figsize=(10,8))
    ax = sns.barplot(
              data = importance_df.head(20),
              x='mean_abs_shap', y='feature'
          )

    plt.title('Feature Importance Model Select(SHAP) - {title_suffix}')
    
    #  在長條圖上顯示數字
    for container in ax.containers:
         ax.bar_label(container, fmt="%d", label_type='edge', padding=3)
    plt.savefig(f"{GRAPH_PATH}/{title_suffix}_SHAP_Feature Importance.png", dpi=300)
    plt.show()

    return shap_values_used, importance_df



# ==================== 混淆矩陣 ====================
def plot_confusion_matrix(y_test, y_pred, model_name, class_names):
    """
    繪製混淆矩陣
    
    參數：
        y_test: 測試目標變數
        y_pred: 預測結果
        model_name: 模型名稱
        class_names: 類別名稱
    """
    import matplotlib.pyplot as plt
    import seaborn as sns
    from config import GRAPH_PATH
    
    from sklearn.metrics import confusion_matrix
    
    cm = confusion_matrix(y_test, y_pred)
    plt.figure(figsize=(10,8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=class_names, yticklabels=class_names)
    plt.title(f'Confusion Matrix - {model_name}')
    plt.ylabel('True')
    plt.xlabel('Predicted')
    plt.savefig(f"{GRAPH_PATH}/{model_name}_confusion_matrix.png", dpi=300)
    plt.show() 
    
    
    
    
    
    
    
    
    