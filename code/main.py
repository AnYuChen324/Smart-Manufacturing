#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Oct 13 10:04:55 2025

@author: anyuchen
"""
"""
main.py：主程式進入點
串接所有模組，執行完整的機器學習管道
"""

import warnings
warnings.filterwarnings('ignore')

from config import TRAIN_TEST_SIZE, GRAPH_PATH
from data_loading import load_manufacture_data, display_basic_info
from data_preprocessing import (
    generate_derived_features, prepare_model_data,
    train_test_split_timeseries, encode_features
)
from eda import run_eda
from model_train import quick_model_compare
from pso_optimization import optimize_model_with_pso, auto_potimize_model, optimize_all_models
from final_model_train import (
    train_final_model,
    check_model_support_feature_importance,
    check_model_support_shap
)
from visualization import (
    plot_multiple_roc, plot_prc_curve, plot_feature_importance,
    plot_shap_analysis, plot_confusion_matrix
)
import pandas as pd

# ==================== 主程式 ====================
def main():
    """
    完整的機器學習管道
    
    流程：
        1. 資料載入
        2. 資料預處理（特徵衍生、編碼）
        3. 探索性資料分析 (EDA)
        4. 模型訓練（GridSearchCV）
        5. 模型比較
        6. PSO 最優化
        7. 最終模型訓練與評估
        8. 結果視覺化
    """
    
    print("\n" + "="*70)
    print(" "*15 + "智慧製造 - 機器學習管道")
    print("="*70)
    
    
    # ===== 1. 資料載入 =====
    print("\n[第 1 步] 資料載入")
    print("-" * 70)
    manufacture_data = load_manufacture_data()
    display_basic_info(manufacture_data)
    
    
    # ===== 2. 資料預處理 =====
    print("\n[第 2 步] 資料預處理")
    print("-" * 70)
    
    # 2.1 生成衍生特徵
    manufacture_data = generate_derived_features(manufacture_data)
    
    # 2.2 準備模型資料（分離特徵與目標變數，但尚未 encode）
    X, y = prepare_model_data(manufacture_data)
    
    # 2.3 時間序列切分（先切再 encode，避免資料洩漏）
    X_train, X_test, y_train, y_test = train_test_split_timeseries(
        X, y, train_ratio=TRAIN_TEST_SIZE
    )
    
    # 2.4 編碼特徵（只在訓練集上 fit，用於轉換測試集）
    X_train, X_test, y_train, y_test, le, preprocessor = encode_features(
        X_train, X_test, y_train, y_test
    )
    
    
    # ===== 3. 探索性資料分析 (EDA) =====
    print("\n[第 3 步] 探索性資料分析 (EDA)")
    print("-" * 70)
    run_eda(manufacture_data, X_train, y_train)

    
    # ===== 4. 模型訓練：模型初選 =====
    # 只先做模型初選，因此不使用tscv
    print("\n[第 4 步] 模型初選（快速比較）")
    print("-" * 70)
    results_model_original, best_model_row = quick_model_compare(X_train, X_test, y_train, y_test)
    best_model_name = best_model_row['Model']
    
    print(f"\n初選最佳模型為： {best_model_name}")
    print(f"✅ F1 Macro 分數：{best_model_row['F1_macro']:.4f}")
    
    
    # ===== 5. 超參數優化(PSO支援所有模型) =====
    print("\n[第 5 步] PSO 最優化")
    print("-" * 70)

    
    best_params = auto_potimize_model(
        model_name = best_model_name,
        X_train = X_train,
        y_train = y_train,
        enable_pso = True,   # 設為 False 可以跳過 PSO，直接用預設參數
        swarmsize = 20,  # 粒子數（可調整）
        maxiter =10  # 迭代次數（可調整）
    )
    
    if best_params is not None:
        print("\n PSO 優化完成！找到最佳參數：")
        # 新增顯示優化後的參數
        for k , v in best_params.items():
            print(f" {k}:{v}")
    else:
        print(f"\n 未執行 PSO  優化，將使用 {best_model_name} 的預設參數")
    
    
    # ===== 6. 訓練最終模型 =====
    print("\n[第 6 步] 訓練最終模型")
    print("-" * 70)
    
    # 統一調用 train_final_model(),回傳results
    #train_final_model(model_name, X_train, X_test, y_train, y_test, le, best_params=None):
    final_results = train_final_model(
        model_name = best_model_name,
        X_train = X_train,
        X_test = X_test,
        y_train = y_train,
        y_test = y_test,
        le = le,
        best_params = best_params
    )
    
    # 結果解包
    final_model = final_results['model']
    y_pred = final_results['y_pred']
    y_prob = final_results['y_prob']
    y_test_enc = final_results['y_test']
    accuracy = final_results['accuracy']
    f1_macro = final_results['f1_macro']
    feature_names = final_results['feature_names']    
    
    
    # ===== 7. 結果視覺化 =====
    print("\n[第 7 步] 結果視覺化")
    print("-" * 70)
    
    # 包裝視覺化函數需要的格式
    viz_results = {
        f'{best_model_name}_Final':{
            'model': final_model,
            'y_prob': y_prob,
            'y_pred': y_pred
        }
    }
    
    # 7.1 ROC 曲線
    print("\n繪製 ROC 曲線...")
    plot_multiple_roc(viz_results, y_test_enc, title_suffix=f"{best_model_name}_Final_ROC")
    
    # 7.2 PRC 曲線
    print("\n繪製 PRC 曲線...")
    plot_prc_curve(viz_results, y_test_enc, title_suffix=f"{best_model_name}_Final_PRC")
    
    # 7.3 特徵重要性
    if check_model_support_feature_importance(best_model_name):
        print("\n繪製特徵重要性...")
        feature_importance_df = plot_feature_importance(
            final_model,
            feature_names,
            title_suffix=f"{best_model_name}_Final",
            top_n=20
        )
    else:
        print(f" {best_model_name} 不支援特徵重要性分析")
        feature_importance_df = None
    
    # 7.4 SHAP 分析(僅樹模型)
    if check_model_support_shap(best_model_name):
        print("\n執行 SHAP 分析...")
        # 使用try/except的原因是即使Shap失敗，程式還能繼續
        try:
            shap_values_used, shap_importance_df = plot_shap_analysis(
                final_model,
                X_train,
                feature_names,
                title_suffix=f"{best_model_name}_Final_Shap"
            )
        except Exception as e:
            print(f"⚠️ SHAP 分析失敗：{e}")
            shap_values_used, shap_importance_df = None, None
    else:
        print(f" {best_model_name} 不支援 Shap 分析")
        shap_values_used, shap_importance_df = None, None
        
        
    # 7.5 混淆矩陣
    print("\n繪製混淆矩陣...")
    plot_confusion_matrix(
        y_test_enc, 
        y_pred,
        model_name = best_model_name,
        class_names=le.classes_
    )
    
    # ===== 8. 總結 =====
    print("\n" + "="*70)
    print(" "*20 + "✅ 完整管道執行完成！")
    print("="*70)
    
    print(f"\n📊 最終結果摘要：")
    print(f"  最佳模型：{best_model_name}")
    print(f"  準確度：{accuracy:.4f}")
    print(f"  Macro F1：{f1_macro:.4f}")
    
    if best_params is not None:
        print(f"\n🔧 PSO 優化參數（已使用）：")
        for k, v in best_params.items():
            if k in ['learning_rate', 'n_estimators', 'max_depth', 'num_leaves']:
                print(f"  {k}: {v}")
    else:
        print(f"\n🔧 使用模型預設參數（未執行 PSO）")
    
    print(f"\n📁 所有圖表已保存至：{GRAPH_PATH}")
    print(f"\n💡 提示：所有變數已回傳，可在 Console 中查看")
    
    
    # ===== 9. 回傳所有關鍵變數 =====
    return {
        'X_train': X_train,
        'X_test': X_test,
        'y_train': y_train,
        'y_test': y_test,
        'le': le,
        'preprocessor': preprocessor,
        'final_model': final_model,
        'best_model_name': best_model_name,
        'y_pred': y_pred,
        'y_prob': y_prob,
        'accuracy': accuracy,
        'f1_macro': f1_macro,
        'feature_names': feature_names,
        'best_params': best_params,
        'feature_importance_df': feature_importance_df if check_model_support_feature_importance(best_model_name) else None,
        'shap_importance_df': shap_importance_df if check_model_support_shap(best_model_name) else None
    }

# ==================== 執行點 ====================
if __name__ == "__main__":
    
    # 執行主程式，回傳所有變數
    results = main()
    
    # 🎯 現在您可以輕鬆查看任何變數！
    print("\n" + "="*70)
    print("📊 變數檢查範例：")
    print("="*70)
    print(f"  results['best_model_name'] = {results['best_model_name']}")
    print(f"  results['accuracy'] = {results['accuracy']:.4f}")
    print(f"  results['X_train'].shape = {results['X_train'].shape}")
    print(f"  results['final_model'] 類型 = {type(results['final_model'])}")
    print(f"\n💡 提示：使用 results['變數名'] 來查看任何回傳的變數")
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    