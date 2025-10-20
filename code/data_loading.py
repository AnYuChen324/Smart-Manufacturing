#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Oct 11 16:10:14 2025

@author: anyuchen
"""

"""
01_data_loading.py：資料讀取與基本初始化
"""
import pandas as pd
from config import DATA_PATH

def load_manufacture_data():
    """
    讀取製造業資料
    
    回傳：
        pd.DataFrame: 讀取的資料框
    
    流程：
        1. 從 CSV 讀取資料
        2. 轉換時間戳為 datetime 格式
        3. 按照機器編號與時間排序
    """
    print("Loading manufacturing data...")
    
    manufacture_data = pd.read_csv(DATA_PATH)
    
    #  將資料照時間排序，需要先把資料轉成datetime格式
    manufacture_data['Timestamp'] = pd.to_datetime(manufacture_data['Timestamp'])
    #  將資料根據時間還有機台ID排列
    manufacture_data = manufacture_data.sort_values(['Machine_ID', 'Timestamp'])
    
    print(f"✓ Data loaded successfully!")
    print(f" Shape: {manufacture_data.shape}")
    print(f" Columns: {manufacture_data.columns.tolist()}")
    
    return manufacture_data

def display_basic_info(manufacture_data):
    """
    顯示資料基本資訊
    
    參數：
        manufacture_data (pd.DataFrame): 讀取的資料框
    """
    print("\n=== 資料基本資訊 ===")
    print(f"總筆數: {len(manufacture_data)}")
    print(f"總欄數: {len(manufacture_data.columns)}")
    print(f"\n欄位名稱與型別：")
    print(manufacture_data.dtypes)
    print(f"\n前 5 筆資料：")
    print(manufacture_data.head())
    print(f"\n資料敘述統計：")
    print(manufacture_data.describe())
    print(f"\n缺失值情況：")
    print(manufacture_data.isnull().sum())