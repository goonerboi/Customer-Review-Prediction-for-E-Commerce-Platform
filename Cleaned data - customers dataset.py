# -*- coding: utf-8 -*-
"""
Created on Wed Nov 13 13:18:26 2024

@author: Admin
"""
import pandas as pd

def process_zip_codes(file_path):
    data = pd.read_csv(file_path)
    missing_values_na = data.isna().sum()
    missing_values_null = data.isnull().sum()
    data['customer_zip_code_prefix'] = data['customer_zip_code_prefix'].astype(str)
    data['customer_zip_code_prefix'] = data['customer_zip_code_prefix'].apply(lambda x: x.zfill(5) if len(x) == 4 else x)
    return data, missing_values_na, missing_values_null