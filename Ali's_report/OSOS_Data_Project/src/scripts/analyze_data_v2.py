# -*- coding: utf-8 -*-
import pandas as pd
import numpy as np
import json
import os

pd.set_option('display.max_columns', None)
pd.set_option('display.width', 2000)
pd.set_option('display.max_colwidth', 100)

data_dir = r'c:\Users\Ali Gökay Bozok\OneDrive - Kadir Has University\Masaüstü\BEDAŞ Bitirme Projesi\OSOS'
os.chdir(data_dir)

results = {}

# Analyze File 1: November OSF
print("=" * 80)
print("ANALYZING: osf_2193681000_11_2025_kiifa.xlsx")
print("=" * 80)

df1 = pd.read_excel('osf_2193681000_11_2025_kiifa.xlsx', header=None)
print(f"Total rows: {len(df1)}, Total columns: {len(df1.columns)}")

# Print first 20 rows to understand structure
for i in range(min(20, len(df1))):
    row_data = df1.iloc[i].tolist()
    # Filter out NaN values for cleaner display
    row_clean = [str(x) if pd.notna(x) else '' for x in row_data]
    print(f"Row {i}: {row_clean[:15]}")  # First 15 columns

print("\n" + "=" * 80)
print("ANALYZING: osf_2193681000_12_2025_ak4zd.xlsx")
print("=" * 80)

df2 = pd.read_excel('osf_2193681000_12_2025_ak4zd.xlsx', header=None)
print(f"Total rows: {len(df2)}, Total columns: {len(df2.columns)}")

for i in range(min(20, len(df2))):
    row_data = df2.iloc[i].tolist()
    row_clean = [str(x) if pd.notna(x) else '' for x in row_data]
    print(f"Row {i}: {row_clean[:15]}")

print("\n" + "=" * 80)
print("ANALYZING: rpt-101_endeks_raporu_(_tesisat_bazli_)_M1mgk (1).xlsx")
print("=" * 80)

df3 = pd.read_excel('rpt-101_endeks_raporu_(_tesisat_bazli_)_M1mgk (1).xlsx', header=None)
print(f"Total rows: {len(df3)}, Total columns: {len(df3.columns)}")

for i in range(min(25, len(df3))):
    row_data = df3.iloc[i].tolist()
    row_clean = [str(x) if pd.notna(x) else '' for x in row_data]
    print(f"Row {i}: {row_clean[:10]}")

print("\n" + "=" * 80)
print("DATA SUMMARY")
print("=" * 80)
print(f"File 1 (Nov): {df1.shape}")
print(f"File 2 (Dec): {df2.shape}")
print(f"File 3 (Endeks): {df3.shape}")
