# -*- coding: utf-8 -*-
import pandas as pd
import numpy as np
import json
import os

data_dir = r'c:\Users\Ali Gökay Bozok\OneDrive - Kadir Has University\Masaüstü\BEDAŞ Bitirme Projesi\OSOS'
os.chdir(data_dir)

# Create a comprehensive analysis dictionary
analysis_results = {}

# ============== FILE 1: November OSF ==============
df1 = pd.read_excel('osf_2193681000_11_2025_kiifa.xlsx', header=None)
analysis_results['file1'] = {
    'filename': 'osf_2193681000_11_2025_kiifa.xlsx',
    'shape': list(df1.shape),
    'headers': [],
    'sample_data': []
}

# Extract meaningful rows
for i in range(min(15, len(df1))):
    row = [str(x) if pd.notna(x) else 'NaN' for x in df1.iloc[i].tolist()]
    analysis_results['file1']['sample_data'].append(row[:12])

# ============== FILE 2: December OSF ==============
df2 = pd.read_excel('osf_2193681000_12_2025_ak4zd.xlsx', header=None)
analysis_results['file2'] = {
    'filename': 'osf_2193681000_12_2025_ak4zd.xlsx',
    'shape': list(df2.shape),
    'sample_data': []
}

for i in range(min(15, len(df2))):
    row = [str(x) if pd.notna(x) else 'NaN' for x in df2.iloc[i].tolist()]
    analysis_results['file2']['sample_data'].append(row[:12])

# ============== FILE 3: Endeks Report ==============
df3 = pd.read_excel('rpt-101_endeks_raporu_(_tesisat_bazli_)_M1mgk (1).xlsx', header=None)
analysis_results['file3'] = {
    'filename': 'rpt-101_endeks_raporu_(_tesisat_bazli_)_M1mgk (1).xlsx',
    'shape': list(df3.shape),
    'sample_data': []
}

for i in range(min(20, len(df3))):
    row = [str(x) if pd.notna(x) else 'NaN' for x in df3.iloc[i].tolist()]
    analysis_results['file3']['sample_data'].append(row[:10])

# Save as JSON for easy reading
with open('data_analysis_results.json', 'w', encoding='utf-8') as f:
    json.dump(analysis_results, f, ensure_ascii=False, indent=2)

print("Analysis complete. Results saved to data_analysis_results.json")
print(f"File 1 shape: {df1.shape}")
print(f"File 2 shape: {df2.shape}")
print(f"File 3 shape: {df3.shape}")
