# -*- coding: utf-8 -*-
import pandas as pd
import json

filepath = r"c:\Users\Ali Gökay Bozok\OneDrive - Kadir Has University\Masaüstü\BEDAŞ Bitirme Projesi\OSOS\raw_data\osf_2193681000_01_2025_yn30j.xlsx"
df = pd.read_excel(filepath, header=None)

output = {}
for i in range(len(df)):
    row = [str(x) if pd.notna(x) else '' for x in df.iloc[i].tolist()]
    cleaned_row = [x for x in row if x != '']
    if len(cleaned_row) > 0:
        output[f"Row {i}"] = cleaned_row

with open(r"c:\Users\Ali Gökay Bozok\OneDrive - Kadir Has University\Masaüstü\BEDAŞ Bitirme Projesi\OSOS\inspect_raw.json", "w", encoding="utf-8") as f:
    json.dump(output, f, ensure_ascii=False, indent=2)
