import os
import pandas as pd
import glob

def clean_file(filepath):
    # Read the excel file
    df = pd.read_excel(filepath, header=None)
    
    # Extract metadata
    sayac_kodu = None
    sayac_seri_no = None
    donem = None
    
    for i in range(20):
        row_vals = [str(x) for x in df.iloc[i].values if pd.notna(x)]
        for j, val in enumerate(row_vals):
            if "Sayaç Kodu" in val and j+1 < len(row_vals):
                sayac_kodu = row_vals[j+1]
            if "Sayaç Seri" in val and j+1 < len(row_vals):
                sayac_seri_no = row_vals[j+1]
            if "Uzlaştırma Dönemi" in val and j+1 < len(row_vals):
                donem = row_vals[j+1]
                
    # Find the row where "Saat/Gün" starts
    start_row = -1
    for i in range(15, 30):
        if pd.notna(df.iloc[i, 0]) and "Saat/Gün" in str(df.iloc[i, 0]):
            start_row = i
            break
            
    if start_row == -1:
        print(f"Could not find data table in {filepath}")
        return pd.DataFrame()
        
    # The next rows contain the data. Usually there's a "00-01" etc.
    # Because of merged cells, time labels are on "Veriş" rows, and "Çekiş" rows might be shifted or below them.
    # We will iterate through rows from start_row + 1 to the end, looking for "Çekiş"
    
    records = []
    current_time = "Unknown"
    
    for i in range(start_row + 1, len(df)):
        row = df.iloc[i].values
        # If the first column is a time range like "00-01"
        col0 = str(row[0]).strip() if pd.notna(row[0]) else ""
        col1 = str(row[1]).strip() if pd.notna(row[1]) else ""
        
        # Check if first column has time format
        if '-' in col0 and len(col0) <= 5 and col0[0].isdigit():
            current_time = col0
            type_val = col1
            data_start_idx = 2
        else:
            # It might be in col0 (if shifted) or it might just be the type
            if "Çekiş" in col0:
                type_val = "Çekiş"
                data_start_idx = 1
            elif "Çekiş" in col1:
                type_val = "Çekiş"
                data_start_idx = 2
            else:
                type_val = col0
                data_start_idx = 1
                
        if "Çekiş" in type_val:
            # Here are the days
            for d in range(1, 32):
                idx = data_start_idx + d - 1
                if idx < len(row):
                    val = row[idx]
                    if pd.notna(val) and str(val).strip() != '':
                        try:
                            val_float = float(val)
                            records.append({
                                'Sayaç_Kodu': sayac_kodu,
                                'Sayaç_Seri_No': sayac_seri_no,
                                'Dönem': donem,
                                'Gün': d,
                                'Saat': current_time,
                                'Çekiş': val_float
                            })
                        except:
                            pass
                            
    return pd.DataFrame(records)

folder = os.path.join(os.path.dirname(os.path.dirname(__file__)), "data", "raw")
all_files = glob.glob(os.path.join(folder, "*.xlsx"))

df_list = []
for f in all_files:
    print(f"Processing {os.path.basename(f)}...")
    df_clean = clean_file(f)
    if not df_clean.empty:
        df_list.append(df_clean)

if df_list:
    final_df = pd.concat(df_list, ignore_index=True)
    out_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), "data", "processed", "cleaned_asos_data.csv")
    final_df.to_csv(out_path, index=False)
    print(f"Saved cleaned data to {out_path} with {len(final_df)} records.")
else:
    print("No data extracted.")
