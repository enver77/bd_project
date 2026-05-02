import pandas as pd
import os

# Set display options
pd.set_option('display.max_columns', None)
pd.set_option('display.width', 1000)
pd.set_option('display.max_colwidth', 50)

# List all Excel files
data_dir = r'c:\Users\Ali Gökay Bozok\OneDrive - Kadir Has University\Masaüstü\BEDAŞ Bitirme Projesi\OSOS'
files = [f for f in os.listdir(data_dir) if f.endswith('.xlsx')]

print("=" * 100)
print("OSOS DATA ANALYSIS FOR LUMINAIRE FAILURE PREDICTION")
print("=" * 100)

for file in files:
    filepath = os.path.join(data_dir, file)
    print(f"\n{'='*100}")
    print(f"FILE: {file}")
    print(f"{'='*100}")
    
    # Try different header settings
    try:
        # First, read without header to see structure
        df_raw = pd.read_excel(filepath, header=None)
        print(f"\nShape: {df_raw.shape}")
        print(f"\n--- RAW DATA (First 15 rows) ---")
        print(df_raw.head(15).to_string())
        
        # Show last few rows too
        print(f"\n--- LAST 5 ROWS ---")
        print(df_raw.tail(5).to_string())
        
        # Try to identify header row
        print(f"\n--- COLUMN INFO (from row 0-3) ---")
        for i in range(min(4, len(df_raw))):
            print(f"Row {i}: {df_raw.iloc[i].tolist()}")
            
    except Exception as e:
        print(f"Error reading file: {e}")

print("\n" + "=" * 100)
print("ANALYSIS COMPLETE")
print("=" * 100)
