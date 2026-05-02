import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os

# Set pandas display options
pd.set_option('display.max_columns', None)
pd.set_option('display.width', 1000)

csv_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), "data", "processed", "cleaned_asos_data.csv")

# 1. Veriyi Oku
df = pd.read_csv(csv_path)

print("="*50)
print("VERİ ÇERÇEVESİ (DATAFRAME) BİLGİSİ")
print("="*50)
df.info()

print("\n" + "="*50)
print("EKSİK (NULL) VERİ KONTROLÜ")
print("="*50)
print(df.isnull().sum())

print("\n" + "="*50)
print("İLK 10 SATIR (ÖRNEK VERİ)")
print("="*50)
print(df.head(10))

print("\n" + "="*50)
print("TEMEL İSTATİSTİKLER (Sadece Sayısal)")
print("="*50)
print(df.describe())

# 2. Veri Tiplerini Düzenle
# Çekiş değerlerinin sayısal olduğundan emin olalım
df['Çekiş'] = pd.to_numeric(df['Çekiş'], errors='coerce')

# 3. Görselleştirme
print("\nGrafikler oluşturuluyor ve PNG olarak kaydediliyor...")

sns.set_theme(style="whitegrid")

# --- Grafik 1: Aylara Göre Toplam Çekiş (Tüketim) ---
if 'Dönem' in df.columns:
    plt.figure(figsize=(12, 6))
    monthly_sum = df.groupby('Dönem')['Çekiş'].sum().reset_index()
    # Dönem değerlerini metin sıralamasına bırakalım
    sns.barplot(data=monthly_sum, x='Dönem', y='Çekiş', palette='viridis')
    plt.title('Dönemlere (Aylara) Göre Toplam Tüketim', fontsize=14, fontweight='bold')
    plt.xlabel('Dönem', fontsize=12)
    plt.ylabel('Toplam Çekiş (kWh)', fontsize=12)
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig(os.path.join(os.path.dirname(os.path.dirname(__file__)), "outputs", "visualizations", "aylik_toplam_tuketim.png"), dpi=150)
    plt.close()

# --- Grafik 2: Saatlik Ortalama Çekiş (Profilleme) ---
if 'Saat' in df.columns:
    plt.figure(figsize=(14, 6))
    hourly_avg = df.groupby('Saat')['Çekiş'].mean().reset_index()
    sns.barplot(data=hourly_avg, x='Saat', y='Çekiş', palette='magma')
    plt.title('Saat Dilimlerine Göre Ortalama Tüketim', fontsize=14, fontweight='bold')
    plt.xlabel('Saat Dilimi', fontsize=12)
    plt.ylabel('Ortalama Çekiş (kWh)', fontsize=12)
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig(os.path.join(os.path.dirname(os.path.dirname(__file__)), "outputs", "visualizations", "saatlik_ortalama_tuketim.png"), dpi=150)
    plt.close()

print("\nBAŞARILI: Görselleştirme dosyaları ('aylik_toplam_tuketim.png' ve 'saatlik_ortalama_tuketim.png') kaydedildi.")
