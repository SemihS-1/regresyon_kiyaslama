import numpy as np
import pandas as pd
from sklearn.datasets import load_diabetes
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
import matplotlib.pyplot as plt

print("\nVeri seti yükleniyor...\n")

# Veriyi yükle
diabetes = load_diabetes()

# Veriyi Pandas DataFrame'e çevir
df = pd.DataFrame(diabetes.data, columns=diabetes.feature_names)

# Hedef değişkeni ekle (target)
df['target'] = diabetes.target

print("Çoklu lineer regresyon veri seti oluşturuldu.\n")
print(df.head())

# Çoklu Lineer Regresyon için tüm bağımsız değişkenler kullanılıyor
print("\nÇoklu Lineer Regresyon için tüm özellikler seçiliyor...\n")
X_all = df.drop('target', axis=1)  # Tüm bağımsız değişkenler
y = df['target']

# Veriyi eğitim ve test setlerine ayır
X_train_all, X_test_all, y_train_all, y_test_all = train_test_split(X_all, y, test_size=0.3, random_state=42)

print(f"Eğitim seti boyutu: {X_train_all.shape}")
print(f"Test seti boyutu: {X_test_all.shape}")

# Modeli oluştur ve eğit
print("\nÇoklu Lineer Regresyon modeli eğitiliyor...\n")
model_all = LinearRegression()
model_all.fit(X_train_all, y_train_all)

# Model parametreleri
print("\nModel eğitildi.\n")
print("Model katsayıları:")
print(model_all.coef_)
print("Model kesişim noktası:")
print(model_all.intercept_)

# Tahmin yap
y_pred_all = model_all.predict(X_test_all)

# Gerçek vs tahmin edilen değerler
print("\nGerçek vs Tahmin edilen değerler (Tüm Özellikler):")
for gerçek, tahmin in zip(y_test_all, y_pred_all):
    print(f"Gerçek: {gerçek:.2f}, Tahmin: {tahmin:.2f}")

# Model performansı
print("\nÇoklu Lineer Regresyon modelinin performansı değerlendiriliyor...\n")
mse_all = mean_squared_error(y_test_all, y_pred_all)
r2_all = r2_score(y_test_all, y_pred_all)
print(f"Ortalama Kare Hatası (MSE): {mse_all:.4f}")
print(f"R-Kare skoru: {r2_all:.4f}")

# Çoklu Lineer Regresyon görselleştirmesi
plt.scatter(y_test_all, y_pred_all, color='green')
plt.xlabel('Gerçek Değerler (Tüm Özellikler)')
plt.ylabel('Tahmin Edilen Değerler')
plt.title('Çoklu Lineer Regresyon: Gerçek vs Tahmin (Tüm Özellikler)')
plt.grid(True)
plt.savefig('coklu_lineer_regresyon_all.png')
print("\nÇoklu Lineer Regresyon tahminleri kaydedildi.\n")
