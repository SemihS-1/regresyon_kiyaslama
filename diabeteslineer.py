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

print("Lineer regresyon veri seti oluşturuldu.\n")
print(df.head())

# Basit Lineer Regresyon için sadece BMI sütunu kullanıyoruz
print("\nBasit Lineer Regresyon için BMI verisi seçiliyor...\n")
X_bmi = df[['bmi']]
y = df['target']

# Veriyi eğitim ve test setlerine ayır
X_train, X_test, y_train, y_test = train_test_split(X_bmi, y, test_size=0.35, random_state=42)

print(f"Eğitim seti boyutu: {X_train.shape}")
print(f"Test seti boyutu: {X_test.shape}")

# Modeli oluştur ve eğit
print("\nBasit Lineer Regresyon modeli eğitiliyor...\n")
model_bmi = LinearRegression()
model_bmi.fit(X_train, y_train)

# Model parametreleri
print("\nModel eğitildi.\n")
print("BMI için model katsayıları:")
print(model_bmi.coef_)
print("Model kesişim noktası:")
print(model_bmi.intercept_)

# Tahmin yap
y_pred_bmi = model_bmi.predict(X_test)

# Gerçek vs tahmin edilen değerler
print("\nGerçek vs Tahmin edilen değerler (BMI):")
for gerçek, tahmin in zip(y_test, y_pred_bmi):
    print(f"Gerçek: {gerçek:.2f}, Tahmin: {tahmin:.2f}")

# Model performansı
print("\nBasit Lineer Regresyon modelinin performansı değerlendiriliyor...\n")
mse_bmi = mean_squared_error(y_test, y_pred_bmi)
r2_bmi = r2_score(y_test, y_pred_bmi)
print(f"Ortalama Kare Hatası (MSE): {mse_bmi:.4f}")
print(f"R-Kare skoru: {r2_bmi:.4f}")

# Basit Lineer Regresyon görselleştirmesi (BMI)
plt.scatter(y_test, y_pred_bmi, color='blue')
plt.xlabel('Gerçek Değerler (BMI)')
plt.ylabel('Tahmin Edilen Değerler')
plt.title('Basit Lineer Regresyon: Gerçek vs Tahmin (BMI)')
plt.grid(True)

plt.clf()
print("\nBasit Lineer Regresyon tahminleri kaydedildi.\n")

