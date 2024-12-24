# -*- coding: utf-8 -*-
"""
Created on Mon Dec 16 14:35:20 2024

@author: erena

Bu kod, 0-1 knapsack (sırt çantası) problemini çözmek için optimize fonksiyonu ile yapay arı algoritmasını uygular. 
Her örnek için en iyi çözüm ve yakınsama eğrisi oluşturulur.
"""

import numpy as np
import matplotlib.pyplot as plt

def obj(X, W, P, C):
    """
    Amaç fonksiyonu: Pozitif değerlendirme için hesaplama yapılır.
    C (kapasite) kısıtı ihlal edilirse ceza uygulanır.

    Args:
        X (ndarray): Çözüm vektörü.
        W (ndarray): Ağırlık vektörü.
        P (ndarray): Değer vektörü.
        C (int): Çantanın kapasitesi.

    Returns:
        float: Amaç fonksiyonunun sonucu.
    """
    f = np.sum(P * X)  # Toplam değer
    g = np.sum(W * X) - C  # Kapasite kısıtı
    r = 10**3  # Ceza katsayısı
    phi = f - r * max(0, g)**2  # Ceza terimi ile birlikte amaç fonksiyonu
    return phi

def optimize(w, p, c, n=10, d=5, T=100):
    """
    Yapay arı algoritması kullanarak knapsack problemini optimize eder.

    Args:
        w (ndarray): Ağırlık vektörü.
        p (ndarray): Değer vektörü.
        c (int): Çantanın kapasitesi.
        n (int): Arı sayısı (çözüm sayısı).
        d (int): Çözüm boyutu (ürün sayısı).
        T (int): Maksimum iterasyon sayısı.

    Returns:
        ndarray: En iyi çözüm vektörü.
        float: En iyi çözümün değeri.
        ndarray: Yakınsama verileri.
    """
    np.random.seed(0)  # Rastgelelik için sabit tohum
    fi = np.random.choice([0, 1], size=(n, d))
    fii = np.random.choice([0, 1], size=(n, d))
    x = np.random.choice([0, 1], size=(n, d))
    fit = np.empty(n)

    # Başlangıç uygunluk değerlerini hesapla
    for i in range(n):
        fit[i] = obj(x[i], w, p, c)

    zeros = np.zeros(n)  # Hareketsiz arı sayacı
    lim = 10  # Maksimum hareketsizlik sınırı
    f = np.zeros(T)  # Yakınsama verilerini kaydetmek için dizi

    enIyiCozum = x[np.argmax(fit)].copy()
    enIyiKar = max(fit)
    uygunluk = np.empty(n)
    v = np.empty((n, d))

    # İterasyon döngüsü
    for t in range(T):
        for i in range(n):
            k = np.random.randint(0, n)
            while i == k:
                k = np.random.randint(0, n)

            # Yeni çözüm üret
            kısıt1 = np.logical_or(x[i], x[k]).astype(int)
            kısıt2 = np.logical_and(fi[i], kısıt1).astype(int)
            v[i] = np.logical_and(fii[i], kısıt2).astype(int)

            # Yeni çözümün uygunluğunu değerlendir
            fit_val = obj(v[i], w, p, c)
            if fit_val > fit[i]:
                fit[i] = fit_val
                x[i] = v[i].copy()
                zeros[i] = 0
            else:
                zeros[i] += 1

            # En iyi çözümü güncelle
            if fit_val > enIyiKar:
                enIyiKar = fit_val
                enIyiCozum = v[i].copy()

        # Uygunluk değerlerini hesapla
        for i in range(n):
            if fit[i] > 0:
                uygunluk[i] = 1 + fit[i]
            else:
                uygunluk[i] = 1 / (1 - fit[i])

        p_vals = uygunluk / np.sum(uygunluk)

        # İzci arıların hareketi
        for ii in range(n):
            i = np.random.choice(n, size=1, replace=True, p=p_vals)[0]
            k = np.random.randint(0, n)
            while i == k:
                k = np.random.randint(0, n)

            v = x.copy()
            v[k] = 1 - v[k]

            a_val = obj(v[i], w, p, c)
            if fit_val > fit[i]:
                fit[i] = fit_val
                x[i] = v[i].copy()
                zeros[i] = 0
            else:
                zeros[i] += 1

            if a_val > enIyiKar:
                enIyiKar = a_val
                enIyiCozum = v[i].copy()

        # Hareketsiz arılar için rastgele yeniden başlatma
        for i in range(n):
            if zeros[i] > lim:
                x[i] = np.random.choice([0, 1], size=d)
                fit[i] = obj(x[i], w, p, c)
                zeros[i] = 0

        f[t] = enIyiKar  # Yakınsama verisini kaydet

    return enIyiCozum, enIyiKar, f

# Test örnekleri
ornekler = [
    {
        "W": np.array([21, 33, 5, 7, 1]),  
        "P": np.array([10, 20, 30, 40, 50]),  
        "C": 10  
    },
    
    {
        "W": np.array([15, 25, 35, 45, 5]),  
        "P": np.array([5, 10, 15, 20, 25]),  
        "C": 20  
    },
    
    {
        "W": np.array([10, 20, 30, 40, 50]),  
        "P": np.array([1, 2, 3, 4, 5]), 
        "C": 15 
    },

    {
        "W": np.array([5, 10, 15, 20, 25]), 
        "P": np.array([50, 40, 30, 20, 10]), 
        "C": 25  
    },

    {
        "W": np.array([12, 24, 36, 48, 60]), 
        "P": np.array([3, 6, 9, 12, 15]),
        "C": 30  
    }
]

# Sonuçları saklamak için matrisler
data = np.empty([30, 5])
solution = np.empty([5, 5])

# Yakınsama eğrisi çizimi
plt.figure(figsize=(10, 6))  # Grafik boyutu

for i, ornek in enumerate(ornekler):
    W = ornek["W"]
    P = ornek["P"]
    C = ornek["C"]
    all_f = []  # Yakınsama verilerini toplamak için liste

    for r in range(30):
        cozum, deger, f = optimize(W, P, C)
        data[(r, i)] = deger  # Pozitif değerlendirme
        all_f.append(f)

    # Yakınsama eğrisi
    avg_f = np.mean(all_f, axis=0)  # Her iterasyonun ortalaması
    plt.plot(avg_f, label=f"Ornek {i + 1}")

    print(f"\nOrnek {i + 1}:")
    print("Ağırlıklar (W):", W)
    print("Değerler (P):", P)
    print("Kapasite (C):", C)
    print("En iyi Çözüm:", cozum)
    print("En Yüksek Değer:", deger)

plt.title("Yakınsama Eğrisi")
plt.xlabel("Iterasyon")
plt.ylabel("En İyi Değer")
plt.legend()
plt.grid()
plt.show()

# İstatistiksel analiz
std_value = np.std(data, axis=0)
max_value = np.max(data, axis=0)
min_value = np.min(data, axis=0) 
mean_value = np.mean(data, axis=0) 

print("-------------------------------------------")
print(f"En Yüksek Değer (Ortalama): {mean_value}")    
print(f"Standart Sapma: {std_value}")
print(f"Max: {max_value}")
print(f"Min: {min_value}")
print(f"Çözüm değerleri : {solution}")
