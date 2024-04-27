import numpy as np
import matplotlib.pyplot as plt

import sounddevice as sd
from scipy.io.wavfile import write
from scipy.io.wavfile import read


def kaydet(sure, dosya_adi):
    fs = 44100  # Örnekleme frekansı
    print("Kayıt başladı. Konuşmaya başlayın...")

    # Ses kaydı
    ses_kaydi = sd.rec(int(sure * fs), samplerate=fs, channels=1)
    sd.wait()  # Kayıt tamamlanana kadar bekleyin

    print("Kayıt tamamlandı.")

    # Ses dosyasını kaydet
    write(dosya_adi, fs, ses_kaydi)

    print("Ses dosyası kaydedildi:", dosya_adi)

def myConv(x,y):
    # Konvolüsyon sonucunu saklayacak bir dizi oluştur
    result = [0] * (len(x) + len(y) - 1)

    # Konvolüsyon işlemi
    for i in range(len(x)):
        for j in range(len(y)):
            result[i + j] += x[i] * y[j]

    return result

def lastConv(array, m, A):
    result1 = [0] * len(array)  # Sonuç listesi tanımlandı
    for n in range(len(array)):
        result = 0  # Her n değeri için result sıfırlanmalı
        result1[n] = array[n]
        for k in range(1,m):
            result += (A ** (-k)) * (k * array[n - 3000 * k])  # Parantez kapatıldı
        result1[n] += result  # Her bir döngü sonucunu result1 listesine eklenmeli
    return result1

# Veri setlerini oluştur
n = int(input("x[n] işaretinin uzunluğunu girin: "))
x1 = []
for i in range(n):
    x1.append(float(input("x[{}] değerini girin: ".format(i))))

m = int(input("y[m] işaretinin uzunluğunu girin: "))
y1 = []
for i in range(m):
    y1.append(float(input("y[{}] değerini girin: ".format(i))))

print(myConv(x1,y1))

conv_result_1 = myConv(x1, y1)
conv_result_1_2 = np.convolve(x1, y1, 'full')

# Grafikleri çizdir
plt.figure(figsize=(12, 10))

plt.subplot(2, 2, 1)
plt.stem(x1)
plt.title('X[n]')
plt.xlabel('n')
plt.ylabel('Amplitude')
plt.grid(True)

plt.subplot(2, 2, 2)
plt.stem(y1)
plt.title('Y[m]')
plt.xlabel('m')
plt.ylabel('Amplitude')
plt.grid(True)

plt.subplot(2, 2, 3)
plt.stem(np.arange(len(x1) + len(y1) - 1), conv_result_1)
plt.title('MyConv Sonucu')
plt.xlabel('n')
plt.ylabel('Amplitude')
plt.grid(True)

plt.subplot(2, 2, 4)
plt.stem(np.arange(len(x1) + len(y1) - 1), conv_result_1_2)
plt.title('Hazır Fonksiyon Sonucu')
plt.xlabel('n')
plt.ylabel('Amplitude')
plt.grid(True)

plt.tight_layout()
plt.show()

# Vektörel gösterim
print("X[n] =", x1)
print("Y[m] =", y1)
print("MyConv Sonucu =", conv_result_1)
print("Hazır Fonksiyon Sonucu =", conv_result_1_2)

# 5 saniyelik ses kaydi
kayit_1 = kaydet(5, 'X1.wav')

# 10 saniyelik ses kaydi
kayit_2 =kaydet(10, 'X2.wav')

fs, data1 = read('X1.wav')
fs2, data2 = read('X2.wav')

A = 2
m = 3
Ak = np.array([A**(-k) for k in range(1, m+1)])

condata1 = lastConv(data1,3,2)
con_result1_1 = np.convolve(condata1, Ak, 'full')

condata2 = lastConv(data2,3,2)
con_result2_1 = np.convolve(condata2, Ak, 'full')

m = 4
Ak = np.array([A**(-k) for k in range(1, m+1)])

condata3 = lastConv(data1,4,2)
con_result3_1 = np.convolve(condata3, Ak, 'full')

condata4 = lastConv(data2,4,2)
con_result4_1 = np.convolve(condata4, Ak, 'full')

m = 5
Ak = np.array([A**(-k) for k in range(1, m+1)])

condata5 = lastConv(data1,5,2)
con_result5_1 = np.convolve(condata5, Ak, 'full')

condata6 = lastConv(data2,5,2)
con_result6_1= np.convolve(condata6, Ak, 'full')

print("kayit 1")
sd.play(data1, fs)
sd.wait()  # Ses dosyasının çalmasını bekleyin

print("kayit 2")

sd.play(data2, fs2)
sd.wait()  # Ses dosyasının çalmasını bekleyin

print("Kayit 1 M = 3 Benim Foknsiyonum")
sd.play(condata1, fs)
sd.wait()  # Ses dosyasının çalmasını bekleyin
print("Kayit 2 M = 3 Benim Foknsiyonum")
sd.play(condata2, fs2)
sd.wait()  # Ses dosyasının çalmasını bekleyin
print("Kayit 1 M = 4 Benim Foknsiyonum")
sd.play(condata3, fs)
sd.wait()  # Ses dosyasının çalmasını bekleyin
print("Kayit 2 M = 4 Benim Foknsiyonum")
sd.play(condata4, fs2)
sd.wait()  # Ses dosyasının çalmasını bekleyin
print("Kayit 1 M = 5 Benim Foknsiyonum")
sd.play(condata5, fs)
sd.wait()  # Ses dosyasının çalmasını bekleyin
print("Kayit 2 M = 5 Benim Foknsiyonum")
sd.play(condata6, fs2)
sd.wait()  # Ses dosyasının çalmasını bekleyin

print("Kayit 1 M = 3 Hazir Foknsiyonum")
sd.play(con_result1_1, fs)
sd.wait()  # Ses dosyasının çalmasını bekleyin
print("Kayit 2 M = 3 Hazir Foknsiyonum")
sd.play(con_result2_1, fs2)
sd.wait()  # Ses dosyasının çalmasını bekleyin
print("Kayit 1 M = 4 Hazir Foknsiyonum")
sd.play(con_result3_1, fs)
sd.wait()  # Ses dosyasının çalmasını bekleyin
print("Kayit 2 M = 4 Hazir Foknsiyonum")
sd.play(con_result4_1, fs2)
sd.wait()  # Ses dosyasının çalmasını bekleyin
print("Kayit 1 M = 5 Hazir Foknsiyonum")
sd.play(con_result5_1, fs)
sd.wait()  # Ses dosyasının çalmasını bekleyin
print("Kayit 2 M = 5 Hazir Foknsiyonum")
sd.play(con_result6_1, fs2)
sd.wait()  # Ses dosyasının çalmasını bekleyin

plt.figure(figsize=(12, 10))

plt.subplot(2, 2, 1)
plt.stem(condata2)
plt.title('conv2')
plt.xlabel('n')
plt.ylabel('Amplitude')
plt.grid(True)

plt.subplot(2, 2, 2)  # İkinci subplot
plt.stem(con_result2_1)
plt.title('conv2 hazır m 3 ')
plt.xlabel('n')
plt.ylabel('Amplitude')
plt.grid(True)

plt.subplot(2, 2, 3)  # Üçüncü subplot
plt.stem(condata4)
plt.title('conv4')
plt.xlabel('n')
plt.ylabel('Amplitude')
plt.grid(True)

plt.subplot(2, 2, 4)  # Dördüncü subplot
plt.stem(condata6)
plt.title('conv6')
plt.xlabel('n')
plt.ylabel('Amplitude')
plt.grid(True)

plt.tight_layout()
plt.show()

plt.figure(figsize=(12, 10))

plt.subplot(2, 2, 1)
plt.stem(condata1)
plt.title('conv2')
plt.xlabel('n')
plt.ylabel('Amplitude')
plt.grid(True)

plt.subplot(2, 2, 2)  # İkinci subplot
plt.stem(condata3)
plt.title('conv2 hazır m 3 ')
plt.xlabel('n')
plt.ylabel('Amplitude')
plt.grid(True)

plt.subplot(2, 2, 3)  # Üçüncü subplot
plt.stem(condata5)
plt.title('conv4')
plt.xlabel('n')
plt.ylabel('Amplitude')
plt.grid(True)

plt.tight_layout()
plt.show()