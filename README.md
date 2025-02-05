# 📈 Stock Price Prediction with PyTorch

Bu proje, hisse senedi fiyatlarını tahmin etmek için **PyTorch** kullanarak derin öğrenme modelleri geliştirmektedir. 
Proje, **yfinance** kütüphanesi kullanarak hisse senedi verilerini çeker ve **RNN / LSTM modelleri** ile tahminleme yapar. 

---

## 📌 Özellikler
✔️ Yahoo Finance üzerinden otomatik veri çekme  
✔️ Veri görselleştirme (Matplotlib & Seaborn)  
✔️ PyTorch ile RNN/LSTM modeli eğitimi  
✔️ Eğitim ve test performans analizi  

---

## 🚀 Kurulum

Aşağıdaki komutları çalıştırarak gerekli tüm kütüphaneleri yükleyebilirsiniz:

```bash
pip install torch numpy pandas tqdm yfinance seaborn matplotlib plotly scikit-learn
```

---

## 📊 Kullanılan Teknolojiler

- **PyTorch** - Derin öğrenme modeli eğitimi  
- **pandas & NumPy** - Veri işleme  
- **yfinance** - Hisse senedi verisi çekme  
- **Matplotlib & Seaborn** - Grafik çizimi  
- **Plotly** - Etkileşimli veri görselleştirme  
- **Scikit-learn** - Model değerlendirme  

---

## 🔧 Kullanım

### 1️⃣ **Veri Setinin Çekilmesi**
Aşağıdaki Python kodu ile **yfinance** kütüphanesi kullanılarak belirlenen hisse senedine ait geçmiş veriler çekilir:

```python
import yfinance as yf

ticker = "AAPL"  # Apple hisse senedi (Örnek)
df = yf.download(ticker, start="2020-01-01", end="2024-01-01")
print(df.head())  # İlk 5 satırı göster
```

### 2️⃣ **Veri Ön İşleme**
Veriler, eksik değerlerden temizlenir ve modelin kullanabileceği formatta ölçeklendirilir.

```python
from sklearn.preprocessing import MinMaxScaler

scaler = MinMaxScaler(feature_range=(0, 1))
df_scaled = scaler.fit_transform(df["Close"].values.reshape(-1,1))
```

### 3️⃣ **PyTorch ile Model Eğitimi**
📌 Aşağıdaki kod, **PyTorch LSTM modelini** eğitir:

```python
import torch
import torch.nn as nn
import torch.optim as optim

class LSTMModel(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(LSTMModel, self).__init__()
        self.lstm = nn.LSTM(input_size, hidden_size)
        self.fc = nn.Linear(hidden_size, output_size)
    
    def forward(self, x):
        lstm_out, _ = self.lstm(x)
        output = self.fc(lstm_out[-1])
        return output

# Model oluşturma
model = LSTMModel(input_size=1, hidden_size=50, output_size=1)
optimizer = optim.Adam(model.parameters(), lr=0.001)
criterion = nn.MSELoss()
```

---

## 📉 Model Performansı ve Sonuçlar
Eğitim tamamlandıktan sonra modelin performansı aşağıdaki gibi görselleştirilir:

```python
import matplotlib.pyplot as plt

plt.plot(y_test, label="Gerçek Fiyat")
plt.plot(y_pred, label="Tahmin Edilen Fiyat")
plt.legend()
plt.show()
```

---

## 📂 Proje Dosya Yapısı

```
📂 StockPricePredict
 ├── 📜 StockPricePredict.ipynb    # Model eğitim dosyası (Jupyter Notebook)
 ├── 📜 README.md                  # Proje dökümantasyonu
 ├── 📜 requirements.txt            # Gerekli kütüphaneler
 └── 📜 data                        # Hisse senedi verileri (Opsiyonel)
```

---

## 🔗 Kaynaklar
- 📌 [Yahoo Finance API](https://www.yfinance.org/)
- 📌 [PyTorch Documentation](https://pytorch.org/docs/stable/index.html)
- 📌 [Scikit-learn Documentation](https://scikit-learn.org/stable/)

---

## 🤝 Katkıda Bulunma
Proje ile ilgili öneri veya katkılarınız varsa **Pull Request (PR)** gönderebilirsiniz. Geliştirici topluluğuna katkı sağlamak için aşağıdaki adımları takip edebilirsiniz:

1. 🍴 **Projeyi Fork'layın**
2. 🛠️ **Geliştirme Yapın**
3. 🔄 **Kodunuzu Güncelleyin**
4. ✅ **Pull Request Gönderin**

---

## 📝 Lisans
📌 **MIT Lisansı** - Bu projeyi özgürce kullanabilir, değiştirebilir ve geliştirebilirsiniz.

---
```
