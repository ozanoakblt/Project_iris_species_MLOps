# 🌸 Iris Species Classification MLOps Project

Bu proje, **Naive Bayes** algoritması kullanılarak Iris çiçek türlerini sınıflandıran bir **MLOps uygulamasıdır**.  
Model, **FastAPI** ile servis edilip **Bootstrap tabanlı bir web arayüzü** üzerinden tahmin yapılabilir.  
Proje ayrıca **Docker** ile containerize edilebilir ve GitHub/LinkedIn üzerinde paylaşılmaya uygundur. 🚀

---


---

## 📊 Kullanılan Veri Seti

- **Veri Seti Adı:** Iris Species Dataset  
- **Kaynak:** [Kaggle - Iris Species](https://www.kaggle.com/datasets/uciml/iris)  
- **Açıklama:** 3 farklı iris türünü (Setosa, Versicolor, Virginica) sepal ve petal ölçülerine göre sınıflandırmak.

---

## ⚙ Kullanılan Teknolojiler

- **Python 3.x**
- **Scikit-learn** (Naive Bayes, StandardScaler, LabelEncoder)
- **Pandas / Numpy**
- **FastAPI**
- **Jinja2 Templates**
- **Bootstrap 5**
- **HTML / CSS / JavaScript**
- **Pickle** (model saklama)
- **Docker** (isteğe bağlı)

---

## 🧠 Model Eğitimi

1. **Veri Ön İşleme**
   - `LabelEncoder` ile hedef değişken encode edildi.
   - `StandardScaler` ile veriler normalize edildi.
2. **Model**
   - `GaussianNB()` kullanıldı.
3. **Değerlendirme**
   - Accuracy, Confusion Matrix ve Classification Report hesaplandı.
4. **Model Kaydetme**
   - `pickle` ile model, scaler ve label encoder `.pkl` dosyasına kaydedildi.

```python
with open("iris_gnb_model.pkl", "wb") as f:
    pickle.dump({
        "model": gnb,
        "scaler": scaler,
        "label_encoder": label_encoder
    }, f)
{
    "SepalLengthCm": 5.1,
    "SepalWidthCm": 3.5,
    "PetalLengthCm": 1.4,
    "PetalWidthCm": 0.2
}
{
    "predicted_species": "Iris-setosa"
}
📬 İletişim

Geliştirici: Ozan Akbulut

LinkedIn: linkedin.com/in/ozan-akbulutt

E-posta: ozan.akbltt@gmail.com





