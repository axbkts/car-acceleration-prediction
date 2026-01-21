## Car Acceleration Prediction with MLP Regressor
### MLP Regressor ile Araç Hızlanma Tahmini

## Project Description / Proje Açıklaması

**English**

This project aims to predict the 0-100 km/h acceleration time of car models using a Multi-Layer Perceptron (MLP) Artificial Neural Network. The project involves a comprehensive data science pipeline consisting of data cleaning, feature engineering, and hyperparameter optimization using GridSearch. The goal is to establish a regression model that correlates technical specifications (such as horsepower and torque) with vehicle performance.

**Türkçe**

Bu proje, Çok Katmanlı Algılayıcı (MLP) Yapay Sinir Ağı kullanarak araç modellerinin 0-100 km/s hızlanma sürelerini tahmin etmeyi amaçlamaktadır. Proje; veri temizleme, öznitelik mühendisliği ve GridSearch kullanarak hiperparametre optimizasyonu içeren kapsamlı bir veri bilimi hattından oluşmaktadır. Amaç, teknik özellikler (beygir gücü ve tork gibi) ile araç performansı arasında ilişki kuran bir regresyon modeli oluşturmaktır.

---

## Dataset and Features / Veri Seti ve Özellikler

The dataset consists of technical specifications for various car models. Unstructured data containing units and ranges were processed into numerical formats.

Veri seti, çeşitli araç modellerinin teknik özelliklerini içermektedir. Birimler ve aralıklar içeren yapılandırılmamış veriler sayısal formatlara dönüştürülmüştür.

* **Input Features (Girdi Değişkenleri):**
    * Horsepower (HP)
    * Torque (Nm)
    * Top Speed (km/h)
    * Engine Displacement / Battery Capacity
    * Seating Capacity
    * Fuel Type (Label Encoded)
    * Price (USD)
* **Target Variable (Hedef Değişken):**
    * Acceleration 0-100 km/h (Seconds)

---

## Methodology / Yöntem

The project is divided into two main scripts:

Proje iki ana betiğe ayrılmıştır:

### 1. Data Preprocessing (`data_preprocessing.py`)
This script handles the cleaning of the raw dataset.
Bu betik, ham veri setinin temizlenmesini sağlar.

* **Regex Cleaning:** Removes non-numeric characters and units (e.g., "hp", "km/h", "$") from the dataset.
* **Range Handling:** Identifies and removes rows containing ambiguous range values (e.g., "200-300").
* **Encoding:** Converts categorical "Fuel Types" into numerical format using Label Encoding.
* **Missing Values:** Removes rows with null values to ensure data integrity.

* **Regex Temizliği:** Veri setinden sayısal olmayan karakterleri ve birimleri (örneğin "hp", "km/h", "$") temizler.
* **Aralık Yönetimi:** Belirsiz aralık değerleri (örneğin "200-300") içeren satırları tespit eder ve çıkarır.
* **Kodlama:** Kategorik "Fuel Types" verisini Label Encoding kullanarak sayısal formata dönüştürür.
* **Eksik Veriler:** Veri bütünlüğünü sağlamak için boş değer içeren satırları çıkarır.

### 2. Model Training (`model_training.py`)
This script builds, trains, and evaluates the Neural Network.
Bu betik, Sinir Ağını kurar, eğitir ve değerlendirir.

* **Scaling:** Applies `StandardScaler` to normalize features, which is critical for MLP convergence.
* **Architecture:** Uses Scikit-Learn's `MLPRegressor`.
* **Optimization:** Implements `GridSearchCV` to test combinations of:
    * Hidden Layer Sizes
    * Activation Functions (ReLU, Tanh)
    * Solvers (Adam)
    * Learning Rates
* **Evaluation:** Calculates R2 Score, RMSE (Root Mean Squared Error), and MAPE (Mean Absolute Percentage Error).

* **Ölçekleme:** MLP yakınsamasını sağlamak için özelliklere `StandardScaler` uygular.
* **Mimari:** Scikit-Learn kütüphanesinin `MLPRegressor` modelini kullanır.
* **Optimizasyon:** Aşağıdaki hiperparametre kombinasyonlarını test etmek için `GridSearchCV` uygular:
    * Gizli Katman Boyutları
    * Aktivasyon Fonksiyonları (ReLU, Tanh)
    * Çözücüler (Adam)
    * Öğrenme Oranları
* **Değerlendirme:** R2 Skoru, RMSE (Kök Ortalama Kare Hata) ve MAPE (Ortalama Mutlak Yüzde Hata) hesaplar.

---

## Installation / Kurulum

To run this project locally, follow these steps:
Bu projeyi yerel ortamda çalıştırmak için aşağıdaki adımları izleyin:

1.  **Clone the repository / Repoyu klonlayın:**
    ```bash
    git clone [https://github.com/axbkts/car-acceleration-prediction.git](https://github.com/axbkts/car-acceleration-prediction.git)
    cd car-acceleration-prediction
    ```

2.  **Install dependencies / Bağımlılıkları yükleyin:**
    ```bash
    pip install pandas numpy matplotlib seaborn scikit-learn
    ```

---

## Usage / Kullanım

1.  **Run Preprocessing / Önişlemeyi Çalıştırın:**
    Generates the clean dataset (`cars_data_preprocessed_no_missing.csv`).
    Temiz veri setini (`cars_data_preprocessed_no_missing.csv`) oluşturur.
    ```bash
    python data_preprocessing.py
    ```

2.  **Run Training / Eğitimi Çalıştırın:**
    Trains the model, displays performance graphs, and saves metrics to CSV files.
    Modeli eğitir, performans grafiklerini görüntüler ve metrikleri CSV dosyalarına kaydeder.
    ```bash
    python model_training.py
    ```

---

## Results / Sonuçlar

Upon running the model training script, the following outputs are generated:
Model eğitim betiği çalıştırıldığında aşağıdaki çıktılar üretilir:

* **Performance Table:** A CSV file listing R2 and RMSE scores for all tested hyperparameter combinations.
* **Test Metrics:** A CSV file detailing the performance on the unseen test set.
* **Visualizations:**
    * Actual vs. Predicted Scatter Plot
    * Error Distribution Histogram
    * Feature Correlation Chart
    * Training Loss Curve

* **Performans Tablosu:** Test edilen tüm hiperparametre kombinasyonları için R2 ve RMSE skorlarını listeleyen bir CSV dosyası.
* **Test Metrikleri:** Görülmemiş test seti üzerindeki performansı detaylandıran bir CSV dosyası.
* **Görselleştirmeler:**
    * Gerçek ve Tahmin Edilen Dağılım Grafiği
    * Hata Dağılımı Histogramı
    * Özellik Korelasyon Grafiği
    * Eğitim Kayıp Eğrisi

---

## License / Lisans

This project is licensed under the MIT License.
Bu proje MIT Lisansı ile lisanslanmıştır.
