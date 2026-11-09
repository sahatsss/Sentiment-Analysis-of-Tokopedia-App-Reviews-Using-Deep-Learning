# Laporan Proyek Machine Learning - Samuel Sahat Mardyantoro

## Domain Proyek

Industri e-commerce di Indonesia berkembang pesat dengan munculnya berbagai platform belanja daring. Tokopedia, sebagai salah satu marketplace terbesar, menerima jutaan ulasan dari penggunanya setiap tahun. Ulasan ini mencerminkan pengalaman pengguna, baik positif maupun negatif, terkait kualitas layanan, kecepatan pengiriman, hingga pengalaman menggunakan aplikasi.

Namun, volume ulasan yang sangat besar membuat sulit bagi perusahaan untuk menganalisis satu per satu secara manual. Oleh karena itu, analisis sentimen berbasis machine learning dapat membantu mengklasifikasikan ulasan pelanggan secara otomatis ke dalam kategori sentimen (positif, netral, negatif). Hasil klasifikasi ini penting bagi perusahaan untuk memahami persepsi publik, meningkatkan layanan, dan mengidentifikasi area yang perlu diperbaiki.

## Business Understanding

### Problem Statements

1. Bagaimana cara mengklasifikasikan ulasan pengguna aplikasi Tokopedia menjadi kategori sentimen positif, netral, atau negatif?
2. Bagaimana hasil analisis sentimen ini dapat membantu perusahaan meningkatkan kualitas layanan dan pengalaman pengguna?

### Goals

1. Membangun model machine learning berbasis deep learning untuk melakukan analisis sentimen terhadap ulasan aplikasi Tokopedia.
2. Memberikan wawasan berdasarkan distribusi sentimen untuk mendukung pengambilan keputusan strategis.

### Solution Statements

* Menggunakan teknik *Natural Language Processing (NLP)*, khususnya *TF-IDF vectorization* untuk ekstraksi fitur teks.
* Melatih model deep learning berbasis *Artificial Neural Network (ANN)* untuk klasifikasi multi-kelas (positif, netral, negatif).
* Membandingkan hasil evaluasi model dengan target akurasi minimal 85% pada data testing.

## Data Understanding

Dataset yang digunakan merupakan hasil *web scraping* ulasan aplikasi Tokopedia dari Google Play Store menggunakan library `google-play-scraper`. Data yang terkumpul sebanyak **10.000 ulasan**, dengan tiga atribut utama:

* `userName`: nama pengguna yang memberikan ulasan.
* `score`: rating bintang (1–5) yang diberikan.
* `content`: teks ulasan dari pengguna.

Dari teks ulasan, dilakukan *preprocessing* dan *feature extraction*. Label sentimen dibuat menggunakan model *pre-trained* IndoBERT untuk *sentiment classification* dengan tiga kategori:

* **label\_0** → sentimen positif (4–5 bintang).
* **label\_1** → sentimen negatif (1–2 bintang).
* **label\_2** → sentimen netral (3 bintang).

Distribusi data setelah labeling:

* Positif: 4837 ulasan
* Netral: 4229 ulasan
* Negatif: 934 ulasan

Dataset cukup berimbang untuk kelas positif dan netral, meskipun kelas negatif lebih sedikit.

## Data Preparation

Tahapan yang dilakukan meliputi:

* **Pembersihan teks (cleaning):** menghapus URL, angka, tanda baca, dan teks tidak relevan.
* **Stopword removal dan stemming:** menggunakan library Sastrawi untuk bahasa Indonesia.
* **Ekstraksi fitur:** menggunakan TF-IDF dengan `max_features=5000` dan `ngram_range=(1,2)` agar model menangkap konteks kata dan frasa.
* **Encoding label:** label dikonversi menjadi numerik lalu diubah ke format *one-hot encoding* untuk klasifikasi multi-kelas.
* **Split data:** dataset dibagi menjadi 80% data latih dan 20% data uji.

## Modeling

Model yang digunakan adalah **Artificial Neural Network (ANN)** dengan arsitektur:

* Input layer: ukuran sama dengan jumlah fitur TF-IDF (5000).
* Hidden layer 1: Dense 512 neuron dengan aktivasi ReLU.
* Hidden layer 2: Dense 256 neuron dengan aktivasi ReLU.
* Output layer: Dense 3 neuron dengan aktivasi Softmax.

Model dilatih dengan:

* Loss function: categorical crossentropy.
* Optimizer: Adam.
* Epochs: 15.
* Batch size: 64.

## Evaluation

Hasil pelatihan model:

* **Akurasi training:** \~99%
* **Akurasi testing:** \~85–86%
* **Val loss:** cenderung naik pada epoch > 5, mengindikasikan kemungkinan *overfitting*.

Distribusi prediksi menunjukkan bahwa model mampu mengklasifikasikan ulasan positif dan netral dengan cukup baik, namun masih kesulitan mengenali ulasan negatif karena jumlah data yang relatif sedikit.

Secara keseluruhan, model ANN dengan TF-IDF menunjukkan performa baik dan dapat digunakan sebagai baseline untuk analisis sentimen otomatis. Namun, peningkatan kualitas model dapat dilakukan dengan:

* Menambah data kelas negatif agar lebih seimbang.
* Menerapkan regularisasi (dropout) untuk mengurangi overfitting.
* Menggunakan model *transformer-based* seperti IndoBERT untuk pelatihan langsung.

## Kesimpulan

Dengan membangun model analisis sentimen berbasis deep learning, ulasan pengguna Tokopedia berhasil diklasifikasikan ke dalam kategori positif, netral, dan negatif dengan akurasi testing sekitar 85%. Hasil ini menunjukkan bahwa mayoritas pengguna memberikan ulasan positif, namun ulasan negatif yang lebih sedikit tetap penting untuk ditindaklanjuti agar perusahaan dapat meningkatkan kualitas layanan.

Proyek ini menunjukkan bahwa penerapan NLP dan deep learning dapat membantu perusahaan e-commerce mengolah data dalam skala besar secara otomatis, menghasilkan wawasan yang relevan, serta mendukung pengambilan keputusan berbasis data (*data-driven decision making*).
