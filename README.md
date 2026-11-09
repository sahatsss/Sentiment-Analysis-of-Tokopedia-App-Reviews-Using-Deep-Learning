# Sentiment Analysis of Tokopedia App Reviews

Klasifikasi sentimen ulasan aplikasi Tokopedia di Google Play Store (positif, netral, negatif) menggunakan TF-IDF dan artificial neural network.

## Ringkasan

1. Mengumpulkan 10.000 ulasan dengan `google-play-scraper`.
2. Memberi label sentimen dengan model IndoBERT pre-trained.
3. Membersihkan teks, menghapus stopword, dan melakukan stemming dengan Sastrawi.
4. Mengekstraksi fitur dengan TF-IDF (5.000 fitur, unigram dan bigram).
5. Melatih ANN (Dense 512 → Dense 256 → Softmax 3) dengan split 80/20.

Distribusi label: 4.837 positif, 4.229 netral, dan 934 negatif.

## Hasil

| Metrik | Nilai |
|---|---|
| Training accuracy | ~0.99 |
| Test accuracy | ~0.85 |

Jarak antara akurasi training dan test, ditambah validation loss yang naik setelah epoch ke-5, menunjukkan overfitting. Kelas negatif paling sulit dikenali karena jumlah datanya paling sedikit.

## Struktur

```
.
├── 01_scraping.ipynb
├── 02_model_training.ipynb
├── tokopedia_reviews.csv
├── requirements.txt
└── docs/
    └── laporan.md
```

## Menjalankan

```bash
pip install -r requirements.txt
jupyter notebook
```

`01_scraping.ipynb` hanya perlu dijalankan jika ingin mengambil data baru. Dataset sudah tersedia di `tokopedia_reviews.csv`.

## Pengembangan Lanjutan

- Menyeimbangkan kelas negatif (resampling atau penambahan data)
- Menambahkan dropout atau early stopping
- Fine-tuning IndoBERT secara langsung sebagai classifier

## Dokumentasi

Laporan lengkap ada di [docs/laporan.md](docs/laporan.md).
