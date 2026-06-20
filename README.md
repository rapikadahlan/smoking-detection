# smoking-detection

# 🚭 Smoking Detection System

Sistem deteksi aktivitas merokok secara real-time menggunakan YOLOv8 dan MediaPipe. Sistem ini dapat mendeteksi keberadaan rokok dan posisi tangan secara bersamaan untuk mengidentifikasi aktivitas merokok, kemudian menyimpan bukti pelanggaran dan menampilkan data melalui dashboard berbasis web.

## 📌 Features

- Real-time smoking detection menggunakan webcam
- Deteksi rokok menggunakan YOLOv8
- Deteksi tangan menggunakan MediaPipe
- Alarm otomatis saat aktivitas merokok terdeteksi
- Penyimpanan bukti pelanggaran (evidence image)
- Dashboard monitoring berbasis Flask
- Riwayat pelanggaran tersimpan pada database SQLite

---

## 🛠️ Technologies Used

- Python
- Flask
- YOLOv8 (Ultralytics)
- MediaPipe
- OpenCV
- SQLite
- Pygame

---

## 📂 Project Structure

```bash
Smoking-Detection/
│
├── app.py
├── dashboard.py
├── best.pt
├── smoking_violations.db
│
├── static/
│   ├── evidence/
│   └── alarm.wav
│
├── templates/
│   ├── dashboard.html
│   └── index.html
│
├── requirements.txt
└── README.md
```

---

## ⚙️ Installation

### 1. Clone Repository

```bash
git clone https://github.com/username/Smoking-Detection.git
cd Smoking-Detection
```

### 2. Create Virtual Environment

```bash
python -m venv venv
```

### Windows

```bash
venv\Scripts\activate
```

### Linux / MacOS

```bash
source venv/bin/activate
```

### 3. Install Dependencies

```bash
pip install -r requirements.txt
```

---

## ▶️ Run Application

### Start Detection System

```bash
python app.py
```

### Open Dashboard

```bash
python dashboard.py
```

Dashboard dapat diakses melalui browser:

```text
http://127.0.0.1:5000
```

---

## 🧠 Detection Workflow

1. Kamera menangkap video secara real-time.
2. YOLOv8 mendeteksi objek rokok.
3. MediaPipe mendeteksi posisi tangan.
4. Sistem memeriksa kedekatan rokok dengan tangan.
5. Jika aktivitas merokok terdeteksi:
   - Alarm berbunyi.
   - Screenshot disimpan.
   - Data pelanggaran masuk ke database.
6. Dashboard menampilkan riwayat pelanggaran.

---

## 📊 Database

Sistem menggunakan SQLite untuk menyimpan:

- ID Pelanggaran
- Timestamp
- Gambar Bukti
- Status Deteksi

---

## 📸 Evidence Storage

Semua bukti pelanggaran akan disimpan pada:

```bash
static/evidence/
```

---

## 📋 Requirements

```txt
Flask
opencv-python
mediapipe
ultralytics
pygame
numpy
```

---

## 🚀 Future Improvements

- Multi-camera support
- Email notification
- Telegram notification
- Face recognition integration
- Cloud database integration
- Web deployment
Information Systems Student  
Universitas Islam Negeri Sultan Syarif Kasim Riau
