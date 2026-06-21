import os
import cv2
import time
import sqlite3
import threading
from datetime import datetime, date, timedelta

from ultralytics import YOLO
import pygame

from flask import Flask, Response, jsonify, render_template, request, redirect

# =========================
# CONFIG
# =========================
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DB_FILE = os.path.join(BASE_DIR, "smoking_violations.db")
AUDIO_FILE = os.path.join(BASE_DIR, "pemberitahuan.mp3")

EVIDENCE_DIR = os.path.join(BASE_DIR, "static", "evidence")
os.makedirs(EVIDENCE_DIR, exist_ok=True)

CAMERA_INDEX = 0
WINDOW_NAME = "Smoking Detection System"

YOLO_WEIGHTS = "best.pt"

SMOKING_STREAK_FRAMES = 6
ALERT_COOLDOWN_SEC = 8

# =========================
# FLASK
# =========================
app = Flask(__name__, template_folder="templates", static_folder="static")

# =========================
# GLOBAL STATE
# =========================
frame_lock = threading.Lock()
latest_frame_jpeg = None
latest_status = {
    "smoking": False,
    "person_count": 0,
    "today_count": 0,
    "confidence": 0.0,
    "last_alert": None,
    "running": False,
}

stop_event = threading.Event()

# =========================
# AUDIO
# =========================
audio_lock = threading.Lock()
audio_playing = False

try:
    pygame.mixer.init()
    AUDIO_READY = True
except Exception as e:
    AUDIO_READY = False
    print(f"[WARN] Audio device tidak siap: {e}")


def play_warning_audio():
    global audio_playing

    if not AUDIO_READY:
        print("[WARN] Audio system tidak siap.")
        return

    if not os.path.exists(AUDIO_FILE):
        print(f"[WARN] File audio tidak ditemukan: {AUDIO_FILE}")
        return

    if audio_playing:
        return

    def worker():
        global audio_playing
        with audio_lock:
            audio_playing = True
            try:
                pygame.mixer.music.stop()
                pygame.mixer.music.load(AUDIO_FILE)
                pygame.mixer.music.play()
                while pygame.mixer.music.get_busy():
                    time.sleep(0.05)
            except Exception as e:
                print(f"[WARN] Gagal memutar audio: {e}")
            finally:
                audio_playing = False

    threading.Thread(target=worker, daemon=True).start()

def save_evidence(frame):
    try:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"smoking_{timestamp}.jpg"
        filepath = os.path.join(EVIDENCE_DIR, filename)

        cv2.imwrite(filepath, frame)

        print(f"[INFO] Evidence tersimpan: {filepath}")
        return filepath
    except Exception as e:
        print(f"[ERROR] Gagal simpan evidence: {e}")
        return None

# =========================
# DATABASE
# =========================
def get_conn():
    conn = sqlite3.connect(DB_FILE, check_same_thread=False)
    conn.row_factory = sqlite3.Row
    return conn


def init_db():
    with get_conn() as conn:
        conn.execute("""
            CREATE TABLE IF NOT EXISTS violations (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                created_at TEXT NOT NULL,
                violation_date TEXT NOT NULL,
                violation_time TEXT NOT NULL,
                confidence REAL NOT NULL DEFAULT 0.0,
                note TEXT NOT NULL DEFAULT 'Merokok terdeteksi',
                source TEXT NOT NULL DEFAULT 'YOLO + Mediapipe'
            )
        """)
        conn.commit()


def ensure_db_schema():
    with get_conn() as conn:
        cur = conn.cursor()
        cur.execute("PRAGMA table_info(violations)")
        cols = {row["name"] for row in cur.fetchall()}

        if "location" not in cols:
            conn.execute("ALTER TABLE violations ADD COLUMN location TEXT DEFAULT 'Area Kamera 1'")
        if "reporter" not in cols:
            conn.execute("ALTER TABLE violations ADD COLUMN reporter TEXT DEFAULT 'AI System'")
        if "status" not in cols:
            conn.execute("ALTER TABLE violations ADD COLUMN status TEXT DEFAULT 'Baru'")
        if "evidence" not in cols:
            conn.execute("ALTER TABLE violations ADD COLUMN evidence TEXT")

        conn.commit()


def create_violation(confidence, note="Merokok terdeteksi", source="YOLO + Mediapipe"):
    now = datetime.now()
    with get_conn() as conn:
        conn.execute("""
            INSERT INTO violations (
                created_at, violation_date, violation_time,
                confidence, note, source,
                location, reporter, status
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
        """, (
            now.strftime("%Y-%m-%d %H:%M:%S"),
            now.strftime("%Y-%m-%d"),
            now.strftime("%H:%M:%S"),
            float(confidence),
            note,
            source,
            "Area Kamera 1",
            "AI System",
            "Baru"
        ))
        conn.commit()

def update_followup_violation(violation_id, location, operator_name, status, note, evidence=None):
    with get_conn() as conn:
        row = conn.execute(
            "SELECT evidence FROM violations WHERE id = ?",
            (violation_id,)
        ).fetchone()

        if not row:
            return False

        final_evidence = evidence if evidence else row["evidence"]

        conn.execute("""
            UPDATE violations
            SET
                location = ?,
                reporter = ?,
                status = ?,
                note = ?,
                evidence = ?
            WHERE id = ?
        """, (
            location,
            operator_name,
            status,
            note,
            final_evidence,
            violation_id
        ))
        conn.commit()

    return True


def fetch_violations(limit=50, location=None, date_filter=None):
    with get_conn() as conn:

        query = "SELECT * FROM violations WHERE 1=1"
        params = []

        if location:
            query += " AND location=?"
            params.append(location)

        if date_filter:
            query += " AND violation_date=?"
            params.append(date_filter)

        query += " ORDER BY id DESC LIMIT ?"
        params.append(limit)

        rows = conn.execute(query, params).fetchall()

    return [dict(r) for r in rows]


# =========================
# TAMBAHAN: REKAP PER HARI/JAM/Lokasi
# =========================

WEEKDAY_LABELS = ["Senin", "Selasa", "Rabu", "Kamis", "Jumat", "Sabtu", "Minggu"]


def count_total(location=None):
    with get_conn() as conn:
        if location:
            row = conn.execute(
                "SELECT COUNT(*) FROM violations WHERE location = ?",
                (location,)
            ).fetchone()
        else:
            row = conn.execute(
                "SELECT COUNT(*) FROM violations"
            ).fetchone()
    return row[0] or 0


def count_today(location=None):
    today = date.today().isoformat()
    with get_conn() as conn:
        if location:
            row = conn.execute(
                "SELECT COUNT(*) FROM violations WHERE violation_date = ? AND location = ?",
                (today, location)
            ).fetchone()
        else:
            row = conn.execute(
                "SELECT COUNT(*) FROM violations WHERE violation_date = ?",
                (today,)
            ).fetchone()
    return row[0] or 0


def count_week(location=None):
    start = (date.today() - timedelta(days=6)).isoformat()
    with get_conn() as conn:
        if location:
            row = conn.execute(
                "SELECT COUNT(*) FROM violations WHERE violation_date >= ? AND location = ?",
                (start, location)
            ).fetchone()
        else:
            row = conn.execute(
                "SELECT COUNT(*) FROM violations WHERE violation_date >= ?",
                (start,)
            ).fetchone()
    return row[0] or 0


def get_avg_per_day(location=None):
    with get_conn() as conn:
        if location:
            row = conn.execute("""
                SELECT COUNT(*) AS total, COUNT(DISTINCT violation_date) AS days
                FROM violations
                WHERE location = ?
            """, (location,)).fetchone()
        else:
            row = conn.execute("""
                SELECT COUNT(*) AS total, COUNT(DISTINCT violation_date) AS days
                FROM violations
            """).fetchone()

    total = row["total"] or 0
    days = row["days"] or 0
    return round(total / days, 2) if days else 0.0


def get_peak_day(location=None):
    with get_conn() as conn:
        if location:
            row = conn.execute("""
                SELECT violation_date, COUNT(*) AS cnt
                FROM violations
                WHERE location = ?
                GROUP BY violation_date
                ORDER BY cnt DESC, violation_date DESC
                LIMIT 1
            """, (location,)).fetchone()
        else:
            row = conn.execute("""
                SELECT violation_date, COUNT(*) AS cnt
                FROM violations
                GROUP BY violation_date
                ORDER BY cnt DESC, violation_date DESC
                LIMIT 1
            """).fetchone()

    if not row:
        return None, 0
    return row["violation_date"], row["cnt"]


def get_peak_hour(location=None):
    with get_conn() as conn:
        if location:
            row = conn.execute("""
                SELECT substr(violation_time, 1, 2) AS hour, COUNT(*) AS cnt
                FROM violations
                WHERE location = ?
                GROUP BY substr(violation_time, 1, 2)
                ORDER BY cnt DESC, hour DESC
                LIMIT 1
            """, (location,)).fetchone()
        else:
            row = conn.execute("""
                SELECT substr(violation_time, 1, 2) AS hour, COUNT(*) AS cnt
                FROM violations
                GROUP BY substr(violation_time, 1, 2)
                ORDER BY cnt DESC, hour DESC
                LIMIT 1
            """).fetchone()

    if not row:
        return None, 0
    return f"{row['hour']}:00", row["cnt"]


def get_daily_series(location=None):
    """
    Grafik harian jadi Senin–Minggu (bukan 14 hari).
    Ini agregasi semua data, lalu dipisah per weekday.
    """
    values = [0] * 7

    with get_conn() as conn:
        if location:
            rows = conn.execute("""
                SELECT CAST(strftime('%w', violation_date) AS INTEGER) AS wd, COUNT(*) AS cnt
                FROM violations
                WHERE location = ?
                GROUP BY wd
            """, (location,)).fetchall()
        else:
            rows = conn.execute("""
                SELECT CAST(strftime('%w', violation_date) AS INTEGER) AS wd, COUNT(*) AS cnt
                FROM violations
                GROUP BY wd
            """).fetchall()

    for r in rows:
        sqlite_wd = int(r["wd"])  # 0 = Minggu, 1 = Senin, ..., 6 = Sabtu
        idx = 6 if sqlite_wd == 0 else sqlite_wd - 1
        values[idx] = r["cnt"]

    return WEEKDAY_LABELS, values


def get_hourly_series(location=None):
    """
    Grafik per jam 00:00–23:00.
    Agregasi semua data, lalu dipisah per jam.
    """
    labels = [f"{h:02d}:00" for h in range(24)]
    values = [0] * 24

    with get_conn() as conn:
        if location:
            rows = conn.execute("""
                SELECT substr(violation_time, 1, 2) AS hour, COUNT(*) AS cnt
                FROM violations
                WHERE location = ?
                GROUP BY substr(violation_time, 1, 2)
            """, (location,)).fetchall()
        else:
            rows = conn.execute("""
                SELECT substr(violation_time, 1, 2) AS hour, COUNT(*) AS cnt
                FROM violations
                GROUP BY substr(violation_time, 1, 2)
            """).fetchall()

    for r in rows:
        hour = int(r["hour"])
        values[hour] = r["cnt"]

    return labels, values


def get_source_breakdown(location=None):
    with get_conn() as conn:
        if location:
            rows = conn.execute("""
                SELECT source, COUNT(*) AS cnt
                FROM violations
                WHERE location = ?
                GROUP BY source
                ORDER BY cnt DESC
            """, (location,)).fetchall()
        else:
            rows = conn.execute("""
                SELECT source, COUNT(*) AS cnt
                FROM violations
                GROUP BY source
                ORDER BY cnt DESC
            """).fetchall()

    if not rows:
        return ["Belum ada data"], [1]

    return [r["source"] for r in rows], [r["cnt"] for r in rows]


def build_report(location=None):
    scope = location or "Semua lokasi"
    total_all = count_total(location)
    today = count_today(location)
    week = count_week(location)
    avg_day = get_avg_per_day(location)
    peak_day, peak_day_cnt = get_peak_day(location)
    peak_hour, peak_hour_cnt = get_peak_hour(location)

    latest_rows = fetch_violations(limit=1, location=location)
    latest = latest_rows[0] if latest_rows else None

    latest_conf = latest["confidence"] if latest else 0.0
    latest_note = latest["note"] if latest else "-"
    latest_time = latest["created_at"] if latest else "-"
    latest_location = latest["location"] if latest else "-"
    latest_status = latest["status"] if latest else "-"

    if total_all == 0:
        title = f"Laporan Operasional — {scope}"
        body = "Belum ada data yang tersimpan untuk lokasi ini. Sistem tetap siap menerima deteksi realtime."
        recommendation = "Jalankan monitoring atau masukkan laporan manual agar analisis mulai terbentuk."
    else:
        title = f"Laporan Operasional — {scope}"
        body = (
            f"Pada scope {scope}, total pelanggaran tersimpan {total_all}. "
            f"Hari ini tercatat {today} pelanggaran, total 7 hari terakhir {week}, "
            f"dan rata-rata pelanggaran per hari {avg_day}. "
            f"Data terakhir tercatat di {latest_location} dengan status {latest_status}."
        )
        recommendation = (
            f"Puncak pelanggaran terjadi pada {peak_day} ({peak_day_cnt} kali) "
            f"dan jam paling ramai di {peak_hour} ({peak_hour_cnt} kali). "
            f"Confidence terakhir {latest_conf:.2f}, catatan terakhir: {latest_note}."
        )

    return {
        "title": title,
        "body": body,
        "recommendation": recommendation,
        "latest_conf": latest_conf,
        "latest_note": latest_note,
        "latest_time": latest_time,
        "latest_location": latest_location,
        "latest_status": latest_status,
        "peak_day": peak_day,
        "peak_day_cnt": peak_day_cnt,
        "peak_hour": peak_hour,
        "peak_hour_cnt": peak_hour_cnt,
        "scope": scope,
    }


# =========================
# CAMERA WORKER
# =========================
def camera_worker():
    global latest_frame_jpeg, latest_status

    model = YOLO(YOLO_WEIGHTS)

    cap = cv2.VideoCapture(CAMERA_INDEX)

    if not cap.isOpened():
        print("[ERROR] Kamera tidak bisa dibuka.")
        latest_status["running"] = False
        return

    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

    latest_status["running"] = True

    streak = 0
    last_alert_time = 0.0

    while not stop_event.is_set():

        ok, frame = cap.read()

        if not ok:
            time.sleep(0.03)
            continue

        frame = cv2.flip(frame, 1)

        smoking = False
        confidence = 0.0
        detection_count = 0

        results = model.predict(
            frame,
            conf=0.50,
            imgsz=640,
            verbose=False
        )

        for result in results:

            if result.boxes is None:
                continue

            for box in result.boxes:

                cls = int(box.cls[0])

                if cls == 0:

                    smoking = True
                    detection_count += 1

                    conf = float(box.conf[0])

                    if conf > confidence:
                        confidence = conf

                    x1, y1, x2, y2 = map(int, box.xyxy[0])

                    cv2.rectangle(
                        frame,
                        (x1, y1),
                        (x2, y2),
                        (0, 0, 255),
                        2
                    )

                    cv2.putText(
                        frame,
                        f"Smoking {conf:.2f}",
                        (x1, y1 - 10),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.8,
                        (0, 0, 255),
                        2
                    )

        if smoking:
            streak += 1
        else:
            streak = 0

        now_ts = time.time()

        if (
            streak >= SMOKING_STREAK_FRAMES
            and (now_ts - last_alert_time) >= ALERT_COOLDOWN_SEC
        ):

            evidence_path = save_evidence(frame)

            create_violation(
                confidence=round(confidence, 2),
                note=f"Merokok terdeteksi | bukti: {evidence_path}",
                source="YOLOv8 Smoking Detection"
            )

            play_warning_audio()

            latest_status["last_alert"] = datetime.now().strftime(
                "%Y-%m-%d %H:%M:%S"
            )

            last_alert_time = now_ts
            streak = 0

            print("[ALERT] Smoking detected!")

        total_today = count_today()

        h, w = frame.shape[:2]

        cv2.rectangle(
            frame,
            (18, 18),
            (w - 18, 170),
            (0, 0, 0),
            -1
        )

        cv2.putText(
            frame,
            "SMOKING DETECTION SYSTEM",
            (30, 52),
            cv2.FONT_HERSHEY_SIMPLEX,
            1.0,
            (255, 255, 255),
            2,
            cv2.LINE_AA
        )

        status_text = (
            "DILARANG MEROKOK!"
            if smoking
            else "MONITORING AREA"
        )

        status_color = (
            (0, 0, 255)
            if smoking
            else (0, 255, 0)
        )

        cv2.putText(
            frame,
            status_text,
            (30, 92),
            cv2.FONT_HERSHEY_SIMPLEX,
            1.2,
            status_color,
            3,
            cv2.LINE_AA
        )

        cv2.putText(
            frame,
            f"Pelanggaran hari ini: {total_today}",
            (30, 130),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.9,
            (255, 255, 255),
            2,
            cv2.LINE_AA
        )

        cv2.putText(
            frame,
            f"Smoking Objects: {detection_count}",
            (30, 160),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            (255, 255, 0),
            2,
            cv2.LINE_AA
        )

        latest_status.update({
            "smoking": smoking,
            "person_count": detection_count,
            "today_count": total_today,
            "confidence": round(confidence, 2),
            "running": True,
        })

        ret2, buf = cv2.imencode(".jpg", frame)

        if ret2:
            with frame_lock:
                latest_frame_jpeg = buf.tobytes()

        time.sleep(0.01)

    cap.release()
    latest_status["running"] = False

# =========================
# DASHBOARD REPORT
# =========================
def build_report():
    total_all = len(fetch_violations(limit=999999))
    today = count_today()
    week = count_week()
    avg_day = get_avg_per_day()
    peak_day, peak_day_cnt = get_peak_day()
    peak_hour, peak_hour_cnt = get_peak_hour()
    latest_rows = fetch_violations(limit=1)
    last_alert = latest_rows[0]["created_at"] if latest_rows else None
    latest_conf = latest_rows[0]["confidence"] if latest_rows else 0.0
    latest_note = latest_rows[0]["note"] if latest_rows else "-"

    if total_all == 0:
        title = "Belum ada pelanggaran tersimpan"
        body = "Dashboard siap menerima data dari deteksi realtime."
        recommendation = "Jalankan monitoring untuk mengisi data dan grafik."
    else:
        title = "Ringkasan monitoring aktif"
        body = f"Hari ini terdeteksi {today} pelanggaran, total mingguan {week}, rata-rata {avg_day} per hari."
        recommendation = (
            f"Puncak pelanggaran terjadi pada {peak_day} ({peak_day_cnt} kali) "
            f"dan jam paling ramai di {peak_hour} ({peak_hour_cnt} kali)."
            if peak_day and peak_hour else
            "Data belum cukup untuk analisis puncak."
        )

    return {
        "title": title,
        "body": body,
        "recommendation": recommendation,
        "last_alert": last_alert,
        "latest_conf": latest_conf,
        "latest_note": latest_note,
    }

# =========================
# API
# =========================
@app.route("/")
def index():
    location = request.args.get("location")
    date_filter = request.args.get("date")

    data = fetch_violations(location=location, date_filter=date_filter)
    total = count_today(location)

    return render_template(
        "dashboard.html",
        data=data,
        count=total,
        selected_location=location,
        selected_date=date_filter
    )

@app.route("/camera")
def camera_page():
    return render_template("camera.html")

@app.route("/video")
def video():
    def generate():
        while True:
            with frame_lock:
                if latest_frame_jpeg is None:
                    time.sleep(0.05)
                    continue
                frame = latest_frame_jpeg

            yield (b"--frame\r\n"
                   b"Content-Type: image/jpeg\r\n\r\n" + frame + b"\r\n")
            time.sleep(0.03)

    return Response(generate(), mimetype="multipart/x-mixed-replace; boundary=frame")


@app.route("/api/dashboard")
def api_dashboard():
    location = request.args.get("location") or None
    date_filter = request.args.get("date") or None

    recent = fetch_violations(limit=12, location=location, date_filter=date_filter)
    daily_labels, daily_values = get_daily_series(location=location)
    hourly_labels, hourly_values = get_hourly_series(location=location)
    source_labels, source_values = get_source_breakdown(location=location)
    report = build_report()

    total_all = count_total(location)
    today_count = count_today(location)
    week_count = count_week(location)
    avg_day = get_avg_per_day(location)
    peak_day, peak_day_cnt = get_peak_day(location)
    peak_hour, peak_hour_cnt = get_peak_hour(location)

    data = {
        "summary": {
            "scope": location or "Semua lokasi",
            "total_all": total_all,
            "today": today_count,
            "week": week_count,
            "avg_day": avg_day,
            "peak_day": peak_day or "-",
            "peak_day_cnt": peak_day_cnt,
            "peak_hour": peak_hour or "-",
            "peak_hour_cnt": peak_hour_cnt,
            "latest_conf": round(report["latest_conf"], 2),
            "latest_note": report["latest_note"],
            "live_status": "Aktif" if latest_status["running"] else "Offline",
            "today_count": today_count,
            "person_count": latest_status["person_count"],
            "smoking": latest_status["smoking"],
            "last_alert": latest_status["last_alert"],
        },
        "charts": {
            "daily": {"labels": daily_labels, "values": daily_values},
            "hourly": {"labels": hourly_labels, "values": hourly_values},
            "source": {"labels": source_labels, "values": source_values},
        },
        "recent": recent,
        "report": report
    }

    return jsonify(data)


@app.route("/export/csv")
def export_csv():
    rows = fetch_violations(limit=999999)
    lines = ["id,created_at,violation_date,violation_time,confidence,note,source"]
    for r in rows:
        lines.append(
            f'{r["id"]},"{r["created_at"]}","{r["violation_date"]}","{r["violation_time"]}",'
            f'"{r["confidence"]}","{r["note"]}","{r["source"]}"'
        )
    csv_data = "\n".join(lines)
    return Response(
        csv_data,
        mimetype="text/csv",
        headers={"Content-Disposition": "attachment; filename=smoking_violations_report.csv"}
    )

@app.route("/followup", methods=["POST"])
def followup():
    violation_id = request.form["violation_id"]
    location = request.form["location"]
    operator_name = request.form["reporter"]  # tetap pakai field lama
    status = request.form["status"]
    note = request.form["note"]

    file = request.files.get("evidence")
    filepath = None

    if file and file.filename:
        filepath = os.path.join("static/evidence", file.filename)
        file.save(filepath)

    ok = update_followup_violation(
        violation_id=violation_id,
        location=location,
        operator_name=operator_name,
        status=status,
        note=note,
        evidence=filepath
    )

    if not ok:
        return "ID pelanggaran tidak ditemukan", 404

    return redirect("/")

@app.route("/add", methods=["POST"])
def add_manual():
    location = request.form["location"]
    reporter = request.form["reporter"]
    status = request.form["status"]
    note = request.form["note"]

    file = request.files.get("evidence")
    filepath = None

    if file and file.filename:
        filepath = os.path.join("static/evidence", file.filename)
        file.save(filepath)

    now = datetime.now()

    with get_conn() as conn:
        conn.execute("""
            INSERT INTO violations (
                created_at, violation_date, violation_time,
                confidence, note, source,
                location, reporter, status, evidence
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """, (
            now.strftime("%Y-%m-%d %H:%M:%S"),
            now.strftime("%Y-%m-%d"),
            now.strftime("%H:%M:%S"),
            0.0,
            note,
            "Manual",
            location,
            reporter,
            status,
            filepath
        ))
        conn.commit()

    return redirect("/")
# =========================
# STARTUP
# =========================
def start_background():
    t = threading.Thread(target=camera_worker, daemon=True)
    t.start()


if __name__ == "__main__":
    init_db()
    ensure_db_schema()
    start_background()
    app.run(debug=True, port=5000, use_reloader=False)