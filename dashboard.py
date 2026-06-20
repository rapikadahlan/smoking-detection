import csv
import io
import os
import sqlite3
from collections import Counter
from datetime import date, datetime, timedelta

from flask import Flask, jsonify, render_template, Response

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DB_FILE = os.path.join(BASE_DIR, "smoking_violations.db")

app = Flask(__name__, template_folder="templates", static_folder="static")


def get_conn():
    conn = sqlite3.connect(DB_FILE)
    conn.row_factory = sqlite3.Row
    return conn


def ensure_db_schema():
    with get_conn() as conn:
        conn.execute("""
            CREATE TABLE IF NOT EXISTS violations (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                created_at TEXT NOT NULL,
                violation_date TEXT NOT NULL,
                violation_time TEXT NOT NULL,
                confidence REAL NOT NULL DEFAULT 0.0,
                note TEXT NOT NULL DEFAULT 'Merokok terdeteksi',
                source TEXT NOT NULL DEFAULT 'AI system'
            )
        """)
        cur = conn.cursor()
        cur.execute("PRAGMA table_info(violations)")
        cols = {row["name"] for row in cur.fetchall()}

        if "confidence" not in cols:
            conn.execute("ALTER TABLE violations ADD COLUMN confidence REAL NOT NULL DEFAULT 0.0")
        if "note" not in cols:
            conn.execute("ALTER TABLE violations ADD COLUMN note TEXT NOT NULL DEFAULT 'Merokok terdeteksi'")
        if "source" not in cols:
            conn.execute("ALTER TABLE violations ADD COLUMN source TEXT NOT NULL DEFAULT 'AI system'")
        conn.commit()


def fetch_all_violations():
    with get_conn() as conn:
        rows = conn.execute("""
            SELECT id, created_at, violation_date, violation_time, confidence, note, source
            FROM violations
            ORDER BY id DESC
        """).fetchall()
    return [dict(r) for r in rows]


def fetch_recent_violations(limit=12):
    with get_conn() as conn:
        rows = conn.execute("""
            SELECT id, created_at, violation_date, violation_time, confidence, note, source
            FROM violations
            ORDER BY id DESC
            LIMIT ?
        """, (limit,)).fetchall()
    return [dict(r) for r in rows]


def count_today():
    today = date.today().isoformat()
    with get_conn() as conn:
        return conn.execute(
            "SELECT COUNT(*) FROM violations WHERE violation_date = ?",
            (today,)
        ).fetchone()[0]


def count_week():
    start = (date.today() - timedelta(days=6)).isoformat()
    with get_conn() as conn:
        return conn.execute(
            "SELECT COUNT(*) FROM violations WHERE violation_date >= ?",
            (start,)
        ).fetchone()[0]


def get_daily_series(days=14):
    today = date.today()
    labels = []
    counts = []
    mapping = {}

    with get_conn() as conn:
        rows = conn.execute("""
            SELECT violation_date, COUNT(*) AS cnt
            FROM violations
            GROUP BY violation_date
        """).fetchall()

    for r in rows:
        mapping[r["violation_date"]] = r["cnt"]

    for i in range(days - 1, -1, -1):
        d = today - timedelta(days=i)
        ds = d.isoformat()
        labels.append(ds)
        counts.append(mapping.get(ds, 0))

    return labels, counts


def get_hourly_series():
    today = date.today().isoformat()
    labels = [f"{h:02d}:00" for h in range(24)]
    mapping = {f"{h:02d}": 0 for h in range(24)}

    with get_conn() as conn:
        rows = conn.execute("""
            SELECT substr(violation_time, 1, 2) AS hour, COUNT(*) AS cnt
            FROM violations
            WHERE violation_date = ?
            GROUP BY substr(violation_time, 1, 2)
        """, (today,)).fetchall()

    for r in rows:
        mapping[r["hour"]] = r["cnt"]

    counts = [mapping[f"{h:02d}"] for h in range(24)]
    return labels, counts


def get_source_breakdown():
    with get_conn() as conn:
        rows = conn.execute("""
            SELECT source, COUNT(*) AS cnt
            FROM violations
            GROUP BY source
            ORDER BY cnt DESC
        """).fetchall()

    labels = [r["source"] for r in rows] if rows else ["Belum ada data"]
    counts = [r["cnt"] for r in rows] if rows else [1]
    return labels, counts


def get_peak_day():
    with get_conn() as conn:
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


def get_peak_hour():
    today = date.today().isoformat()
    with get_conn() as conn:
        row = conn.execute("""
            SELECT substr(violation_time, 1, 2) AS hour, COUNT(*) AS cnt
            FROM violations
            WHERE violation_date = ?
            GROUP BY substr(violation_time, 1, 2)
            ORDER BY cnt DESC, hour DESC
            LIMIT 1
        """, (today,)).fetchone()
    if not row:
        return None, 0
    return f"{row['hour']}:00", row["cnt"]


def get_avg_per_day():
    with get_conn() as conn:
        row = conn.execute("""
            SELECT COUNT(*) AS total, COUNT(DISTINCT violation_date) AS days
            FROM violations
        """).fetchone()

    total = row["total"] or 0
    days = row["days"] or 0
    if days == 0:
        return 0.0
    return round(total / days, 2)


def get_last_alert():
    with get_conn() as conn:
        row = conn.execute("""
            SELECT created_at
            FROM violations
            ORDER BY id DESC
            LIMIT 1
        """).fetchone()
    return row["created_at"] if row else None


def build_report():
    total_all = len(fetch_all_violations())
    today = count_today()
    week = count_week()
    avg_day = get_avg_per_day()
    peak_day, peak_day_cnt = get_peak_day()
    peak_hour, peak_hour_cnt = get_peak_hour()
    last_alert = get_last_alert()

    if total_all == 0:
        title = "Belum ada pelanggaran tersimpan"
        body = "Dashboard siap menerima data dari app.py ketika sistem mendeteksi pelanggaran merokok."
        recommendation = "Jalankan deteksi realtime untuk mulai mengisi data."
    else:
        title = "Ringkasan monitoring aktif"
        body = (
            f"Hari ini terdeteksi {today} pelanggaran, "
            f"total mingguan {week}, dengan rata-rata {avg_day} per hari."
        )
        recommendation = (
            f"Puncak pelanggaran terjadi pada {peak_day} ({peak_day_cnt} kali) "
            f"dan jam paling ramai di {peak_hour} ({peak_hour_cnt} kali)."
            if peak_day and peak_hour
            else "Data belum cukup untuk analisis puncak."
        )

    return {
        "title": title,
        "body": body,
        "recommendation": recommendation,
        "last_alert": last_alert,
    }


def build_dashboard_payload():
    all_rows = fetch_all_violations()
    today = count_today()
    week = count_week()
    total_all = len(all_rows)
    avg_day = get_avg_per_day()
    peak_day, peak_day_cnt = get_peak_day()
    peak_hour, peak_hour_cnt = get_peak_hour()
    last_alert = get_last_alert()

    daily_labels, daily_counts = get_daily_series(14)
    hourly_labels, hourly_counts = get_hourly_series()
    source_labels, source_counts = get_source_breakdown()

    latest_conf = all_rows[0]["confidence"] if all_rows else 0.0
    latest_note = all_rows[0]["note"] if all_rows else "-"
    latest_time = all_rows[0]["created_at"] if all_rows else "-"

    if last_alert:
        try:
            dt = datetime.strptime(last_alert, "%Y-%m-%d %H:%M:%S")
            minutes_ago = int((datetime.now() - dt).total_seconds() // 60)
            live_status = f"Aktif • alert terakhir {minutes_ago} menit lalu"
        except Exception:
            live_status = "Aktif"
    else:
        live_status = "Belum ada data"

    report = build_report()

    return {
        "summary": {
            "total_all": total_all,
            "today": today,
            "week": week,
            "avg_day": avg_day,
            "peak_day": peak_day or "-",
            "peak_day_cnt": peak_day_cnt,
            "peak_hour": peak_hour or "-",
            "peak_hour_cnt": peak_hour_cnt,
            "latest_conf": round(latest_conf, 2),
            "latest_note": latest_note,
            "latest_time": latest_time,
            "live_status": live_status,
        },
        "charts": {
            "daily": {"labels": daily_labels, "values": daily_counts},
            "hourly": {"labels": hourly_labels, "values": hourly_counts},
            "source": {"labels": source_labels, "values": source_counts},
        },
        "recent": fetch_recent_violations(12),
        "report": report,
    }


@app.route("/")
def index():
    return render_template("dashboard.html")


@app.route("/api/dashboard")
def api_dashboard():
    return jsonify(build_dashboard_payload())


@app.route("/export/csv")
def export_csv():
    rows = fetch_all_violations()
    buffer = io.StringIO()
    writer = csv.writer(buffer)
    writer.writerow(["id", "created_at", "violation_date", "violation_time", "confidence", "note", "source"])
    for r in rows:
        writer.writerow([
            r["id"],
            r["created_at"],
            r["violation_date"],
            r["violation_time"],
            r["confidence"],
            r["note"],
            r["source"],
        ])
    csv_data = buffer.getvalue()
    buffer.close()

    return Response(
        csv_data,
        mimetype="text/csv",
        headers={"Content-Disposition": "attachment; filename=smoking_violations_report.csv"}
    )


if __name__ == "__main__":
    ensure_db_schema()
    app.run(debug=True, port=5000)