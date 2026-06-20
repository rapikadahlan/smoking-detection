let dailyChart = null;
let hourlyChart = null;

const $ = (id) => document.getElementById(id);

function fmtNumber(n) {
  return new Intl.NumberFormat("id-ID").format(n ?? 0);
}

function escapeHtml(str) {
  return String(str ?? "")
    .replaceAll("&", "&amp;")
    .replaceAll("<", "&lt;")
    .replaceAll(">", "&gt;")
    .replaceAll('"', "&quot;")
    .replaceAll("'", "&#039;");
}

function renderSummary(data) {
  const s = data.summary;
  $("mTotal").textContent = fmtNumber(s.total_all);
  $("mToday").textContent = fmtNumber(s.today);
  $("mWeek").textContent = fmtNumber(s.week);
  $("mAvg").textContent = s.avg_day;

  $("mPeakDay").textContent = `${s.peak_day} (${s.peak_day_cnt})`;
  $("mPeakHour").textContent = `${s.peak_hour} (${s.peak_hour_cnt})`;
  $("mLatestConf").textContent = Number(s.latest_conf || 0).toFixed(2);

  $("liveStatus").textContent = s.live_status;
}

function renderCharts(data) {
  if (dailyChart) dailyChart.destroy();
  if (hourlyChart) hourlyChart.destroy();

  dailyChart = new Chart($("dailyChart"), {
    type: "bar",
    data: {
      labels: data.charts.daily.labels,
      datasets: [{
        label: "Jumlah Pelanggaran",
        data: data.charts.daily.values,
        backgroundColor: "rgba(124, 196, 255, 0.72)",
        borderColor: "rgba(124, 196, 255, 1)",
        borderWidth: 1,
        borderRadius: 10,
      }]
    },
    options: {
      responsive: true,
      maintainAspectRatio: false,
      plugins: {
        legend: { labels: { color: "#eaf2ff" } }
      },
      scales: {
        x: {
          ticks: { color: "#93a4bf" },
          grid: { color: "rgba(255,255,255,.05)" }
        },
        y: {
          beginAtZero: true,
          ticks: { color: "#93a4bf" },
          grid: { color: "rgba(255,255,255,.07)" }
        }
      }
    }
  });

  hourlyChart = new Chart($("hourlyChart"), {
    type: "bar",
    data: {
      labels: data.charts.hourly.labels,
      datasets: [{
        label: "Pelanggaran per Jam",
        data: data.charts.hourly.values,
        backgroundColor: "rgba(159, 123, 255, 0.72)",
        borderColor: "rgba(159, 123, 255, 1)",
        borderWidth: 1,
        borderRadius: 10,
      }]
    },
    options: {
      responsive: true,
      maintainAspectRatio: false,
      plugins: {
        legend: { labels: { color: "#eaf2ff" } }
      },
      scales: {
        x: {
          ticks: { color: "#93a4bf", maxRotation: 0, autoSkip: true },
          grid: { color: "rgba(255,255,255,.05)" }
        },
        y: {
          beginAtZero: true,
          ticks: { color: "#93a4bf" },
          grid: { color: "rgba(255,255,255,.07)" }
        }
      }
    }
  });
}

function renderRecentTable(rows) {
  const tbody = document.getElementById("violationsTableBody");

  if (!rows || rows.length === 0) {
    tbody.innerHTML = `<tr><td colspan="10" class="text-center py-4 text-muted">Belum ada data pelanggaran.</td></tr>`;
    return;
  }

  tbody.innerHTML = rows.map(r => `
    <tr>
      <td>#${escapeHtml(r.id)}</td>
      <td>${escapeHtml(r.violation_time)}</td>
      <td>${escapeHtml(r.violation_date)}</td>
      <td>${escapeHtml(r.location || "-")}</td>
      <td>${escapeHtml(r.reporter || "-")}</td>
      <td>${escapeHtml(r.status || "-")}</td>
      <td><span class="badge text-bg-info">${Number(r.confidence || 0).toFixed(2)}</span></td>
      <td>${escapeHtml(r.note || "-")}</td>
      <td>${escapeHtml(r.source || "-")}</td>
      <td>${r.evidence ? `<img src="/${escapeHtml(r.evidence)}" width="70" style="border-radius:8px;object-fit:cover;">` : "-"}</td>
    </tr>
  `).join("");
}

function renderNotifications(rows) {
  const el = $("notificationList");
  if (!rows || rows.length === 0) {
    el.innerHTML = `
      <div class="notification-item">
        <div class="time">Belum ada notifikasi</div>
        <div class="title">Sistem siap memantau</div>
        <div class="desc">Ketika pelanggaran terdeteksi, notifikasi akan muncul di sini.</div>
      </div>
    `;
    return;
  }

  el.innerHTML = rows.slice(0, 6).map(r => `
    <div class="notification-item">
      <div class="time">${escapeHtml(r.created_at)}</div>
      <div class="title">
        <i class="fa-solid fa-triangle-exclamation me-2 text-warning"></i>
        ${escapeHtml(r.note || "Merokok terdeteksi")}
      </div>
      <div class="desc">
        Lokasi: ${escapeHtml(r.location || "-")} • Status: ${escapeHtml(r.status || "-")} • Source: ${escapeHtml(r.source || "-")} • Confidence: ${Number(r.confidence || 0).toFixed(2)} • ID: ${escapeHtml(r.id)}
      </div>
    </div>
  `).join("");
}

function renderReport(data) {
  const s = data.summary || {};
  const latest = (data.recent && data.recent.length > 0) ? data.recent[0] : null;
  const report = data.report || {};

  $("reportTitle").textContent = report.title || "Laporan Operasional Pelanggaran Merokok";

  const detailLines = [
    `Scope: ${s.scope || "Semua lokasi"}`,
    `Total pelanggaran tersimpan: ${fmtNumber(s.total_all)}`,
    `Pelanggaran hari ini: ${fmtNumber(s.today)}`,
    `Pelanggaran 7 hari terakhir: ${fmtNumber(s.week)}`,
    `Rata-rata per hari: ${s.avg_day}`,
    `Hari puncak: ${s.peak_day} (${s.peak_day_cnt})`,
    `Jam puncak: ${s.peak_hour} (${s.peak_hour_cnt})`,
    latest ? `Lokasi terakhir: ${latest.location || "-"}` : null,
    latest ? `Pelapor terakhir: ${latest.reporter || "-"}` : null,
    latest ? `Status terakhir: ${latest.status || "-"}` : null,
    latest ? `Confidence terakhir: ${Number(latest.confidence || 0).toFixed(2)}` : null,
    latest ? `Catatan terakhir: ${latest.note || "-"}` : null,
    latest ? `Update terakhir: ${latest.created_at || "-"}` : null,
  ].filter(Boolean);

  $("reportBody").innerHTML = `
    <ul class="mb-0 ps-3">
      ${detailLines.map(line => `<li>${escapeHtml(line)}</li>`).join("")}
    </ul>
  `;

  $("reportRecommendation").textContent = report.recommendation || "Belum ada rekomendasi.";
  $("reportConf").textContent = Number(s.latest_conf || 0).toFixed(2);
  $("reportPeakDay").textContent = s.peak_day === "-" ? "-" : `${s.peak_day} (${s.peak_day_cnt})`;
  $("reportPeakHour").textContent = s.peak_hour === "-" ? "-" : `${s.peak_hour} (${s.peak_hour_cnt})`;
}

function renderHeaderInfo(data) {
  $("mTotal").textContent = fmtNumber(data.summary.total_all);
  $("mToday").textContent = fmtNumber(data.summary.today);
  $("mWeek").textContent = fmtNumber(data.summary.week);
  $("mAvg").textContent = data.summary.avg_day;

  $("mPeakDay").textContent = `${data.summary.peak_day} (${data.summary.peak_day_cnt})`;
  $("mPeakHour").textContent = `${data.summary.peak_hour} (${data.summary.peak_hour_cnt})`;
  $("mLatestConf").textContent = Number(data.summary.latest_conf || 0).toFixed(2);

  const scope = data.summary.scope && data.summary.scope !== "Semua lokasi"
    ? ` • ${data.summary.scope}`
    : "";
  $("liveStatus").textContent = `${data.summary.live_status}${scope}`;
}

function filterTable() {
  const q = $("tableSearch").value.toLowerCase().trim();
  const rows = Array.from(document.querySelectorAll("#violationsTableBody tr"));

  rows.forEach(row => {
    const text = row.innerText.toLowerCase();
    row.style.display = text.includes(q) ? "" : "none";
  });
}

async function loadDashboard() {
  try {
    const params = new URLSearchParams(window.location.search);
    const location = params.get("location") || "";
    const date = params.get("date") || "";

    const res = await fetch(`/api/dashboard?location=${encodeURIComponent(location)}&date=${encodeURIComponent(date)}`);
    const data = await res.json();

    renderHeaderInfo(data);
    renderCharts(data);
    renderRecentTable(data.recent);
    renderNotifications(data.recent);
    renderReport(data);
    filterTable();
  } catch (err) {
    console.error("ERROR DASHBOARD:", err);
  }
}

document.addEventListener("DOMContentLoaded", () => {
  loadDashboard();
  $("refreshBtn").addEventListener("click", loadDashboard);
  $("tableSearch").addEventListener("input", filterTable);

  setInterval(loadDashboard, 3000);
});