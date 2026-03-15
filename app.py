"""
Heatmap viewer UI: browse Grad-CAM results by model, corruption, and severity.

Run: python app.py
Open: http://127.0.0.1:5000
"""

from pathlib import Path
import sys

# Project root
ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

try:
    from flask import Flask, send_from_directory, jsonify, request, render_template_string
except ImportError:
    print("Install Flask: pip install flask")
    sys.exit(1)

from src.utils.io import load_yaml

app = Flask(__name__)

# Config
def get_heatmap_dir():
    config_path = ROOT / "configs" / "experiment.yaml"
    if config_path.exists():
        config = load_yaml(config_path)
        return ROOT / config.get("results", {}).get("heatmap_samples_dir", "results/heatmap_samples")
    return ROOT / "results" / "heatmap_samples"


HEATMAP_DIR = get_heatmap_dir()

RESULTS_DIR = ROOT / "results"


def get_metrics():
    """Load lead_stats and dasc_summary from results dir if present."""
    out = {"lead_stats": None, "dasc_summary": None}
    try:
        p = RESULTS_DIR / "lead_stats.json"
        if p.exists():
            from src.utils.io import load_json
            out["lead_stats"] = load_json(p)
    except Exception:
        pass
    try:
        p = RESULTS_DIR / "dasc_summary.json"
        if p.exists():
            from src.utils.io import load_json
            out["dasc_summary"] = load_json(p)
    except Exception:
        pass
    return out


@app.route("/")
def index():
    """Model list + viewer."""
    models = []
    if HEATMAP_DIR.exists():
        for p in sorted(HEATMAP_DIR.iterdir()):
            if p.is_dir() and not p.name.startswith("."):
                models.append(p.name)
    return render_template_string(INDEX_HTML, models=models, heatmap_dir=str(HEATMAP_DIR))


@app.route("/api/models")
def api_models():
    """List model names (subdirs of heatmap_samples)."""
    models = []
    if HEATMAP_DIR.exists():
        for p in sorted(HEATMAP_DIR.iterdir()):
            if p.is_dir() and not p.name.startswith("."):
                models.append(p.name)
    return jsonify(models=models)


@app.route("/api/models/<model>/corruptions")
def api_corruptions(model):
    """List corruptions for a model."""
    path = HEATMAP_DIR / model
    if not path.exists() or not path.is_dir():
        return jsonify(corruptions=[]), 404
    corruptions = [p.name for p in sorted(path.iterdir()) if p.is_dir() and not p.name.startswith(".")]
    return jsonify(corruptions=corruptions)


@app.route("/api/models/<model>/<corruption>/severities")
def api_severities(model, corruption):
    """List severity folders (L0..L4) for model/corruption."""
    path = HEATMAP_DIR / model / corruption
    if not path.exists() or not path.is_dir():
        return jsonify(severities=[]), 404
    severities = [p.name for p in sorted(path.iterdir()) if p.is_dir() and p.name.startswith("L")]
    return jsonify(severities=severities)


@app.route("/api/models/<model>/<corruption>/<severity>/images")
def api_images(model, corruption, severity):
    """List image filenames for model/corruption/severity."""
    path = HEATMAP_DIR / model / corruption / severity
    if not path.exists() or not path.is_dir():
        return jsonify(images=[]), 404
    images = [p.name for p in sorted(path.iterdir()) if p.suffix.lower() in (".png", ".jpg", ".jpeg")]
    return jsonify(images=images)


@app.route("/api/models/<model>/<corruption>/samples")
def api_samples(model, corruption):
    """List sample IDs that have L0–L4 all present (intersection). Only these are shown in the UI."""
    base = HEATMAP_DIR / model / corruption
    if not base.exists() or not base.is_dir():
        return jsonify(samples=[]), 404
    by_level = {}
    for p in sorted(base.iterdir()):
        if p.is_dir() and p.name.startswith("L"):
            by_level[p.name] = {
                f.name for f in p.iterdir()
                if f.suffix.lower() in (".png", ".jpg", ".jpeg")
            }
    required = ["L0", "L1", "L2", "L3", "L4"]
    if not all(lev in by_level for lev in required):
        return jsonify(samples=[])
    full = by_level["L0"]
    for lev in required[1:]:
        full = full & by_level[lev]
    return jsonify(samples=sorted(full))


@app.route("/api/models/<model>/<corruption>/sample/<path:sample_id>")
def api_sample_severities(model, corruption, sample_id):
    """For one sample (filename), return severities that have it and image URLs."""
    base = HEATMAP_DIR / model / corruption
    if not base.exists() or not base.is_dir():
        return jsonify(severities=[]), 404
    if not (sample_id.endswith(".png") or sample_id.endswith(".jpg") or sample_id.endswith(".jpeg")):
        sample_id = sample_id + ".png"
    result = []
    for p in sorted(base.iterdir()):
        if p.is_dir() and p.name.startswith("L"):
            f = p / sample_id
            if f.exists() and f.is_file():
                result.append({"severity": p.name, "url": f"/heatmaps/{model}/{corruption}/{p.name}/{sample_id}"})
    return jsonify(severities=result)


@app.route("/api/metrics")
def api_metrics():
    """Overall experiment metrics: lead_stats, dasc_summary."""
    return jsonify(get_metrics())


@app.route("/heatmaps/<path:filepath>")
def serve_heatmap(filepath):
    """Serve a single image from heatmap_samples."""
    base = HEATMAP_DIR.resolve()
    path = (base / filepath).resolve()
    try:
        path.relative_to(base)
    except ValueError:
        return "Forbidden", 403
    if not path.exists() or not path.is_file():
        return "Not found", 404
    return send_from_directory(path.parent, path.name)


INDEX_HTML = """<!DOCTYPE html>
<html lang="ko">
<head>
  <meta charset="UTF-8">
  <meta name="viewport" content="width=device-width, initial-scale=1.0">
  <title>Heatmap Viewer · 샘플별 변조 단계 비교</title>
  <style>
    :root {
      --bg: #0f1419;
      --card: #1a2332;
      --accent: #00d4aa;
      --text: #e6edf3;
      --muted: #8b949e;
      --border: rgba(139,148,158,0.25);
    }
    * { box-sizing: border-box; }
    body {
      font-family: 'Segoe UI', system-ui, sans-serif;
      background: var(--bg);
      color: var(--text);
      margin: 0;
      padding: 0;
      line-height: 1.5;
      min-height: 100vh;
    }
    .header {
      padding: 16px 24px;
      background: var(--card);
      border-bottom: 1px solid var(--border);
    }
    .header h1 { font-size: 1.35rem; margin: 0 0 4px 0; color: var(--accent); }
    .header .subtitle { color: var(--muted); font-size: 0.85rem; margin: 0; }

    .metrics {
      display: grid;
      grid-template-columns: repeat(auto-fill, minmax(160px, 1fr));
      gap: 12px;
      padding: 16px 24px;
      background: var(--card);
      margin: 0 24px 16px 24px;
      border-radius: 12px;
      border: 1px solid var(--border);
    }
    .metric {
      padding: 10px 12px;
      background: var(--bg);
      border-radius: 8px;
      border: 1px solid var(--border);
    }
    .metric .label { font-size: 0.7rem; color: var(--muted); text-transform: uppercase; letter-spacing: 0.05em; margin-bottom: 4px; }
    .metric .value { font-size: 1.1rem; font-weight: 600; color: var(--accent); }

    .layout {
      display: flex;
      padding: 0 24px 24px;
      gap: 20px;
      min-height: 60vh;
    }
    .sidebar {
      width: 280px;
      flex-shrink: 0;
      background: var(--card);
      border-radius: 12px;
      border: 1px solid var(--border);
      display: flex;
      flex-direction: column;
      max-height: 70vh;
    }
    .sidebar .sidebar-head {
      padding: 12px 14px;
      border-bottom: 1px solid var(--border);
      font-size: 0.85rem;
      color: var(--muted);
    }
    .sidebar .sample-list {
      flex: 1;
      overflow-y: auto;
      padding: 8px;
    }
    .sample-item {
      padding: 10px 12px;
      margin-bottom: 4px;
      border-radius: 8px;
      font-size: 0.8rem;
      cursor: pointer;
      word-break: break-all;
      color: var(--text);
      background: transparent;
      border: 1px solid transparent;
    }
    .sample-item:hover { background: rgba(0,212,170,0.1); border-color: var(--border); }
    .sample-item.active { background: rgba(0,212,170,0.2); border-color: var(--accent); color: var(--accent); }

    .filters-inline {
      display: flex;
      flex-wrap: wrap;
      gap: 12px;
      align-items: center;
      padding: 12px 14px;
      border-bottom: 1px solid var(--border);
    }
    .filters-inline label { color: var(--muted); font-size: 0.8rem; margin-right: 6px; }
    .filters-inline select {
      background: var(--bg);
      color: var(--text);
      border: 1px solid var(--border);
      border-radius: 6px;
      padding: 6px 10px;
      font-size: 0.9rem;
      min-width: 120px;
    }

    .main {
      flex: 1;
      min-width: 0;
      background: var(--card);
      border-radius: 12px;
      border: 1px solid var(--border);
      padding: 20px;
      display: flex;
      flex-direction: column;
    }
    .main .sample-title {
      font-size: 0.85rem;
      color: var(--muted);
      margin-bottom: 16px;
      word-break: break-all;
    }
    .severity-row {
      display: flex;
      flex-wrap: wrap;
      gap: 16px;
      align-items: flex-start;
      justify-content: flex-start;
    }
    .severity-cell {
      flex: 1 1 140px;
      max-width: 220px;
      background: var(--bg);
      border-radius: 8px;
      overflow: hidden;
      border: 1px solid var(--border);
    }
    .severity-cell .sev-label {
      padding: 8px 10px;
      font-size: 0.75rem;
      font-weight: 600;
      color: var(--accent);
      border-bottom: 1px solid var(--border);
    }
    .severity-cell img {
      width: 100%;
      height: auto;
      display: block;
      cursor: pointer;
    }
    .empty-main {
      color: var(--muted);
      text-align: center;
      padding: 48px 24px;
    }
    .loading { color: var(--muted); padding: 24px; }

    .modal {
      display: none;
      position: fixed;
      inset: 0;
      background: rgba(0,0,0,0.9);
      z-index: 100;
      align-items: center;
      justify-content: center;
      padding: 24px;
    }
    .modal.show { display: flex; }
    .modal img { max-width: 100%; max-height: 90vh; border-radius: 8px; }
    .modal .close { position: absolute; top: 16px; right: 24px; color: #fff; font-size: 28px; cursor: pointer; }
  </style>
</head>
<body>
  <div class="header">
    <h1>Heatmap Viewer</h1>
    <p class="subtitle">샘플 선택 후 동일 이미지의 변조 단계(L0~L4)별 Grad-CAM 비교 · 전체 실험 지표</p>
  </div>

  <div id="metrics" class="metrics"></div>

  <div class="layout">
    <aside class="sidebar">
      <div class="filters-inline">
        <div>
          <label>모델</label>
          <select id="model">
            <option value="">선택</option>
            {% for m in models %}
            <option value="{{ m }}">{{ m }}</option>
            {% endfor %}
          </select>
        </div>
        <div>
          <label>Corruption</label>
          <select id="corruption" disabled>
            <option value="">선택</option>
          </select>
        </div>
      </div>
      <div class="sidebar-head">샘플 목록 (선택 시 동일 이미지 L0~L4 표시)</div>
      <div id="sampleList" class="sample-list"></div>
    </aside>

    <main class="main">
      <div id="sampleTitle" class="sample-title" style="display:none;"></div>
      <div id="severityRow" class="severity-row"></div>
      <div id="emptyMain" class="empty-main">모델 · Corruption를 선택한 뒤, 왼쪽 목록에서 샘플을 선택하세요.</div>
      <div id="loadingMain" class="loading" style="display:none;">로딩 중...</div>
    </main>
  </div>

  <div id="modal" class="modal">
    <span class="close" onclick="document.getElementById('modal').classList.remove('show')">&times;</span>
    <img id="modalImg" src="" alt="">
  </div>

  <script>
    const modelEl = document.getElementById('model');
    const corruptionEl = document.getElementById('corruption');
    const sampleListEl = document.getElementById('sampleList');
    const sampleTitleEl = document.getElementById('sampleTitle');
    const severityRowEl = document.getElementById('severityRow');
    const emptyMainEl = document.getElementById('emptyMain');
    const loadingMainEl = document.getElementById('loadingMain');
    const metricsEl = document.getElementById('metrics');
    const modal = document.getElementById('modal');
    const modalImg = document.getElementById('modalImg');

    async function loadMetrics() {
      try {
        const r = await fetch('/api/metrics');
        const d = await r.json();
        let html = '';
        const ls = d.lead_stats;
        if (ls) {
          const nTotal = ls.n_total || 0;
          const nLead = ls.n_lead || 0;
          const nCoincident = ls.n_coincident || 0;
          const nLag = ls.n_lag || 0;
          const denom = nLead + nCoincident + nLag || 1;
          const leadRate = denom > 0 ? (100 * nLead / denom).toFixed(1) : '-';
          const leadCoincidentPct = nTotal > 0 ? (100 * (nLead + nCoincident) / nTotal).toFixed(1) : '-';
          html += '<div class="metric"><div class="label">선행률 (Lead %)</div><div class="value">' + leadRate + '%</div></div>';
          html += '<div class="metric"><div class="label">선행+동시 비율</div><div class="value">' + leadCoincidentPct + '%</div></div>';
          html += '<div class="metric"><div class="label">선행 건수</div><div class="value">' + nLead + '</div></div>';
          html += '<div class="metric"><div class="label">동시 건수</div><div class="value">' + nCoincident + '</div></div>';
          html += '<div class="metric"><div class="label">총 이벤트</div><div class="value">' + nTotal + '</div></div>';
          html += '<div class="metric"><div class="label">평균 Lead (프레임)</div><div class="value">' + (ls.mean_lead != null ? ls.mean_lead.toFixed(2) : '-') + '</div></div>';
          if (ls.sign_test && ls.sign_test.p_value != null)
            html += '<div class="metric"><div class="label">Sign test p-value</div><div class="value">' + ls.sign_test.p_value.toExponential(2) + '</div></div>';
          if (ls.permutation_test && ls.permutation_test.p_value != null)
            html += '<div class="metric"><div class="label">Permutation p-value</div><div class="value">' + ls.permutation_test.p_value.toExponential(2) + '</div></div>';
        }
        const ds = d.dasc_summary;
        if (ds && ds.miss_rate_curve && ds.miss_rate_curve.length) {
          const byCorr = {};
          ds.miss_rate_curve.forEach(function(e) {
            if (!byCorr[e.corruption]) byCorr[e.corruption] = [];
            byCorr[e.corruption].push(e);
          });
          Object.keys(byCorr).forEach(function(c) {
            const arr = byCorr[c];
            const l4 = arr.find(function(x) { return x.severity === 4; });
            if (l4) html += '<div class="metric"><div class="label">Miss rate ' + c + ' L4</div><div class="value">' + (l4.miss_rate * 100).toFixed(1) + '%</div></div>';
          });
        }
        metricsEl.innerHTML = html || '<div class="metric"><div class="label">지표</div><div class="value">데이터 없음</div></div>';
      } catch (e) {
        metricsEl.innerHTML = '<div class="metric"><div class="label">지표</div><div class="value">로드 실패</div></div>';
      }
    }

    modelEl.addEventListener('change', async () => {
      corruptionEl.innerHTML = '<option value="">선택</option>';
      corruptionEl.disabled = true;
      sampleListEl.innerHTML = '';
      severityRowEl.innerHTML = '';
      emptyMainEl.style.display = 'block';
      sampleTitleEl.style.display = 'none';
      const m = modelEl.value;
      if (!m) return;
      const r = await fetch('/api/models/' + encodeURIComponent(m) + '/corruptions');
      const d = await r.json();
      (d.corruptions || []).forEach(c => {
        const o = document.createElement('option');
        o.value = c;
        o.textContent = c;
        corruptionEl.appendChild(o);
      });
      corruptionEl.disabled = false;
    });

    corruptionEl.addEventListener('change', async () => {
      sampleListEl.innerHTML = '';
      severityRowEl.innerHTML = '';
      emptyMainEl.style.display = 'block';
      sampleTitleEl.style.display = 'none';
      const m = modelEl.value;
      const c = corruptionEl.value;
      if (!m || !c) return;
      const r = await fetch('/api/models/' + encodeURIComponent(m) + '/' + encodeURIComponent(c) + '/samples');
      const d = await r.json();
      const samples = d.samples || [];
      samples.forEach(name => {
        const item = document.createElement('button');
        item.type = 'button';
        item.className = 'sample-item';
        item.textContent = name.replace(/\\.(png|jpg|jpeg)$/i, '');
        item.dataset.sample = name;
        item.onclick = () => selectSample(m, c, name);
        sampleListEl.appendChild(item);
      });
      if (samples.length === 0) sampleListEl.innerHTML = '<span style="color:var(--muted);font-size:0.85rem;">샘플 없음</span>';
    });

    function selectSample(model, corruption, sampleId) {
      document.querySelectorAll('.sample-item').forEach(el => { el.classList.remove('active'); if (el.dataset.sample === sampleId) el.classList.add('active'); });
      sampleTitleEl.textContent = sampleId.replace(/\\.(png|jpg|jpeg)$/i, '');
      sampleTitleEl.style.display = 'block';
      emptyMainEl.style.display = 'none';
      loadingMainEl.style.display = 'block';
      severityRowEl.innerHTML = '';
      fetch('/api/models/' + encodeURIComponent(model) + '/' + encodeURIComponent(corruption) + '/sample/' + encodeURIComponent(sampleId))
        .then(r => r.json())
        .then(d => {
          loadingMainEl.style.display = 'none';
          const sevs = d.severities || [];
          severityRowEl.innerHTML = '';
          sevs.forEach(({ severity, url }) => {
            const cell = document.createElement('div');
            cell.className = 'severity-cell';
            cell.innerHTML = '<div class="sev-label">' + severity + '</div>';
            const img = document.createElement('img');
            img.src = url;
            img.alt = severity;
            img.loading = 'lazy';
            img.onclick = () => { modalImg.src = url; modal.classList.add('show'); };
            cell.appendChild(img);
            severityRowEl.appendChild(cell);
          });
          if (sevs.length === 0) { emptyMainEl.textContent = '이 샘플에 대한 이미지가 없습니다.'; emptyMainEl.style.display = 'block'; }
        })
        .catch(() => { loadingMainEl.style.display = 'none'; emptyMainEl.textContent = '로드 실패'; emptyMainEl.style.display = 'block'; });
    }

    loadMetrics();
  </script>
</body>
</html>
"""


def main():
    if not HEATMAP_DIR.exists():
        print(f"Heatmap directory not found: {HEATMAP_DIR}")
        print("Run scripts/05_gradcam_failure_analysis.py first to generate heatmaps.")
    else:
        print(f"Heatmap root: {HEATMAP_DIR}")
    print("Open http://127.0.0.1:5000")
    # host="127.0.0.1" = local only; use "0.0.0.0" to allow LAN access
    app.run(host="127.0.0.1", port=5000, debug=False)


if __name__ == "__main__":
    main()
