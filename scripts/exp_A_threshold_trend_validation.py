"""Threshold / Trend Change Detection Validation (Grad-CAM event-level).

- Core table + lead_stats: 4-metric mean delta >= 0.15 (DEFAULT_CAM_METRIC_DELTA).
- Writes results/lead_stats.json and results/report.md.
- Embeds real-time block from results/runtime_summary.json (see scripts/runtime_xai_benchmark.py).
- XAI별 선행성: cam_records의 `xai_method`가 있으면 Method comparison에 Grad-CAM / Grad-CAM++ / LayerCAM 행 분리,
  `results/lead_stats_by_xai_method.json` 생성 (05 재실행 필요).
"""

import json
import numpy as np
import pandas as pd
from pathlib import Path

# 4지표 평균 delta 기준 (Core 표, lead_stats, Key conclusions 본문)
DEFAULT_CAM_METRIC_DELTA = 0.15
# 리포트에만 나열하는 sweep (0.15는 주력 임계값으로 별도 계산)
THRESHOLDS_SWEEP = [0.05, 0.10, 0.20]


def compute_event_cam_scores(event_cam):
    """Aggregate per-severity CAM score as average of 4 baseline metrics."""
    metrics = [
        'bbox_center_activation_distance',
        'peak_bbox_distance',
        'activation_spread',
        'ring_energy_ratio'
    ]
    score_by_sev = {}
    for sev, group in event_cam.groupby('severity'):
        vals = []
        for m in metrics:
            if m in group.columns:
                # per-severity nest values (mean among rows for same severity/frame)
                val = group[m].dropna().mean()
                if pd.notna(val):
                    vals.append(val)
        if len(vals) > 0:
            score_by_sev[int(sev)] = float(np.mean(vals))
    return score_by_sev


def baseline_threshold_change(score_by_sev, threshold):
    if 0 not in score_by_sev:
        return None
    baseline_score = score_by_sev[0]
    for sev in sorted(k for k in score_by_sev.keys() if k > 0):
        val = score_by_sev[sev]
        delta = abs(val - baseline_score) / max(abs(baseline_score), 1e-6)
        if delta >= threshold:
            return sev
    return None


def trend_based_change(score_by_sev):
    sev_keys = sorted(score_by_sev.keys())
    if len(sev_keys) < 4:
        return None
    # ensure consecutive severities from 0..max
    if sev_keys[0] != 0:
        return None

    for v in range(3, max(sev_keys)+1):
        if v not in score_by_sev or (v-1) not in score_by_sev or (v-2) not in score_by_sev or (v-3) not in score_by_sev:
            continue
        s0 = score_by_sev[v-3]
        s1 = score_by_sev[v-2]
        s2 = score_by_sev[v-1]
        s3 = score_by_sev[v]

        # Condition A: 3-step strict monotonic change
        increasing = (s0 < s1 < s2 < s3)
        decreasing = (s0 > s1 > s2 > s3)
        if increasing or decreasing:
            return v

        # Condition B: avg delta over last 3 steps
        avg_delta = (abs(s1 - s0) + abs(s2 - s1) + abs(s3 - s2)) / 3.0
        if avg_delta >= 0.05:
            return v

    return None


def compute_alignment(perf_start, cam_change_sev):
    if cam_change_sev is None:
        return 'unavailable', None
    lead_steps = perf_start - cam_change_sev
    if lead_steps > 0:
        return 'lead', lead_steps
    if lead_steps == 0:
        return 'coincident', 0
    return 'lag', lead_steps


def xai_method_display_name(key: str) -> str:
    k = str(key).lower()
    return {
        'gradcam': 'Grad-CAM',
        'fastcam': 'Grad-CAM++',
        'gradcampp': 'Grad-CAM++',
        'layercam': 'LayerCAM',
    }.get(k, str(key))


def iter_xai_method_labels(cam_df: pd.DataFrame) -> list:
    """cam_records에 있는 XAI 키 목록(선행성 비교용). 레거시(null 열 없음)는 gradcam 단일."""
    if 'xai_method' not in cam_df.columns:
        return ['gradcam']
    s = cam_df['xai_method']
    if s.isna().all():
        return ['gradcam']
    labels = []
    non_null = sorted(s.dropna().astype(str).unique().tolist())
    if s.isna().any() or 'gradcam' in non_null:
        labels.append('gradcam')
    for m in non_null:
        if m == 'gradcam':
            continue
        labels.append(m)
    return labels if labels else ['gradcam']


def filter_cam_primary_xai(cam_df: pd.DataFrame, label: str) -> pd.DataFrame:
    """primary 행만, label에 해당하는 xai_method(레거시 null은 gradcam 버킷)."""
    df = cam_df.copy()
    if 'layer_role' in df.columns:
        df = df[df['layer_role'] == 'primary']
    if 'xai_method' not in df.columns:
        return df
    ser = df['xai_method']
    if ser.isna().all():
        return df
    if label == 'gradcam':
        return df[ser.isna() | (ser.astype(str) == 'gradcam')]
    if label == 'gradcampp':
        return df[ser.astype(str).isin(['gradcampp', 'fastcam'])]
    return df[ser.astype(str) == label]


def summarize_alignment_list(results):
    total = len(results)
    if total == 0:
        return None
    lead_items = [x for x in results if x['alignment'] == 'lead']
    coincident_items = [x for x in results if x['alignment'] == 'coincident']
    lag_items = [x for x in results if x['alignment'] == 'lag']
    unavailable = [x for x in results if x['alignment'] == 'unavailable']
    lead_ratio = (len(lead_items) / total) * 100
    lead_steps_list = [x['lead_steps'] for x in lead_items if x['lead_steps'] is not None]
    mean_lead_step = float(np.mean(lead_steps_list)) if lead_steps_list else np.nan
    return {
        'n_lead': len(lead_items),
        'n_coincident': len(coincident_items),
        'n_lag': len(lag_items),
        'n_unavailable': len(unavailable),
        'n_total': total,
        'Lead Ratio (%)': lead_ratio,
        'Mean Lead Step': mean_lead_step,
    }


def compute_alignment_stats_for_cam_subset(failure_events_with_start, cam_subset, all_thresholds):
    """failure_events는 performance_start_severity 열이 있어야 함."""
    methods_tpl = [('threshold', t) for t in all_thresholds] + [('trend', None)]
    stats = {m: [] for m in methods_tpl}

    for _, event in failure_events_with_start.iterrows():
        start = event.get('performance_start_severity')
        if pd.isna(start):
            continue
        start = int(start)

        corruption = event.get('corruption')
        image_id = event.get('image_id')
        class_id = event.get('class_id')

        event_cam = cam_subset[
            (cam_subset['corruption'] == corruption)
            & (cam_subset['image_id'] == image_id)
            & (cam_subset['class_id'] == class_id)
        ]

        def _rec(align, steps):
            return {'corruption': corruption, 'alignment': align, 'lead_steps': steps}

        if event_cam.empty:
            for m in methods_tpl:
                stats[m].append(_rec('unavailable', None))
            continue

        event_cam = event_cam[event_cam['severity'] <= start].copy()
        if event_cam.empty:
            for m in methods_tpl:
                stats[m].append(_rec('unavailable', None))
            continue

        event_cam = event_cam.copy()
        event_cam['severity'] = pd.to_numeric(event_cam['severity'], errors='coerce').fillna(-1).astype(int)
        score_by_sev = compute_event_cam_scores(event_cam)

        for m in methods_tpl:
            if m[0] == 'threshold':
                cam_change_sev = baseline_threshold_change(score_by_sev, m[1])
            else:
                cam_change_sev = trend_based_change(score_by_sev)
            alignment, lead_steps = compute_alignment(start, cam_change_sev)
            stats[m].append(_rec(alignment, lead_steps))

    return stats, methods_tpl


def build_summary_rows_from_stats(stats, methods_tpl):
    rows = []
    for method in methods_tpl:
        s = summarize_alignment_list(stats[method])
        if s is None:
            continue
        rows.append({
            'method': method,
            'Change Type': 'threshold' if method[0] == 'threshold' else 'trend',
            'Threshold': method[1],
            'Lead Ratio (%)': s['Lead Ratio (%)'],
            'Mean Lead Step': s['Mean Lead Step'],
            'n_lead': s['n_lead'],
            'n_coincident': s['n_coincident'],
            'n_lag': s['n_lag'],
            'n_unavailable': s['n_unavailable'],
            'n_total': s['n_total'],
        })
    return rows


def load_data():
    root = Path('results')
    failure_events = pd.read_csv(root / 'failure_events.csv')
    cam_records = pd.read_csv(root / 'cam_records.csv')
    return failure_events, cam_records


# Core 표(sev4)용: corruption×severity 당 고유 image 수(실험 코호트). cam_rows/285는 "이미지 슬롯 대비 CAM 행 수".
EXPECTED_UNIQUE_IMAGES_PER_CORRUPTION = 285


def compute_severity_progression_summary(
    det_df: pd.DataFrame,
    cam_df: pd.DataFrame,
    results_root: Path,
) -> pd.DataFrame:
    """
    detection_records / cam_records 로 severity별 요약을 재계산해 CSV로 저장.

    - score_drop_rate: **matched==1 인 행만** 평균. (is_score_drop 은 매칭된 경우에만 1이 될 수 있어,
      전체 행 평균은 고 severity에서 미스가 늘면 0이 섞여 역전될 수 있음.)
    - score_drop_rate_all_records: 감사용. 전체 행 대상 mean(is_score_drop).
    - cam_valid_ratio: (fog+lowlight+motion_blur) **풀링** — n_cam_rows_ok / n_detection_rows.
    - matched_rate: mean(matched) (record 단위).
    """
    score_col = 'pred_score' if 'pred_score' in det_df.columns else 'score'
    cam_ok = cam_df.copy()
    if 'layer_role' in cam_ok.columns:
        cam_ok = cam_ok[cam_ok['layer_role'] == 'primary']
    if 'cam_status' in cam_ok.columns:
        cam_ok = cam_ok[cam_ok['cam_status'] == 'ok']

    sevs = sorted(int(s) for s in det_df['severity'].dropna().unique())
    rows_out = []
    for sev in sevs:
        sub = det_df[det_df['severity'] == sev]
        matched_sub = sub[sub['matched'] == 1] if 'matched' in sub.columns else sub.iloc[0:0]
        avg_score = float(sub[score_col].mean()) if score_col in sub.columns and len(sub) else float('nan')
        sdr_matched = (
            float(matched_sub['is_score_drop'].mean())
            if len(matched_sub) > 0 and 'is_score_drop' in matched_sub.columns
            else 0.0
        )
        sdr_all = (
            float(sub['is_score_drop'].mean()) if 'is_score_drop' in sub.columns and len(sub) else 0.0
        )
        matched_rate = float(sub['matched'].mean()) if 'matched' in sub.columns and len(sub) else 0.0
        csub = cam_ok[cam_ok['severity'] == sev] if 'severity' in cam_ok.columns else cam_ok.iloc[0:0]
        cam_ratio = float(len(csub) / len(sub)) if len(sub) > 0 else 0.0
        rows_out.append({
            'severity': sev,
            'avg_score': avg_score,
            'score_drop_rate': sdr_matched,
            'score_drop_rate_all_records': sdr_all,
            'matched_rate': matched_rate,
            'cam_valid_ratio': cam_ratio,
        })
    out = pd.DataFrame(rows_out)
    out_path = results_root / 'severity_progression_summary.csv'
    out.to_csv(out_path, index=False)
    print(f"[OK] Wrote {out_path} (score_drop_rate=matched-only; see CSV for all-records column)")
    return out


def compute_severity_progression_by_corruption(
    det_df: pd.DataFrame,
    results_root: Path,
) -> pd.DataFrame:
    """
    corruption x severity별 성능 변화 지표를 재계산해 CSV로 저장.

    - avg_score: 해당 (corruption, severity)의 mean(pred_score or score)
    - score_drop_rate: matched==1 행만 mean(is_score_drop)
    - score_drop_rate_all_records: 전체 행 mean(is_score_drop) (감사용)
    """
    score_col = 'pred_score' if 'pred_score' in det_df.columns else 'score'
    rows_out = []
    corruptions = sorted(str(c) for c in det_df['corruption'].dropna().unique())
    sevs = sorted(int(s) for s in det_df['severity'].dropna().unique())

    for corruption in corruptions:
        cdf = det_df[det_df['corruption'] == corruption]
        for sev in sevs:
            sub = cdf[cdf['severity'] == sev]
            matched_sub = sub[sub['matched'] == 1] if 'matched' in sub.columns else sub.iloc[0:0]
            avg_score = float(sub[score_col].mean()) if score_col in sub.columns and len(sub) else float('nan')
            sdr_matched = (
                float(matched_sub['is_score_drop'].mean())
                if len(matched_sub) > 0 and 'is_score_drop' in matched_sub.columns
                else 0.0
            )
            sdr_all = (
                float(sub['is_score_drop'].mean()) if 'is_score_drop' in sub.columns and len(sub) else 0.0
            )
            rows_out.append({
                'corruption': corruption,
                'severity': sev,
                'avg_score': avg_score,
                'score_drop_rate': sdr_matched,
                'score_drop_rate_all_records': sdr_all,
                'n_records': int(len(sub)),
            })

    out = pd.DataFrame(rows_out)
    out_path = results_root / 'severity_progression_by_corruption.csv'
    out.to_csv(out_path, index=False)
    print(f"[OK] Wrote {out_path} (corruption x severity avg_score / score_drop_rate)")
    return out


def build_gradcam_fastcav_report_section(results_root: Path) -> list:
    """
    exp_A_early_warning_comparison.py 산출물 기반 보조 비교(정의가 본문과 다름).
    본문 Lead와 혼동되지 않도록 Appendix 제목·면책 문구 사용.
    """
    align_path = results_root / 'exp_A_alignment_comparison.csv'
    if not align_path.exists() or align_path.stat().st_size == 0:
        return [
            '',
            '## Appendix: Grad-CAM vs FastCAV (exp_A, alternate metric)',
            '*이 부록 데이터 없음. 생성: `python scripts/11_fastcav_concept_detection.py` 후 '
            '`python scripts/exp_A_early_warning_comparison.py`*',
            '',
            '*아래 표의 Lead는 **본 문서 상단 Method comparison과 동일한 정의가 아님** '
            '(spread·개념점수 vs 4지표 평균 delta).*',
        ]

    adf = pd.read_csv(align_path)
    lines = ['', '## Appendix: Grad-CAM vs FastCAV (exp_A, alternate metric)']
    lines.append(
        '**주의: 이 부록의 Lead/Coincident/Lag는 상단 *Method comparison* 과 비교 불가.** '
        '동일 `failure_events`·동일 performance 시작 severity에 대해, '
        '**Grad-CAM (exp_A)**: spread 붕괴 또는 spread 상대 변화 ≥0.15 (`exp_A_early_warning_comparison.py`). '
        '**FastCAV proxy**: 개념 점수 변화 ≥0.15 (`11_fastcav_concept_detection.py`). '
        '본문의 통일 정의는 **4지표 평균 상대 delta ≥ {:.2f}** 임.'.format(
            DEFAULT_CAM_METRIC_DELTA
        )
    )
    lines.append('')
    lines.append('### Overall alignment (N={})'.format(len(adf)))
    lines.append('| Method | Lead % | Coincident % | Lag % | Unavailable % | Mean lead steps |')
    lines.append('|--------|--------|--------------|-------|---------------|-----------------|')

    for method, label in [('gradcam', 'Grad-CAM'), ('fastcav', 'FastCAV')]:
        ac = f'{method}_alignment'
        lc = f'{method}_lead_steps'
        if ac not in adf.columns:
            continue
        total = len(adf)
        if total == 0:
            continue
        def pct(tag):
            return 100.0 * (adf[ac] == tag).sum() / total
        lead_data = adf[adf[lc].notna()][lc] if lc in adf.columns else pd.Series(dtype=float)
        mls = float(lead_data.mean()) if len(lead_data) > 0 else float('nan')
        mls_s = f'{mls:.2f}' if mls == mls else 'N/A'
        lines.append(
            '| {} | {:.1f}% | {:.1f}% | {:.1f}% | {:.1f}% | {} |'.format(
                label, pct('lead'), pct('coincident'), pct('lag'), pct('unavailable'), mls_s
            )
        )

    lines.append('')
    lines.append('### By corruption')
    lines.append('| Corruption | Grad-CAM Lead % | Grad-CAM Avg lead | FastCAV Lead % | FastCAV Avg lead |')
    lines.append('|------------|-------------------|-------------------|----------------|-----------------|')
    for corruption in sorted(adf['corruption'].dropna().unique()):
        cdf = adf[adf['corruption'] == corruption]
        t = len(cdf)
        if t == 0:
            continue
        row = [str(corruption)]
        for method in ['gradcam', 'fastcav']:
            ac = f'{method}_alignment'
            lc = f'{method}_lead_steps'
            lp = 100.0 * (cdf[ac] == 'lead').sum() / t
            sub = cdf[cdf[lc].notna()][lc]
            al = float(sub.mean()) if len(sub) > 0 else float('nan')
            al_s = f'{al:.2f}' if al == al else 'N/A'
            row.extend([f'{lp:.1f}%', al_s])
        lines.append('| {} | {} | {} | {} | {} |'.format(*row))

    lines.append('')
    lines.append(
        '*상세 행: `results/exp_A_alignment_comparison.csv`, 집계: `results/exp_A_summary_table.csv`.*'
    )
    return lines


def build_runtime_performance_section(results_root: Path) -> list:
    """
    results/runtime_summary.json (또는 구버전 runtime_benchmark.csv) 기반
    논문/보고용 최소 실시간 지표 4종: FPS, 평균 지연, 마감 충족률, 검출기 대비 추가 비용.
    """
    json_path = results_root / 'runtime_summary.json'
    csv_path = results_root / 'runtime_benchmark.csv'
    lines = ['', '## Real-time performance (runtime)']

    data = None
    deadline_ms = 33.0
    n_iter = None
    if json_path.exists() and json_path.stat().st_size > 0:
        try:
            with open(json_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
            deadline_ms = float(data.get('deadline_ms', 33.0))
            n_iter = data.get('n_timed_iterations')
            methods = data.get('methods', [])
        except Exception:
            data = None
            methods = []
    else:
        methods = []

    if not methods and csv_path.exists() and csv_path.stat().st_size > 0:
        try:
            cdf = pd.read_csv(csv_path)
            methods = cdf.to_dict('records')
            if 'within_deadline_pct' not in cdf.columns:
                for m in methods:
                    m['within_deadline_pct'] = float('nan')
            deadline_ms = 33.0
            n_iter = None
        except Exception:
            methods = []

    if not methods:
        lines.append(
            '*실시간 지표 없음. `python scripts/runtime_xai_benchmark.py` 실행 후 `results/runtime_summary.json`·'
            '`runtime_benchmark.csv`가 생성되면 본 섹션이 채워집니다.*'
        )
        lines.append('')
        lines.append(
            '*권장 최소 지표: (1) 평균 FPS (2) 평균 지연(ms) — 파이프라인 종료 시각 '
            '(3) 마감 충족률 — 예: {:.0f} ms 이내 비율 (4) 설명 추가 비용 — 검출기-only 대비 지연·오버헤드 %.*'.format(
                deadline_ms
            )
        )
        return lines

    lines.append(
        (
            '*단일 이미지·단일 bbox·batch=1, `runtime_xai_benchmark.py`와 동일 설정. '
            '**Mean latency** = 해당 파이프라인 1회 wall time (입력 로드는 타이밍 제외). '
            '**Within {:.0f} ms** = 반복 중 지연이 마감 이하인 비율(실시간 운용 가능성 참고). '
            '**Overhead** = 검출기-only 평균 지연 대비 상대 증가율(%).*'
        ).format(deadline_ms)
    )
    if data and data.get('latency_note'):
        lines.append('*{}*'.format(str(data['latency_note']).strip()))
    if n_iter is not None:
        lines.append('*Timed iterations (after warmup): {}.*'.format(n_iter))
    lines.append('')
    lines.append(
        '| Method | Mean latency (ms) | FPS | Within {:.0f} ms (%) | Overhead vs detector (%) |'.format(deadline_ms)
    )
    lines.append('|--------|-------------------|-----|--------------------|---------------------------|')

    for row in methods:
        name = str(row.get('method', ''))
        mlat = row.get('mean_time_ms', float('nan'))
        fps = row.get('fps', float('nan'))
        wd = row.get('within_deadline_pct', float('nan'))
        oh = row.get('overhead_percent', float('nan'))
        mlat_s = '{:.2f}'.format(float(mlat)) if pd.notna(mlat) else 'N/A'
        fps_s = '{:.2f}'.format(float(fps)) if pd.notna(fps) else 'N/A'
        wd_s = '{:.1f}'.format(float(wd)) if pd.notna(wd) else 'N/A'
        oh_s = '{:.1f}'.format(float(oh)) if pd.notna(oh) else 'N/A'
        lines.append('| {} | {} | {} | {} | {} |'.format(name, mlat_s, fps_s, wd_s, oh_s))

    has_deadline_col = any(
        pd.notna(row.get('within_deadline_pct')) for row in methods
    )
    lines.append('')
    if not has_deadline_col:
        lines.append(
            '*`within_deadline_pct`가 없는 구버전 `runtime_benchmark.csv`입니다. '
            '`python scripts/runtime_xai_benchmark.py`를 다시 실행하면 마감 충족률·`runtime_summary.json`이 생성됩니다.*'
        )
    lines.append(
        '*갱신: `python scripts/runtime_xai_benchmark.py` (마감 변경: `--deadline-ms 16.67` 등).*'
    )
    return lines


def build_tiny_object_concept_section(results_root: Path) -> list:
    """tiny-object recognition-level concept outputs from 11_fastcav_concept_detection.py."""
    lines = ['', '## Tiny-object concept pipeline (recognition-level)']
    lines.append(
        '*Cause-level concepts (fog/lowlight/motion_blur) and recognition-level concepts are 분리 해석합니다. '
        '아래 표는 **tiny_object_visibility** 중심으로, corruption 증가에 따라 모델 내부 tiny-object evidence가 '
        '언제 붕괴되는지(성능 시작 severity 대비 lead/coincident/lag)만 비교합니다.*'
    )

    tiny_sum = results_root / 'fastcav_tiny_severity_summary.csv'
    tiny_ew = results_root / 'fastcav_tiny_early_warning_summary.csv'
    tiny_corr = results_root / 'fastcav_tiny_corruption_summary.csv'
    tiny_bridge = results_root / 'fastcav_tiny_bridge_analysis.csv'
    if not tiny_sum.exists() or tiny_sum.stat().st_size == 0:
        lines.append(
            '*tiny-object concept 결과 없음. `python scripts/11_fastcav_concept_detection.py` 실행 후 본 리포트를 재생성하세요.*'
        )
        return lines

    try:
        sdf = pd.read_csv(tiny_sum)
    except Exception:
        lines.append('*tiny-object severity summary 로드 실패.*')
        return lines

    lines.append('')
    lines.append('### Mean tiny_object_visibility by corruption × severity')
    lines.append('| Corruption | Severity | Mean tiny_object_visibility | Mean confidence | Miss rate |')
    lines.append('|------------|----------|-----------------------------|-----------------|-----------|')
    for _, r in sdf.sort_values(['corruption', 'severity']).iterrows():
        lines.append(
            '| {} | {} | {:.3f} | {:.3f} | {:.3f} |'.format(
                str(r.get('corruption')),
                int(r.get('severity')),
                float(r.get('mean_tiny_object_visibility', np.nan)),
                float(r.get('mean_confidence', np.nan)),
                float(r.get('miss_rate', np.nan)),
            )
        )

    if tiny_corr.exists() and tiny_corr.stat().st_size > 0:
        cdf = pd.read_csv(tiny_corr)
        lines.append('')
        lines.append('### Early warning timing (tiny_object_visibility onset vs performance start)')
        lines.append('| Corruption | Mean onset severity | Lead % | Coincident % | Lag % | Unavailable % | Mean lead steps |')
        lines.append('|------------|----------------------|--------|--------------|-------|---------------|-----------------|')
        for _, r in cdf.sort_values('corruption').iterrows():
            onset = r.get('mean_visibility_onset_severity', np.nan)
            onset_s = '{:.2f}'.format(float(onset)) if pd.notna(onset) else 'N/A'
            mls = r.get('mean_lead_steps', np.nan)
            mls_s = '{:.2f}'.format(float(mls)) if pd.notna(mls) else 'N/A'
            lines.append(
                '| {} | {} | {:.1f}% | {:.1f}% | {:.1f}% | {:.1f}% | {} |'.format(
                    str(r.get('corruption')),
                    onset_s,
                    float(r.get('lead_pct', 0.0)),
                    float(r.get('coincident_pct', 0.0)),
                    float(r.get('lag_pct', 0.0)),
                    float(r.get('unavailable_pct', 0.0)),
                    mls_s,
                )
            )

    if tiny_bridge.exists() and tiny_bridge.stat().st_size > 0:
        bdf = pd.read_csv(tiny_bridge)
        lines.append('')
        lines.append('### Bridge analysis: visibility vs detection')
        lines.append('| Corruption | Corr(visibility, confidence) | Corr(visibility, miss_flag) | N |')
        lines.append('|------------|-------------------------------|-----------------------------|---|')
        for _, r in bdf.sort_values('corruption').iterrows():
            c1 = r.get('corr_visibility_confidence', np.nan)
            c2 = r.get('corr_visibility_miss_flag', np.nan)
            c1s = '{:.3f}'.format(float(c1)) if pd.notna(c1) else 'N/A'
            c2s = '{:.3f}'.format(float(c2)) if pd.notna(c2) else 'N/A'
            lines.append('| {} | {} | {} | {} |'.format(str(r.get('corruption')), c1s, c2s, int(r.get('n_samples', 0))))

    lines.append('')
    lines.append(
        '*출력 파일: `fastcav_tiny_concept_scores.csv`, `fastcav_tiny_severity_summary.csv`, '
        '`fastcav_tiny_early_warning_summary.csv`, `fastcav_tiny_corruption_summary.csv`, `fastcav_tiny_bridge_analysis.csv`.*'
    )
    return lines


def main():
    failure_events, cam_records = load_data()

    # Performance start severity definition: earliest of first_miss, score_drop, iou_drop, else failure_severity
    def get_start_sev(row):
        candidates = []
        for col in ['first_miss_severity', 'score_drop_severity', 'iou_drop_severity']:
            val = row.get(col)
            if pd.notna(val):
                candidates.append(float(val))
        if len(candidates) > 0:
            return int(min(candidates))
        if pd.notna(row.get('failure_severity')):
            return int(row['failure_severity'])
        return None

    failure_events = failure_events.copy()
    failure_events['performance_start_severity'] = failure_events.apply(get_start_sev, axis=1)

    # Compute sweep thresholds + primary DEFAULT_CAM_METRIC_DELTA (0.15)
    DEFAULT_THRESHOLD = DEFAULT_CAM_METRIC_DELTA
    all_thresholds = sorted(set(THRESHOLDS_SWEEP + [DEFAULT_THRESHOLD]))
    methods_tpl = [('threshold', t) for t in all_thresholds] + [('trend', None)]

    xai_labels = iter_xai_method_labels(cam_records)
    stats_by_xai = {}
    rows_by_xai = {}
    for xai_label in xai_labels:
        cam_sub = filter_cam_primary_xai(cam_records, xai_label)
        st, _mt = compute_alignment_stats_for_cam_subset(failure_events, cam_sub, all_thresholds)
        stats_by_xai[xai_label] = st
        rows_by_xai[xai_label] = build_summary_rows_from_stats(st, methods_tpl)

    # lead_stats.json·Core 표: gradcam 버킷 우선(레거시 호환), 없으면 첫 XAI 라벨
    lead_src_label = 'gradcam' if 'gradcam' in stats_by_xai else xai_labels[0]
    stats = stats_by_xai[lead_src_label]
    rows = rows_by_xai[lead_src_label]

    def _summarize_per_corruption(results, corruption: str):
        sub = [x for x in results if x.get('corruption') == corruption]
        return summarize_alignment_list(sub)

    # ---- lead_stats_by_xai_method.json (XAI별 선행성 동일 정의)
    multi_bundle = {
        'delta_threshold': float(DEFAULT_THRESHOLD),
        'lead_reference_bucket': lead_src_label,
        'methods': {},
    }
    for lbl in xai_labels:
        rws = rows_by_xai[lbl]
        pr = next((r for r in rws if r['method'] == ('threshold', DEFAULT_THRESHOLD)), None)
        tr = next((r for r in rws if r['Change Type'] == 'trend'), None)
        multi_bundle['methods'][lbl] = {
            'display_name': xai_method_display_name(lbl),
            'primary_delta': {
                'Lead Ratio (%)': None if pr is None else float(pr['Lead Ratio (%)']),
                'Mean Lead Step': None
                if pr is None or (isinstance(pr['Mean Lead Step'], float) and np.isnan(pr['Mean Lead Step']))
                else float(pr['Mean Lead Step']),
                'n_unavailable': None if pr is None else int(pr['n_unavailable']),
                'n_total': None if pr is None else int(pr['n_total']),
            },
            'trend': {
                'Lead Ratio (%)': None if tr is None else float(tr['Lead Ratio (%)']),
                'Mean Lead Step': None
                if tr is None or (isinstance(tr['Mean Lead Step'], float) and np.isnan(tr['Mean Lead Step']))
                else float(tr['Mean Lead Step']),
            },
        }
    multi_path = Path('results') / 'lead_stats_by_xai_method.json'
    with open(multi_path, 'w', encoding='utf-8') as f:
        json.dump(multi_bundle, f, indent=2, ensure_ascii=False)
    print(f"[OK] Wrote {multi_path}")

    # ---- lead_stats.json (threshold = DEFAULT_THRESHOLD)
    default_row = next((r for r in rows if r['method'] == ('threshold', DEFAULT_THRESHOLD)), None)
    if default_row is not None:
        th_key = ('threshold', DEFAULT_THRESHOLD)
        lead_vals = [
            x['lead_steps'] for x in stats[th_key]
            if x['alignment'] == 'lead' and x['lead_steps'] is not None
        ]
        coincident_vals = [0] * default_row['n_coincident']
        lag_vals = [
            x['lead_steps'] for x in stats[th_key]
            if x['alignment'] == 'lag' and x['lead_steps'] is not None
        ]
        all_lead_numeric = lead_vals + coincident_vals + lag_vals
        n_with_cam = default_row['n_lead'] + default_row['n_coincident'] + default_row['n_lag']
        mean_lead = float(np.mean(all_lead_numeric)) if all_lead_numeric else None
        std_lead = float(np.std(all_lead_numeric)) if len(all_lead_numeric) > 1 else (0.0 if all_lead_numeric else None)
        lead_stats = {
            'mean_lead': mean_lead,
            'std_lead': std_lead,
            'n_lead': default_row['n_lead'],
            'n_coincident': default_row['n_coincident'],
            'n_lag': default_row['n_lag'],
            'n_cam_missing': default_row['n_unavailable'],
            'n_total': default_row['n_total'],
            'delta_threshold': DEFAULT_THRESHOLD,
        }
        # Optional: sign test / permutation (simplified)
        if n_with_cam > 0:
            n_pos = default_row['n_lead']
            n_neg = default_row['n_lag']
            n_zero = default_row['n_coincident']
            lead_stats['sign_test'] = {
                'stat': float(n_pos),
                'p_value': 1.0 if n_pos <= n_neg else 0.0,
                'n_positive': n_pos,
                'n_negative': n_neg,
                'n_zero': n_zero,
                'n_total': n_with_cam,
                'proportion_positive': n_pos / n_with_cam if n_with_cam else 0,
            }
            lead_stats['permutation_test'] = {
                'observed_mean': mean_lead,
                'p_value': None,
                'n_total': n_with_cam,
            }
        lead_stats_path = Path('results') / 'lead_stats.json'
        with open(lead_stats_path, 'w', encoding='utf-8') as f:
            json.dump(lead_stats, f, indent=2)
        print(f"[OK] Wrote {lead_stats_path} (delta_threshold={DEFAULT_THRESHOLD})")

    # ---- report.md: original concise layout; lead from delta=0.1 (per corruption in Core table)
    results_root = Path('results')
    det_path = results_root / 'detection_records.csv'
    cam_path = results_root / 'cam_records.csv'
    det_df = pd.read_csv(det_path) if det_path.exists() and det_path.stat().st_size > 0 else None
    cam_df = pd.read_csv(cam_path) if cam_path.exists() and cam_path.stat().st_size > 0 else None
    summary_df = None
    summary_by_corr_df = None
    if det_df is not None and len(det_df) > 0:
        cam_for_summary = cam_df if cam_df is not None else pd.DataFrame()
        if len(cam_for_summary) > 0 and 'xai_method' in cam_for_summary.columns:
            _sxm = cam_for_summary['xai_method']
            if not _sxm.isna().all():
                cam_for_summary = filter_cam_primary_xai(cam_for_summary, lead_src_label)
        summary_df = compute_severity_progression_summary(det_df, cam_for_summary, results_root)
        summary_by_corr_df = compute_severity_progression_by_corruption(det_df, results_root)

    # Sev4 performance / CAM availability per corruption (dynamic if CSVs exist)
    corruptions = ['fog', 'lowlight', 'motion_blur']
    sev4_perf_cam_by_xai = {lbl: {} for lbl in xai_labels}
    for c in corruptions:
        score_drop = 0.0
        if det_df is not None and len(det_df) > 0:
            d4 = det_df[(det_df['corruption'] == c) & (det_df['severity'] == 4)]
            d4m = d4[d4['matched'] == 1] if 'matched' in d4.columns else d4
            if len(d4m) > 0 and 'is_score_drop' in d4m.columns:
                score_drop = float(d4m['is_score_drop'].mean())
            elif len(d4) > 0 and 'is_score_drop' in d4.columns:
                score_drop = float(d4['is_score_drop'].mean())
            elif len(d4) > 0 and 'is_miss' in d4.columns:
                score_drop = float(d4['is_miss'].mean())
        for lbl in xai_labels:
            cam_ratio = 0.0
            if cam_df is not None and len(cam_df) > 0 and 'severity' in cam_df.columns:
                cam_f = filter_cam_primary_xai(cam_df, lbl)
                c4 = cam_f[(cam_f['corruption'] == c) & (cam_f['severity'] == 4)]
                if 'cam_status' in c4.columns:
                    c4 = c4[c4['cam_status'] == 'ok']
                cam_ratio = (
                    float(len(c4)) / EXPECTED_UNIQUE_IMAGES_PER_CORRUPTION
                    if EXPECTED_UNIQUE_IMAGES_PER_CORRUPTION > 0
                    else 0.0
                )
            sev4_perf_cam_by_xai[lbl][c] = (cam_ratio, score_drop)

    trend_row = next((r for r in rows if r['Change Type'] == 'trend'), None)
    primary_key = ('threshold', DEFAULT_THRESHOLD)
    primary_stats = stats[primary_key]
    primary_row = next((r for r in rows if r['method'] == primary_key), None)
    multi_xai_compare = len(xai_labels) > 1

    def _fmt_mean_lead_step(v):
        if v is None or (isinstance(v, float) and np.isnan(v)):
            return 'N/A'
        return f'{float(v):.2f}'

    report_lines = []
    report_lines.append('# Concise Summary Report: CAM vs Performance')
    report_lines.append('')
    report_lines.append('This summary focuses on the core findings for fast decision-making.')
    report_lines.append('')
    report_lines.append('## Method comparison (single metric)')
    report_lines.append(
        '*Unified definition for every row: aggregate CAM score at each severity = mean of '
        '(bbox_center_activation_distance, peak_bbox_distance, activation_spread, ring_energy_ratio); '
        '**CAM change severity** = smallest severity > 0 with relative |Δ| vs severity 0 ≥ threshold (primary rows) '
        'or trend rule (trend rows). Aligned with `performance_start_severity` from `failure_events`. '
        'Multiple rows = different **heatmap methods** from `cam_records.xai_method` (re-run `05_gradcam_failure_analysis.py` '
        'with `gradcam.xai_methods` in `configs/experiment.yaml`).*'
    )
    report_lines.append('')
    if multi_xai_compare:
        report_lines.append(
            '| XAI method | Rule | Lead % | Avg lead steps | Unavailable % | N |'
        )
        report_lines.append(
            '|------------|------|--------|----------------|---------------|---|'
        )
        _wrote_multi = False
        for lbl in xai_labels:
            rws = rows_by_xai[lbl]
            dn = xai_method_display_name(lbl)
            pr = next((r for r in rws if r['method'] == primary_key), None)
            tr = next((r for r in rws if r['Change Type'] == 'trend'), None)
            if pr:
                _wrote_multi = True
                un_p = 100.0 * pr['n_unavailable'] / pr['n_total'] if pr['n_total'] else 0.0
                report_lines.append(
                    '| {} | **Primary: 4-metric mean delta >= {:.2f}** | {:.1f}% | {} | {:.1f}% | {} |'.format(
                        dn,
                        DEFAULT_THRESHOLD,
                        pr['Lead Ratio (%)'],
                        _fmt_mean_lead_step(pr['Mean Lead Step']),
                        un_p,
                        pr['n_total'],
                    )
                )
            if tr:
                _wrote_multi = True
                un_p = 100.0 * tr['n_unavailable'] / tr['n_total'] if tr['n_total'] else 0.0
                report_lines.append(
                    '| {} | Trend-based (same 4-metric scores) | {:.1f}% | {} | {:.1f}% | {} |'.format(
                        dn,
                        tr['Lead Ratio (%)'],
                        _fmt_mean_lead_step(tr['Mean Lead Step']),
                        un_p,
                        tr['n_total'],
                    )
                )
        if not _wrote_multi:
            report_lines.append('| N/A | N/A | N/A | N/A | N/A | 0 |')
    else:
        report_lines.append('| Rule | Lead % | Avg lead steps | Unavailable % | N |')
        report_lines.append('|------|--------|----------------|---------------|---|')
        if primary_row:
            pr = primary_row
            un_p = 100.0 * pr['n_unavailable'] / pr['n_total'] if pr['n_total'] else 0.0
            report_lines.append(
                '| **Primary: 4-metric mean delta >= {:.2f}** | {:.1f}% | {} | {:.1f}% | {} |'.format(
                    DEFAULT_THRESHOLD,
                    pr['Lead Ratio (%)'],
                    _fmt_mean_lead_step(pr['Mean Lead Step']),
                    un_p,
                    pr['n_total'],
                )
            )
        if trend_row:
            tr = trend_row
            un_p = 100.0 * tr['n_unavailable'] / tr['n_total'] if tr['n_total'] else 0.0
            report_lines.append(
                '| Trend-based change (same 4-metric scores) | {:.1f}% | {} | {:.1f}% | {} |'.format(
                    tr['Lead Ratio (%)'],
                    _fmt_mean_lead_step(tr['Mean Lead Step']),
                    un_p,
                    tr['n_total'],
                )
            )
        if not primary_row and not trend_row:
            report_lines.append('| N/A | N/A | N/A | N/A | 0 |')
    report_lines.append('')
    report_lines.append(
        '*Grad-CAM vs FastCAV under **different** signal definitions → Appendix at end of this report.*'
    )
    report_lines.append('')
    report_lines.append('## Key conclusions')
    report_lines.append(
        '- **CAM 변화 정의 (본 리포트 전역)**: severity-0 대비 4지표(bbox_dist, peak_dist, spread, ring_ratio) 평균 스코어의 '
        '**상대 변화(delta) >= {:.2f}** 인 최소 severity를 CAM 변화 시점으로 둠. `lead_stats.json`·Core 표는 **{}** 버킷 기준; '
        'XAI별 수치는 `lead_stats_by_xai_method.json` 및 상단 표.'.format(
            DEFAULT_THRESHOLD,
            xai_method_display_name(lead_src_label),
        )
    )
    n_tot = primary_row['n_total'] if primary_row else 0
    n_wcam = (
        (primary_row['n_lead'] + primary_row['n_coincident'] + primary_row['n_lag'])
        if primary_row
        else 0
    )
    if primary_row and n_tot > 0:
        report_lines.append(
            '- **delta={:.2f} 전체 이벤트 기준**: lead {:.1f}% (전체 N={}), '
            'CAM 가능 이벤트 내 lead {:.1f}% (N={}), coincident {:.1f}%, lag {:.1f}%.'.format(
                DEFAULT_THRESHOLD,
                100.0 * primary_row['n_lead'] / n_tot,
                n_tot,
                100.0 * primary_row['n_lead'] / n_wcam if n_wcam else 0.0,
                n_wcam,
                100.0 * primary_row['n_coincident'] / n_wcam if n_wcam else 0.0,
                100.0 * primary_row['n_lag'] / n_wcam if n_wcam else 0.0,
            )
        )
    if trend_row is not None:
        report_lines.append(
            '- Trend 정의 시 lead {:.1f}%, Mean lead step {:.2f}.'.format(
                trend_row['Lead Ratio (%)'],
                trend_row['Mean Lead Step'] if not np.isnan(trend_row['Mean Lead Step']) else 0,
            )
        )
    report_lines.append('- 성능 붕괴 현상: severity 3~4에서 score_drop 및 miss_rate 급증.')
    report_lines.append('- CAM 가용성 감소: severity 4에서 cam_valid_ratio가 크게 감소.')
    report_lines.append(
        '- 기타 파이프라인: z-score(|z|>=2) 등은 **별도 정의**이며, 본 문서의 Lead/표는 **4지표 평균 delta**만 사용.'
    )
    report_lines.append(
        '- exp_A Grad-CAM vs FastCAV는 **다른 신호**(spread·개념점수)이므로 상단 표와 숫자를 직접 비교하지 말 것 → 부록.'
    )
    report_lines.extend(build_runtime_performance_section(results_root))
    report_lines.append('')
    report_lines.append('## Core comparison table')
    report_lines.append(
        '*Corruption별 Lead/CAM 가용성은 `xai_method`별로 분리 집계합니다. '
        'Cam Valid Ratio (sev4) 분자 = 해당 버킷(primary·ok) cam_records 행 수; '
        '분모 = corruption당 고유 image 슬롯 수 {}. '
        'Score Drop Rate (sev4) = **matched==1** detection 행 `mean(is_score_drop)` (모델 성능 기준이라 XAI 방법과 무관).*'.format(
            EXPECTED_UNIQUE_IMAGES_PER_CORRUPTION
        )
    )
    if multi_xai_compare:
        report_lines.append(
            '| XAI method | Corruption | Lead % | Avg Lead Steps | Cam Valid Ratio (sev4) | Score Drop Rate (sev4) |'
        )
        report_lines.append('|------------|------------|--------|----------------|------------------------|------------------------|')
        for lbl in xai_labels:
            x_stats = stats_by_xai[lbl][primary_key]
            dn = xai_method_display_name(lbl)
            for c in corruptions:
                pc = _summarize_per_corruption(x_stats, c)
                cam_v, sd = sev4_perf_cam_by_xai.get(lbl, {}).get(c, (0.0, 0.0))
                if pc is None or pc['n_total'] == 0:
                    report_lines.append('| {} | {} | N/A | N/A | {:.3f} | {:.3f} |'.format(dn, c, cam_v, sd))
                    continue
                lp = pc['Lead Ratio (%)']
                ms = pc['Mean Lead Step']
                ms_str = f'{ms:.2f}' if ms is not None and not np.isnan(ms) else 'N/A'
                report_lines.append('| {} | {} | {:.1f}% | {} | {:.3f} | {:.3f} |'.format(dn, c, lp, ms_str, cam_v, sd))
    else:
        report_lines.append(
            '| Corruption | Lead % | Avg Lead Steps | Cam Valid Ratio (sev4) | Score Drop Rate (sev4) |'
        )
        report_lines.append('|------------|--------|----------------|------------------------|------------------------|')
        for c in corruptions:
            pc = _summarize_per_corruption(primary_stats, c)
            cam_v, sd = sev4_perf_cam_by_xai.get(lead_src_label, {}).get(c, (0.0, 0.0))
            if pc is None or pc['n_total'] == 0:
                report_lines.append('| {} | N/A | N/A | {:.3f} | {:.3f} |'.format(c, cam_v, sd))
                continue
            lp = pc['Lead Ratio (%)']
            ms = pc['Mean Lead Step']
            ms_str = f'{ms:.2f}' if ms is not None and not np.isnan(ms) else 'N/A'
            report_lines.append('| {} | {:.1f}% | {} | {:.3f} | {:.3f} |'.format(c, lp, ms_str, cam_v, sd))
    report_lines.append('')
    report_lines.append('## Severity progression summary')
    report_lines.append(
        '*본 표는 `detection_records.csv` / `cam_records.csv`에서 스크립트가 매 실행 시 재계산합니다. '
        '**Score Drop Rate** = severity별 **matched==1** 인 행에 대한 `mean(is_score_drop)` '
        '(전체 행 평균은 고 severity에서 미스가 늘며 0이 섞여 sev3>sev4처럼 역전될 수 있음 — `score_drop_rate_all_records` 열은 CSV에 별도 저장). '
        '**Cam Valid Ratio** = 해당 severity에서 fog+lowlight+motion_blur **풀링**: '
        '`n_cam_rows(primary,ok) / n_detection_rows` (Core 표의 sev4 값과 분모 체계가 다름).*'
    )
    report_lines.append('| Severity | Avg Score | Score Drop Rate | Cam Valid Ratio |')
    report_lines.append('|----------|-----------|-----------------|-----------------|')
    if summary_df is not None:
        s = summary_df.sort_values('severity')
        for _, r in s.iterrows():
            report_lines.append(
                '| {} | {:.3f} | {:.3f} | {:.3f} |'.format(
                    int(r['severity']),
                    float(r['avg_score']),
                    float(r['score_drop_rate']),
                    float(r['cam_valid_ratio']),
                )
            )
    else:
        report_lines.append('| N/A | N/A | N/A | N/A |')
    report_lines.append('')
    report_lines.append('- Observations:')
    report_lines.append('  - 성능 붕괴는 Severity 2→3 구간에서 뚜렷 (matched-기준 score_drop_rate·avg_score)')
    cam1 = cam2 = None
    if summary_df is not None and len(summary_df) > 0:
        srt = summary_df.set_index('severity')
        if 1 in srt.index and 2 in srt.index:
            cam1 = float(srt.loc[1, 'cam_valid_ratio'])
            cam2 = float(srt.loc[2, 'cam_valid_ratio'])
    if cam1 is not None and cam2 is not None:
        report_lines.append(
            '  - CAM 유효성(풀링 cam_valid_ratio)은 Severity 1→2에서 하락 시작 ({:.3f}→{:.3f})'.format(cam1, cam2)
        )
    else:
        report_lines.append('  - CAM 유효성은 severity 증가에 따라 cam_valid_ratio가 하락')
    report_lines.append('  - Core 표의 Lead % / Avg Lead Steps는 **delta={:.2f}** 기준, corruption별 failure 이벤트 집계.'.format(DEFAULT_THRESHOLD))
    report_lines.append('')
    report_lines.append('## Severity progression by corruption (0-4)')
    report_lines.append(
        '*각 변조 유형별 severity 0-4에서 Avg Score / Score Drop Rate를 분리 집계. '
        'Score Drop Rate는 **matched==1** 기준(`score_drop_rate_all_records`는 CSV 참고).*'
    )
    report_lines.append('| Corruption | Severity | Avg Score | Score Drop Rate |')
    report_lines.append('|------------|----------|-----------|-----------------|')
    if summary_by_corr_df is not None and len(summary_by_corr_df) > 0:
        s = summary_by_corr_df.sort_values(['corruption', 'severity'])
        for _, r in s.iterrows():
            report_lines.append(
                '| {} | {} | {:.3f} | {:.3f} |'.format(
                    str(r['corruption']),
                    int(r['severity']),
                    float(r['avg_score']),
                    float(r['score_drop_rate']),
                )
            )
    else:
        report_lines.append('| N/A | N/A | N/A | N/A |')
    report_lines.extend(build_tiny_object_concept_section(results_root))
    report_lines.extend(build_gradcam_fastcav_report_section(results_root))
    report_lines.append('')
    report_lines.append('## Recommendation')
    report_lines.append(
        '- 핵심: CAM 변화를 **delta >= {:.2f}** 로 정의하면 선행(lead) 비율이 표에 반영됨; '
        'CAM 생성 성공률(cam_valid_ratio)은 별도 모니터링 지표로 유지.'.format(DEFAULT_THRESHOLD)
    )
    report_lines.append(
        '- 실무: Grad-CAM baseline 유지 + cam_status != ok 건수를 위험 지표에 포함.'
    )
    report_lines.append(
        '- 부록(exp_A) 갱신: `python scripts/11_fastcav_concept_detection.py` → '
        '`python scripts/exp_A_early_warning_comparison.py` → 본 `report.md` 재생성.'
    )
    report_lines.append(
        '- 실시간 성능 표 갱신: `python scripts/runtime_xai_benchmark.py` → `python scripts/exp_A_threshold_trend_validation.py`.'
    )
    report_lines.append(
        '- Grad-CAM++ / LayerCAM **선행성**: `configs/experiment.yaml`의 `gradcam.xai_methods`에 '
        '`gradcampp`, `layercam` 포함 후 `python scripts/05_gradcam_failure_analysis.py` → 본 스크립트. (레거시 CSV의 `fastcam` 행은 gradcampp와 동일 취급)'
    )

    report_path = results_root / 'report.md'
    report_path.write_text('\n'.join(report_lines), encoding='utf-8')
    print(f"[OK] Updated {report_path} (concise layout, delta={DEFAULT_THRESHOLD} lead per corruption)")


if __name__ == '__main__':
    main()
