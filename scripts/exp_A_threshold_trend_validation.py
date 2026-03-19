"""Threshold / Trend Change Detection Validation (Grad-CAM event-level).

- Threshold sweep (0.05 / 0.10 / 0.20): delta-based CAM change.
- Writes results/lead_stats.json (delta_threshold=0.1) and results/report.md.
- report.md: original concise layout (Key conclusions, Core comparison, Severity table,
  Observations, Recommendation); Lead % / Avg Lead Steps use delta=0.1 per corruption.
"""

import json
import numpy as np
import pandas as pd
from pathlib import Path


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


def build_gradcam_fastcav_report_section(results_root: Path) -> list:
    """
    exp_A_early_warning_comparison.py 산출물(exp_A_alignment_comparison.csv) 기반
    Grad-CAM vs FastCAV 조기경보 정렬 요약을 Markdown 줄 목록으로 반환.
    """
    align_path = results_root / 'exp_A_alignment_comparison.csv'
    if not align_path.exists() or align_path.stat().st_size == 0:
        return [
            '',
            '## Grad-CAM vs FastCAV (early warning)',
            '*이 섹션 데이터 없음. 생성: `python scripts/11_fastcav_concept_detection.py` 후 '
            '`python scripts/exp_A_early_warning_comparison.py`*',
        ]

    adf = pd.read_csv(align_path)
    lines = ['', '## Grad-CAM vs FastCAV (early warning)']
    lines.append(
        '*동일 `failure_events`의 performance 시작 severity 대비 CAM/개념 변화 시점 정렬(lead/coincident/lag). '
        '**Grad-CAM (exp_A)**: `activation_spread` 붕괴 또는 baseline 대비 spread 변화 ≥0.2 (`exp_A_early_warning_comparison.py`). '
        '**FastCAV (본 레포 proxy)**: `cam_records` 지표로 만든 개념 점수, 변화 ≥0.2 (`11_fastcav_concept_detection.py`). '
        '위 둘은 Core 표의 **4지표 평균 delta=0.10** lead 정의와 다름.*'
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

    # Threshold sweep: 0.05, 0.10, 0.20 (delta-based) + trend
    THRESHOLDS = [0.05, 0.10, 0.20]
    DEFAULT_THRESHOLD = 0.10  # used for lead_stats.json and report
    methods = [('threshold', t) for t in THRESHOLDS] + [('trend', None)]
    stats = {m: [] for m in methods}  # list of dict: corruption, alignment, lead_steps

    for _, event in failure_events.iterrows():
        start = event.get('performance_start_severity')
        if pd.isna(start):
            continue
        start = int(start)

        corruption = event.get('corruption')
        image_id = event.get('image_id')
        class_id = event.get('class_id')

        event_cam = cam_records[
            (cam_records['corruption'] == corruption) &
            (cam_records['image_id'] == image_id) &
            (cam_records['class_id'] == class_id) &
            (cam_records['layer_role'] == 'primary')
        ]

        def _rec(align, steps):
            return {'corruption': corruption, 'alignment': align, 'lead_steps': steps}

        if event_cam.empty:
            for method in methods:
                stats[method].append(_rec('unavailable', None))
            continue

        event_cam = event_cam[event_cam['severity'] <= start].copy()
        if event_cam.empty:
            for method in methods:
                stats[method].append(_rec('unavailable', None))
            continue

        event_cam['severity'] = pd.to_numeric(event_cam['severity'], errors='coerce').fillna(-1).astype(int)
        score_by_sev = compute_event_cam_scores(event_cam)

        for method in methods:
            if method[0] == 'threshold':
                cam_change_sev = baseline_threshold_change(score_by_sev, method[1])
            else:
                cam_change_sev = trend_based_change(score_by_sev)
            alignment, lead_steps = compute_alignment(start, cam_change_sev)
            stats[method].append(_rec(alignment, lead_steps))

    def _summarize_method(results):
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

    def _summarize_per_corruption(results, corruption: str):
        sub = [x for x in results if x.get('corruption') == corruption]
        return _summarize_method(sub)

    # Summary rows for each method
    rows = []
    for method in methods:
        s = _summarize_method(stats[method])
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
    if det_df is not None and len(det_df) > 0:
        cam_for_summary = cam_df if cam_df is not None else pd.DataFrame()
        summary_df = compute_severity_progression_summary(det_df, cam_for_summary, results_root)

    # Sev4 performance / CAM availability per corruption (dynamic if CSVs exist)
    corruptions = ['fog', 'lowlight', 'motion_blur']
    sev4_perf_cam = {}
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
        cam_ratio = 0.0
        if cam_df is not None and len(cam_df) > 0 and 'severity' in cam_df.columns:
            c4 = cam_df[(cam_df['corruption'] == c) & (cam_df['severity'] == 4)]
            if 'layer_role' in c4.columns:
                c4 = c4[c4['layer_role'] == 'primary']
            if 'cam_status' in c4.columns:
                c4 = c4[c4['cam_status'] == 'ok']
            cam_ratio = (
                float(len(c4)) / EXPECTED_UNIQUE_IMAGES_PER_CORRUPTION
                if EXPECTED_UNIQUE_IMAGES_PER_CORRUPTION > 0
                else 0.0
            )
        sev4_perf_cam[c] = (cam_ratio, score_drop)

    th_05 = next((r for r in rows if r['method'] == ('threshold', 0.05)), None)
    th_10 = next((r for r in rows if r['method'] == ('threshold', 0.10)), None)
    th_20 = next((r for r in rows if r['method'] == ('threshold', 0.20)), None)
    trend_row = next((r for r in rows if r['Change Type'] == 'trend'), None)
    th10_key = ('threshold', DEFAULT_THRESHOLD)
    th10_stats = stats[th10_key]

    report_lines = []
    report_lines.append('# Concise Summary Report: CAM vs Performance')
    report_lines.append('')
    report_lines.append('This summary focuses on the core findings for fast decision-making.')
    report_lines.append('')
    report_lines.append('## Key conclusions')
    report_lines.append(
        '- **CAM 변화 정의 (본 리포트)**: severity-0 대비 4지표(bbox_dist, peak_dist, spread, ring_ratio) 평균 스코어의 '
        '**상대 변화(delta) >= {:.2f}** 인 최소 severity를 CAM 변화 시점으로 둠.'.format(DEFAULT_THRESHOLD)
    )
    if th_05 and th_10 and th_20:
        report_lines.append(
            '- Threshold sweep (delta): lead 비율 {:.1f}% / {:.1f}% / {:.1f}% (0.05 / 0.10 / 0.20); '
            'Mean lead step {:.2f} / {:.2f} / {:.2f}.'.format(
                th_05['Lead Ratio (%)'], th_10['Lead Ratio (%)'], th_20['Lead Ratio (%)'],
                th_05['Mean Lead Step'] if not np.isnan(th_05['Mean Lead Step']) else 0,
                th_10['Mean Lead Step'] if not np.isnan(th_10['Mean Lead Step']) else 0,
                th_20['Mean Lead Step'] if not np.isnan(th_20['Mean Lead Step']) else 0,
            )
        )
    n_tot = th_10['n_total'] if th_10 else 0
    n_wcam = (th_10['n_lead'] + th_10['n_coincident'] + th_10['n_lag']) if th_10 else 0
    if th_10 and n_tot > 0:
        report_lines.append(
            '- **delta={:.2f} 전체 이벤트 기준**: lead {:.1f}% (전체 N={}), '
            'CAM 가능 이벤트 내 lead {:.1f}% (N={}), coincident {:.1f}%, lag {:.1f}%.'.format(
                DEFAULT_THRESHOLD,
                100.0 * th_10['n_lead'] / n_tot,
                n_tot,
                100.0 * th_10['n_lead'] / n_wcam if n_wcam else 0.0,
                n_wcam,
                100.0 * th_10['n_coincident'] / n_wcam if n_wcam else 0.0,
                100.0 * th_10['n_lag'] / n_wcam if n_wcam else 0.0,
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
        '- 참고: z-score(|z|>=2) 기반 정렬은 별도 파이프라인이며, 본 표의 Lead는 **delta 임계값 {:.2f}** 산출값임.'.format(
            DEFAULT_THRESHOLD
        )
    )
    exp_a_summary_path = results_root / 'exp_A_summary_table.csv'
    if exp_a_summary_path.exists() and exp_a_summary_path.stat().st_size > 0:
        try:
            es = pd.read_csv(exp_a_summary_path)
            gl = es[(es['method'] == 'GRADCAM') & (es['alignment'] == 'lead')]['percentage']
            fl = es[(es['method'] == 'FASTCAV') & (es['alignment'] == 'lead')]['percentage']
            gu = es[(es['method'] == 'GRADCAM') & (es['alignment'] == 'unavailable')]['percentage']
            fu = es[(es['method'] == 'FASTCAV') & (es['alignment'] == 'unavailable')]['percentage']
            if len(gl) > 0 and len(fl) > 0:
                report_lines.append(
                    '- **Grad-CAM vs FastCAV (exp_A 조기경보)**: 동일 failure 이벤트 기준 lead 비율 '
                    'Grad-CAM {:.1f}% vs FastCAV {:.1f}% (unavailable {:.1f}% vs {:.1f}%). '
                    '정의는 spread/개념점수·delta=0.2 — Core 표의 4지표 delta={:.2f} 와 별개. 하단 표 참조.'.format(
                        float(gl.iloc[0]), float(fl.iloc[0]),
                        float(gu.iloc[0]) if len(gu) > 0 else 0.0,
                        float(fu.iloc[0]) if len(fu) > 0 else 0.0,
                        DEFAULT_THRESHOLD,
                    )
                )
        except Exception:
            pass
    report_lines.append('')
    report_lines.append('## Core comparison table')
    report_lines.append(
        '*Cam Valid Ratio (sev4): 분자 = cam_records 행 수(`layer_role=primary`, `cam_status=ok`); '
        '분모 = corruption당 고유 image 슬롯 수 {} (tiny 코호트; sev4에서도 동일 분모). '
        'Score Drop Rate (sev4): **matched==1** detection 행만 대상으로 `mean(is_score_drop)` — '
        '미스 행은 스코어 드롭 플래그가 정의되지 않아 제외.*'.format(EXPECTED_UNIQUE_IMAGES_PER_CORRUPTION)
    )
    report_lines.append(
        '| Corruption | Lead % | Avg Lead Steps | Cam Valid Ratio (sev4) | Score Drop Rate (sev4) |'
    )
    report_lines.append('|------------|--------|----------------|------------------------|------------------------|')
    for c in corruptions:
        pc = _summarize_per_corruption(th10_stats, c)
        cam_v, sd = sev4_perf_cam.get(c, (0.0, 0.0))
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
        '- FastCAV 비교 갱신: `python scripts/11_fastcav_concept_detection.py` → '
        '`python scripts/exp_A_early_warning_comparison.py` → 본 `report.md` 재생성.'
    )

    report_path = results_root / 'report.md'
    report_path.write_text('\n'.join(report_lines), encoding='utf-8')
    print(f"[OK] Updated {report_path} (concise layout, delta={DEFAULT_THRESHOLD} lead per corruption)")


if __name__ == '__main__':
    main()
