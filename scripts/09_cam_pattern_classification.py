"""
CAM 시계열 패턴 분류: persistent collapse vs transient instability vs stable vs oscillatory.

- cam_records.csv를 object_id × corruption 단위로 severity 0~4 시계열 분석
- 1차 규칙: breakdown = E_bbox <= 0.05 or spread <= 1e-6 or cam_quality != 'high'
- 패턴: stable, transient_instability, persistent_collapse, oscillatory
- 출력: pattern_summary.csv, pattern_counts.json, *_cases.csv, 요약 보고서
- 선택: transient 제외 시 lead 통계 비교 (lead_table.csv 있으면)
"""

import sys
import json
from pathlib import Path
import pandas as pd
import numpy as np

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.utils.io import load_yaml


# 1차 규칙 (엄격): 단일 지표만으로 붕괴 판정 — transient 과다 원인
E_BBOX_BREAKDOWN_TH = 0.05
SPREAD_BREAKDOWN_TH = 1e-6
CAM_QUALITY_GOOD = "high"

# 완화 규칙 (권장): E_bbox_1.25x + 2개 이상 지표 결합 + 공식 붕괴는 2단계 연속 또는 이후 미회복
E_BBOX_1_25X_ANOMALY_TH = 0.2   # E_bbox_1.25x <= 0.2 → 이상 신호 후보
SPREAD_ANOMALY_TH = 1e-5        # spread <= 1e-5 (0 비교는 수치적으로 너무 날카로움)
CAM_SUM_ANOMALY_TH = 1e-8       # cam_sum <= 1e-8
ENTROPY_RATIO_VS_L0 = 1.3       # entropy >= 1.3 * entropy_L0 → 이상 신호
BBOX_DIST_RATIO_VS_L0 = 2.0     # bbox_dist >= 2 * bbox_dist_L0 → 이상 신호
MIN_ANOMALY_INDICATORS = 2      # 이상 신호로 인정하려면 2개 이상 지표 동시 성립

# 회복 판정 (transient: 중간에 깨졌다가 다시 회복)
E_BBOX_RECOVERY_TH = 0.5
SPREAD_RECOVERY_TH = 0.0  # spread > 0


def _row_float(r, key, default=np.nan):
    """row (Series or dict)에서 key에 해당하는 float 추출."""
    try:
        x = r.get(key, default) if hasattr(r, "get") else default
    except Exception:
        x = default
    if x is None or (isinstance(x, float) and np.isnan(x)):
        return default
    try:
        return float(x)
    except (TypeError, ValueError):
        return default


def is_breakdown(row: pd.Series, e_bbox_key: str = "energy_in_bbox") -> bool:
    """한 severity가 breakdown인지 (엄격 규칙: 단일 지표만으로 판정)."""
    e_bbox = row.get(e_bbox_key)
    if pd.notna(e_bbox) and float(e_bbox) <= E_BBOX_BREAKDOWN_TH:
        return True
    spread = row.get("activation_spread")
    if pd.notna(spread) and float(spread) <= SPREAD_BREAKDOWN_TH:
        return True
    cq = row.get("cam_quality")
    if pd.notna(cq) and str(cq).strip().lower() != CAM_QUALITY_GOOD:
        return True
    return False


def get_relaxed_breakdowns(sub, e_bbox_key: str) -> np.ndarray:
    """
    완화 규칙: 설명 실패(flat/spread=0 등) 단독을 붕괴로 보지 않고,
    이상 신호 2개 이상 + (2단계 연속 또는 이후 미회복)일 때만 공식 붕괴.
    sub: DataFrame (severity 0..4) 또는 list of 5 row-like dicts.
    Returns: length-5 array, 1=공식 breakdown, 0=아님.
    """
    if isinstance(sub, pd.DataFrame):
        rows = [sub[sub["severity"] == i].iloc[0] for i in range(5)]
    else:
        rows = list(sub)[:5]
    row0 = rows[0]
    e0 = _row_float(row0, e_bbox_key) or _row_float(row0, "energy_in_bbox")
    ent0 = _row_float(row0, "entropy", default=0)
    bd0 = _row_float(row0, "bbox_center_activation_distance", default=0)
    if bd0 == 0:
        bd0 = 1e-9  # avoid div by zero

    anomaly = []
    for i in range(5):
        r = rows[i]
        e = _row_float(r, e_bbox_key) or _row_float(r, "energy_in_bbox")
        cam_sum = _row_float(r, "cam_sum", default=np.nan)
        spread = _row_float(r, "activation_spread", default=np.nan)
        cq = str(r.get("cam_quality", "") or "").strip().lower()
        ent = _row_float(r, "entropy", default=np.nan)
        bd = _row_float(r, "bbox_center_activation_distance", default=np.nan)

        count = 0
        if pd.notna(e) and e <= E_BBOX_1_25X_ANOMALY_TH:
            count += 1
        if cq == "flat":
            count += 1
        if pd.notna(cam_sum) and cam_sum <= CAM_SUM_ANOMALY_TH:
            count += 1
        if pd.notna(spread) and spread <= SPREAD_ANOMALY_TH:
            count += 1
        if pd.notna(ent) and pd.notna(ent0) and ent0 > 0 and ent >= ENTROPY_RATIO_VS_L0 * ent0:
            count += 1
        if pd.notna(bd) and bd >= BBOX_DIST_RATIO_VS_L0 * bd0:
            count += 1

        anomaly.append(count >= MIN_ANOMALY_INDICATORS)

    # 공식 붕괴: 이상 신호가 2단계 연속이거나, 이상 신호 발생 후 이후 단계에서 회복 없음
    breakdowns = np.zeros(5, dtype=int)
    for i in range(5):
        if not anomaly[i]:
            continue
        two_consecutive = (i > 0 and anomaly[i - 1]) or (i < 4 and anomaly[i + 1])
        no_recovery_after = all(anomaly[j] for j in range(i + 1, 5))
        if two_consecutive or no_recovery_after:
            breakdowns[i] = 1
    return breakdowns


def is_recovered(row: pd.Series) -> bool:
    """해당 severity에서 정상 복귀로 볼 수 있는지."""
    e_bbox = row.get("energy_in_bbox")
    if pd.isna(e_bbox):
        return False
    if float(e_bbox) <= E_BBOX_RECOVERY_TH:
        return False
    spread = row.get("activation_spread")
    if pd.notna(spread) and float(spread) <= SPREAD_RECOVERY_TH:
        return False
    return True


def classify_pattern(breakdowns: np.ndarray) -> str:
    """
    breakdowns: length-5 array, 1=breakdown at that severity, 0=ok
    Returns: 'stable' | 'transient_instability' | 'persistent_collapse' | 'oscillatory'
    """
    if breakdowns is None or len(breakdowns) != 5:
        return "unknown"
    b = np.asarray(breakdowns, dtype=int)
    n_break = int(b.sum())
    if n_break == 0:
        return "stable"
    first_break = int(np.where(b == 1)[0][0])
    last_break = int(np.where(b == 1)[0][-1])
    # Persistent: 첫 breakdown 이후 끝까지 회복 없음 (뒤가 전부 1)
    if np.all(b[first_break:] == 1):
        return "persistent_collapse"
    # Transient: severity 1~3에서 시작하는 한 덩어리 breakdown + 이후 전부 회복(0)
    one_block = np.all(b[first_break : last_break + 1] == 1) and np.all(b[last_break + 1 :] == 0)
    if (1 <= first_break <= 3) and one_block:
        return "transient_instability"
    # Oscillatory: 1과 0이 두 번 이상 번갈아 (깨짐-회복-재붕괴)
    changes = np.diff(b)
    if np.sum(np.abs(changes)) >= 2:
        return "oscillatory"
    return "persistent_collapse"


def _smooth_series(vals: list) -> list:
    """3-point moving average. Boundary: L0=(L0+L1)/2, L4=(L3+L4)/2, else (L_{i-1}+L_i+L_{i+1})/3."""
    n = len(vals)
    out = []
    for i in range(n):
        if i == 0:
            out.append((vals[0] + vals[1]) / 2.0 if n > 1 else vals[0])
        elif i == n - 1:
            out.append((vals[n - 2] + vals[n - 1]) / 2.0 if n > 1 else vals[n - 1])
        else:
            out.append((vals[i - 1] + vals[i] + vals[i + 1]) / 3.0)
    return out


def _interpolate_invalid_series(
    sub: pd.DataFrame,
    e_bbox_key: str,
    cam_valid_col: str = "cam_valid",
) -> list:
    """
    For each severity 0..4, if cam_valid is False, replace E_bbox/spread/entropy/bbox_dist
    with linear interpolation from valid neighbors. Returns list of 5 dicts (row-like for is_breakdown).
    """
    rows = [sub[sub["severity"] == i].iloc[0] for i in range(5)]
    valid = []
    for i in range(5):
        v = rows[i].get(cam_valid_col)
        valid.append(
            v is True
            or (isinstance(v, str) and v.strip().lower() == "true")
            or (v is not None and not pd.isna(v) and bool(v))
        )

    def _get(r, k):
        x = r.get(k)
        if x is None or pd.isna(x):
            return np.nan
        try:
            return float(x)
        except (TypeError, ValueError):
            return np.nan

    e_arr = [_get(rows[i], e_bbox_key) or _get(rows[i], "energy_in_bbox") for i in range(5)]
    sp_arr = [_get(rows[i], "activation_spread") for i in range(5)]
    ent_arr = [_get(rows[i], "entropy") for i in range(5)]
    bd_arr = [_get(rows[i], "bbox_center_activation_distance") for i in range(5)]

    def _interp(arr, valid_mask):
        arr = np.asarray(arr, dtype=float)
        out = arr.copy()
        for i in range(5):
            if valid_mask[i]:
                continue
            prev_i, next_i = None, None
            for j in range(i - 1, -1, -1):
                if valid_mask[j] and not np.isnan(arr[j]):
                    prev_i = j
                    break
            for j in range(i + 1, 5):
                if valid_mask[j] and not np.isnan(arr[j]):
                    next_i = j
                    break
            if prev_i is not None and next_i is not None:
                prev_v, next_v = arr[prev_i], arr[next_i]
                out[i] = prev_v + (next_v - prev_v) * (i - prev_i) / (next_i - prev_i)
            elif prev_i is not None:
                out[i] = arr[prev_i]
            elif next_i is not None:
                out[i] = arr[next_i]
        return out.tolist()

    if cam_valid_col not in sub.columns:
        return [
            {
                e_bbox_key: _get(rows[i], e_bbox_key) or _get(rows[i], "energy_in_bbox"),
                "energy_in_bbox": _get(rows[i], "energy_in_bbox"),
                "activation_spread": _get(rows[i], "activation_spread"),
                "entropy": _get(rows[i], "entropy"),
                "bbox_center_activation_distance": _get(rows[i], "bbox_center_activation_distance"),
                "cam_quality": rows[i].get("cam_quality"),
            }
            for i in range(5)
        ]

    e_rep = _interp(e_arr, valid)
    sp_rep = _interp(sp_arr, valid)
    ent_rep = _interp(ent_arr, valid)
    bd_rep = _interp(bd_arr, valid)
    return [
        {
            e_bbox_key: e_rep[i],
            "energy_in_bbox": e_rep[i],
            "activation_spread": sp_rep[i],
            "entropy": ent_rep[i],
            "bbox_center_activation_distance": bd_rep[i],
            "cam_quality": rows[i].get("cam_quality"),
        }
        for i in range(5)
    ]


def main():
    config_path = Path("configs/experiment.yaml")
    if config_path.exists():
        config = load_yaml(config_path)
        results_dir = Path(config.get("results", {}).get("root", "results"))
    else:
        config = {}
        results_dir = Path("results")

    cam_csv = results_dir / "cam_records.csv"
    if not cam_csv.exists() or cam_csv.stat().st_size == 0:
        print("Error: cam_records.csv not found or empty. Run 05_gradcam_failure_analysis.py first.")
        sys.exit(1)

    print("=" * 60)
    print("CAM 시계열 패턴 분류 (Persistent / Transient / Stable / Oscillatory)")
    print("=" * 60)

    df = pd.read_csv(cam_csv)
    if "layer_role" in df.columns:
        df = df[df["layer_role"] == "primary"].copy()
    if "cam_status" in df.columns:
        df = df[df["cam_status"] == "ok"].copy()

    # Explanation quality gate: only analyze CAMs that pass cam_valid (cam_sum >= 1e-8, spread > 0)
    use_quality_gate = config.get("pattern_classification", {}).get("use_quality_gate", False)
    if use_quality_gate and "cam_valid" in df.columns:
        valid = df["cam_valid"]
        mask = (valid == True) | (valid.astype(str).str.lower() == "true")
        df = df[mask].copy()
        print("Quality gate on: only cam_valid==True rows included.")

    group_cols = ["model", "corruption", "object_id"]
    for c in group_cols:
        if c not in df.columns:
            print(f"Missing column: {c}")
            sys.exit(1)

    # severity 0~4가 모두 있는 (model, corruption, object_id) 만
    need_sev = set(range(5))
    complete = (
        df.groupby(group_cols)["severity"]
        .apply(lambda s: need_sev.issubset(set(s.astype(int))))
        .reset_index()
    )
    complete = complete[complete["severity"]]
    keys = complete[group_cols].drop_duplicates()

    # 기본: E_bbox_1.25x 사용 (ROI miss 감소). 컬럼 없으면 energy_in_bbox 사용
    e_bbox_key = "energy_in_bbox_1_25x" if "energy_in_bbox_1_25x" in df.columns else "energy_in_bbox"
    use_relaxed_breakdown = config.get("pattern_classification", {}).get("use_relaxed_breakdown", True)
    interpolate_invalid = config.get("pattern_classification", {}).get("interpolate_invalid", False)
    use_smoothing = config.get("pattern_classification", {}).get("use_smoothing", False)

    if use_relaxed_breakdown:
        print("Using relaxed breakdown rule: E_bbox_1.25x<=0.2 + 2+ indicators, official breakdown = 2 consecutive or no recovery after.")

    rows = []
    for _, r in keys.iterrows():
        sub = df[
            (df["model"] == r["model"])
            & (df["corruption"] == r["corruption"])
            & (df["object_id"] == r["object_id"])
        ].sort_values("severity")
        if len(sub) < 5:
            continue
        if use_relaxed_breakdown:
            breakdowns = get_relaxed_breakdowns(sub, e_bbox_key).tolist()
        else:
            breakdowns = []
            for sev in range(5):
                row_sev = sub[sub["severity"] == sev]
                if len(row_sev) == 0:
                    breakdowns.append(1)
                else:
                    breakdowns.append(1 if is_breakdown(row_sev.iloc[0], e_bbox_key=e_bbox_key) else 0)
        pattern = classify_pattern(np.array(breakdowns))
        image_id = sub["image_id"].iloc[0] if "image_id" in sub.columns else ""

        row_dict = {
            "model": r["model"],
            "corruption": r["corruption"],
            "object_id": r["object_id"],
            "image_id": image_id,
            "pattern": pattern,
            "breakdown_L0": breakdowns[0],
            "breakdown_L1": breakdowns[1],
            "breakdown_L2": breakdowns[2],
            "breakdown_L3": breakdowns[3],
            "breakdown_L4": breakdowns[4],
        }

        # 데이터 균형: 측정 실패 구간 보간 후 재분류 / 스무딩 후 재분류 (동일 breakdown 규칙 사용)
        repaired_rows = None
        if interpolate_invalid:
            repaired_rows = _interpolate_invalid_series(sub, e_bbox_key, "cam_valid")
            if use_relaxed_breakdown:
                breakdowns_rep = get_relaxed_breakdowns(repaired_rows, e_bbox_key).tolist()
            else:
                breakdowns_rep = [1 if is_breakdown(repaired_rows[i], e_bbox_key=e_bbox_key) else 0 for i in range(5)]
            row_dict["pattern_repaired"] = classify_pattern(np.array(breakdowns_rep))
        if use_smoothing:
            if repaired_rows is None:
                repaired_rows = _interpolate_invalid_series(sub, e_bbox_key, "cam_valid")

            def _val(r, k):
                x = r.get(k) if hasattr(r, "get") else (r[k] if k in r.index else None)
                if x is None or (isinstance(x, float) and np.isnan(x)):
                    return np.nan
                try:
                    return float(x)
                except (TypeError, ValueError):
                    return np.nan

            e_arr = [_val(repaired_rows[i], e_bbox_key) or _val(repaired_rows[i], "energy_in_bbox") for i in range(5)]
            sp_arr = [_val(repaired_rows[i], "activation_spread") for i in range(5)]
            a = np.asarray(e_arr, dtype=float)
            a = np.where(np.isfinite(a), a, np.nanmean(a[np.isfinite(a)]) if np.any(np.isfinite(a)) else 0)
            e_arr = a.tolist()
            a = np.asarray(sp_arr, dtype=float)
            a = np.where(np.isfinite(a), a, np.nanmean(a[np.isfinite(a)]) if np.any(np.isfinite(a)) else 0)
            sp_arr = a.tolist()
            e_smooth = _smooth_series(e_arr)
            sp_smooth = _smooth_series(sp_arr)
            smoothed_rows = [
                {
                    e_bbox_key: e_smooth[i],
                    "activation_spread": sp_smooth[i],
                    "cam_quality": sub[sub["severity"] == i].iloc[0].get("cam_quality"),
                }
                for i in range(5)
            ]
            if use_relaxed_breakdown:
                breakdowns_smo = get_relaxed_breakdowns(smoothed_rows, e_bbox_key).tolist()
            else:
                breakdowns_smo = [1 if is_breakdown(smoothed_rows[i], e_bbox_key=e_bbox_key) else 0 for i in range(5)]
            row_dict["pattern_smoothed"] = classify_pattern(np.array(breakdowns_smo))
        elif interpolate_invalid:
            row_dict["pattern_smoothed"] = row_dict["pattern_repaired"]

        rows.append(row_dict)

    summary_df = pd.DataFrame(rows)
    if len(summary_df) == 0:
        print("No complete (L0~L4) object×corruption series found.")
        sys.exit(0)

    # Counts
    pattern_counts = summary_df["pattern"].value_counts().to_dict()
    by_corruption = summary_df.groupby("corruption")["pattern"].value_counts().unstack(fill_value=0)

    # Save
    summary_csv = results_dir / "pattern_summary.csv"
    summary_df.to_csv(summary_csv, index=False)
    print(f"Saved {len(summary_df)} rows to {summary_csv}")

    counts_json = results_dir / "pattern_counts.json"
    out = {
        "total_objects": int(len(summary_df)),
        "pattern_counts": {k: int(v) for k, v in pattern_counts.items()},
        "by_corruption": by_corruption.to_dict() if not by_corruption.empty else {},
    }
    with open(counts_json, "w", encoding="utf-8") as f:
        json.dump(out, f, indent=2)
    print(f"Saved {counts_json}")

    # 데이터 균형: 보간/스무딩 분포 (다른 XAI 실험과 비슷한 분포를 기대)
    if "pattern_repaired" in summary_df.columns:
        counts_rep = summary_df["pattern_repaired"].value_counts().to_dict()
        rep_path = results_dir / "pattern_counts_repaired.json"
        with open(rep_path, "w", encoding="utf-8") as f:
            json.dump({"total_objects": int(len(summary_df)), "pattern_counts": {k: int(v) for k, v in counts_rep.items()}}, f, indent=2)
        print(f"Saved (보간) {rep_path}")
    if "pattern_smoothed" in summary_df.columns:
        counts_smo = summary_df["pattern_smoothed"].value_counts().to_dict()
        smo_path = results_dir / "pattern_counts_smoothed.json"
        with open(smo_path, "w", encoding="utf-8") as f:
            json.dump({"total_objects": int(len(summary_df)), "pattern_counts": {k: int(v) for k, v in counts_smo.items()}}, f, indent=2)
        print(f"Saved (스무딩) {smo_path}")

    # 보완 3.2: expanded ROI(1.1x, 1.25x) 기준으로 재분류하여 transient 비율 비교
    expansion_comparison = {}
    for exp_key, label in [("energy_in_bbox_1_1x", "1.1x"), ("energy_in_bbox_1_25x", "1.25x")]:
        if exp_key not in df.columns:
            continue
        rows_exp = []
        for _, r in keys.iterrows():
            sub = df[
                (df["model"] == r["model"])
                & (df["corruption"] == r["corruption"])
                & (df["object_id"] == r["object_id"])
            ].sort_values("severity")
            if len(sub) < 5:
                continue
            breakdowns = []
            for sev in range(5):
                row_sev = sub[sub["severity"] == sev]
                if len(row_sev) == 0:
                    breakdowns.append(1)
                else:
                    breakdowns.append(1 if is_breakdown(row_sev.iloc[0], e_bbox_key=exp_key) else 0)
            pattern = classify_pattern(np.array(breakdowns))
            rows_exp.append({"object_id": r["object_id"], "corruption": r["corruption"], "pattern": pattern})
        if not rows_exp:
            continue
        summary_exp = pd.DataFrame(rows_exp)
        counts_exp = summary_exp["pattern"].value_counts().to_dict()
        total_exp = len(summary_exp)
        expansion_comparison[label] = {
            "total": total_exp,
            "pattern_counts": {k: int(v) for k, v in counts_exp.items()},
            "transient_ratio_pct": round(100.0 * counts_exp.get("transient_instability", 0) / total_exp, 1),
        }
        counts_exp_json = results_dir / f"pattern_counts_expanded_{label.replace('.', '_')}.json"
        with open(counts_exp_json, "w", encoding="utf-8") as f:
            json.dump({"e_bbox_key": exp_key, "total_objects": total_exp, "pattern_counts": expansion_comparison[label]["pattern_counts"]}, f, indent=2)
        print(f"Saved expanded ROI {label} -> {counts_exp_json} (transient_ratio={expansion_comparison[label]['transient_ratio_pct']}%)")
    if expansion_comparison:
        out["expansion_comparison"] = expansion_comparison
        with open(counts_json, "w", encoding="utf-8") as f:
            json.dump(out, f, indent=2)

    for pat in ["transient_instability", "persistent_collapse", "oscillatory", "stable"]:
        subset = summary_df[summary_df["pattern"] == pat]
        if len(subset) > 0:
            path = results_dir / f"pattern_{pat}.csv"
            subset.to_csv(path, index=False)
            print(f"  {pat}: {len(subset)} -> {path}")

    # Sample IDs for examples (up to 20 per pattern)
    samples_json = results_dir / "pattern_sample_ids.json"
    samples = {}
    for pat in ["transient_instability", "persistent_collapse", "oscillatory", "stable"]:
        subset = summary_df[summary_df["pattern"] == pat]
        ids = subset["object_id"].head(20).tolist()
        if ids:
            samples[pat] = ids
    with open(samples_json, "w", encoding="utf-8") as f:
        json.dump(samples, f, indent=2)
    print(f"Saved sample IDs to {samples_json}")

    # Markdown report
    report_md = results_dir / "cam_pattern_report.md"
    rule_section = [
        "# CAM 시계열 패턴 분류 보고서",
        "",
        "## 1. 규칙",
    ]
    if use_relaxed_breakdown:
        rule_section.extend([
            "- **완화 규칙 사용**: 단일 지표(flat/spread=0 등) 단독 붕괴 금지.",
            "- **이상 신호**: 다음 중 2개 이상 동시 성립 시 해당 severity를 이상 신호로 봄.",
            "  - `E_bbox_1.25x <= 0.2`, `cam_quality = flat`, `cam_sum <= 1e-8`, `spread <= 1e-5`,",
            "  - `entropy >= 1.3 * entropy_L0`, `bbox_dist >= 2 * bbox_dist_L0`",
            "- **공식 breakdown**: 이상 신호가 **2단계 연속**이거나, 이상 신호 발생 **후 회복 없음**일 때만 1.",
            "- **회복**: `E_bbox > 0.5` and `activation_spread > 0`",
        ])
    else:
        rule_section.extend([
            "- **Breakdown**: `E_bbox <= 0.05` or `activation_spread <= 1e-6` or `cam_quality != 'high'`",
            "- **회복**: `E_bbox > 0.5` and `activation_spread > 0`",
        ])
    lines = rule_section + [
        "",
        "- **Stable**: 모든 severity에서 breakdown 없음",
        "- **Persistent collapse**: 첫 breakdown 이후 더 높은 severity에서 회복 없음",
        "- **Transient instability**: severity 1~3 중 breakdown 발생 + 이후 회복",
        "- **Oscillatory**: breakdown과 회복이 2회 이상 번갈아 나타남",
        "",
        "## 2. 집계",
        "",
        "| 패턴 | 건수 | 비율(%) |",
        "|------|------|--------|",
    ]
    total = len(summary_df)
    for pat in ["stable", "persistent_collapse", "transient_instability", "oscillatory"]:
        n = pattern_counts.get(pat, 0)
        pct = 100.0 * n / total if total else 0
        lines.append(f"| {pat} | {n} | {pct:.1f} |")
    lines.append("")
    lines.append("## 3. Corruption별")
    lines.append("")
    if not by_corruption.empty:
        lines.append(by_corruption.to_string())
    lines.append("")
    if expansion_comparison:
        lines.append("## 4. ROI 민감도 (보완 3.2): 원본 vs expanded bbox")
        lines.append("")
        lines.append("| 기준 | transient_ratio(%) |")
        lines.append("|------|-------------------|")
        trans_orig = round(100.0 * pattern_counts.get("transient_instability", 0) / total, 1)
        lines.append(f"| 원본 E_bbox | {trans_orig} |")
        for label, data in expansion_comparison.items():
            lines.append(f"| {label} expanded | {data['transient_ratio_pct']} |")
        lines.append("")
        lines.append("확장 시 transient가 크게 줄면 → ROI 민감성(박스 어긋남) 영향이 큼.")
        lines.append("")
    if "pattern_repaired" in summary_df.columns:
        lines.append("## 5. 데이터 균형 (보간/스무딩)")
        lines.append("- **pattern_repaired**: cam_valid==False인 구간을 인접 severity로 선형 보간 후 재분류.")
        lines.append("- **pattern_smoothed**: 3점 이동평균 적용 후 재분류.")
        lines.append("")
        for col, label in [("pattern_repaired", "보간"), ("pattern_smoothed", "스무딩")]:
            if col not in summary_df.columns:
                continue
            cts = summary_df[col].value_counts().to_dict()
            tot = len(summary_df)
            lines.append(f"### {label} 기준 집계")
            lines.append("| 패턴 | 건수 | 비율(%) |")
            lines.append("|------|------|--------|")
            for pat in ["stable", "persistent_collapse", "transient_instability", "oscillatory"]:
                n = cts.get(pat, 0)
                pct = 100.0 * n / tot if tot else 0
                lines.append(f"| {pat} | {n} | {pct:.1f} |")
            lines.append("")
        lines.append("다른 XAI 실험과 분포를 맞추고 싶을 때는 위 보간/스무딩 집계를 참고.")
        lines.append("")
    lines.append("## 5. 해석" if "pattern_repaired" not in summary_df.columns else "## 6. 해석")
    lines.append("- **Transient instability** 비율이 높으면: 중간 severity에서만 CAM이 일시 붕괴했다가 회복하는 비단조 패턴이 많다는 의미.")
    lines.append("- Discussion에서 Grad-CAM/ROI/품질 한계로 설명 가능하다고 기술하고, persistent vs transient를 구분해 해석하는 것을 권장.")
    lines.append("")
    with open(report_md, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))
    print(f"Saved report to {report_md}")

    # 이상적 추세: stable 또는 persistent_collapse 이면서 L0→L4 **단조** 추세만 허용
    # E_bbox: 매 단계 비증가(연속 하강), entropy / bbox_dist / spread: 매 단계 비감소(연속 상승)
    # 오락가락 제거 — "측정 실패 없이 정상적으로 관측된" 연속·단조 열화만
    TOL = 1e-6  # 수치 오차 허용
    ideal_by_model_corr = {}
    for _, r in summary_df.iterrows():
        if r["pattern"] not in ("stable", "persistent_collapse"):
            continue
        sub = df[
            (df["model"] == r["model"])
            & (df["corruption"] == r["corruption"])
            & (df["object_id"] == r["object_id"])
        ].sort_values("severity")
        if len(sub) < 5:
            continue
        def _to_float(x):
            if x is None or (isinstance(x, float) and np.isnan(x)):
                return np.nan
            try:
                return float(x)
            except (TypeError, ValueError):
                return np.nan

        rows = [sub[sub["severity"] == i].iloc[0] for i in range(5)]
        e_vals = [_to_float(rows[i].get("energy_in_bbox_1_25x")) if pd.notna(rows[i].get("energy_in_bbox_1_25x")) else _to_float(rows[i].get("energy_in_bbox")) for i in range(5)]
        ent_vals = [_to_float(rows[i].get("entropy")) for i in range(5)]
        sp_vals = [_to_float(rows[i].get("activation_spread")) for i in range(5)]
        bd_vals = [_to_float(rows[i].get("bbox_center_activation_distance")) for i in range(5)]
        if np.any(np.isnan(e_vals)):
            continue
        # E_bbox: 매 단계 비증가 (단조 감소)
        if any(e_vals[i + 1] > e_vals[i] + TOL for i in range(4)):
            continue
        # entropy: 매 단계 비감소 (단조 증가)
        if any(not np.isnan(ent_vals[i]) and not np.isnan(ent_vals[i + 1]) and ent_vals[i + 1] < ent_vals[i] - TOL for i in range(4)):
            continue
        # spread: 매 단계 비감소 (단조 증가)
        if any(not np.isnan(sp_vals[i]) and not np.isnan(sp_vals[i + 1]) and sp_vals[i + 1] < sp_vals[i] - TOL for i in range(4)):
            continue
        # bbox_dist: 매 단계 비감소 (단조 증가)
        if any(not np.isnan(bd_vals[i]) and not np.isnan(bd_vals[i + 1]) and bd_vals[i + 1] < bd_vals[i] - TOL for i in range(4)):
            continue
        sample_id = str(r["image_id"]) + "_" + str(r["object_id"])
        if not sample_id.endswith(".png"):
            sample_id = sample_id + ".png"
        model = r["model"]
        corr = r["corruption"]
        if model not in ideal_by_model_corr:
            ideal_by_model_corr[model] = {}
        if corr not in ideal_by_model_corr[model]:
            ideal_by_model_corr[model][corr] = []
        ideal_by_model_corr[model][corr].append(sample_id)
    ideal_path = results_dir / "ideal_trend_samples.json"
    for model in ideal_by_model_corr:
        for corr in ideal_by_model_corr[model]:
            ideal_by_model_corr[model][corr] = sorted(ideal_by_model_corr[model][corr])
    with open(ideal_path, "w", encoding="utf-8") as f:
        json.dump(ideal_by_model_corr, f, indent=2)
    n_ideal = sum(len(v) for m in ideal_by_model_corr for v in ideal_by_model_corr[m].values())
    print(f"Saved ideal-trend samples to {ideal_path} (total {n_ideal} samples)")

    # Lead 비교 (transient 제외 시)
    lead_table_path = results_dir / "lead_table.csv"
    lead_stats_path = results_dir / "lead_stats.json"
    if lead_table_path.exists() and lead_stats_path.exists():
        lead_df = pd.read_csv(lead_table_path)
        with open(lead_stats_path, encoding="utf-8") as f:
            lead_stats = json.load(f)
        transient_ids = set(summary_df[summary_df["pattern"] == "transient_instability"]["object_id"].astype(str))
        uid_col = "object_uid" if "object_uid" in lead_df.columns else "object_id"
        if uid_col in lead_df.columns:
            lead_excl = lead_df[~lead_df[uid_col].astype(str).isin(transient_ids)]
            n_lead = int((lead_excl["lead"] > 0).sum()) if "lead" in lead_excl.columns else 0
            n_coincident = int((lead_excl["lead"] == 0).sum()) if "lead" in lead_excl.columns else 0
            n_lag = int((lead_excl["lead"] < 0).sum()) if "lead" in lead_excl.columns else 0
            mean_lead = float(lead_excl["lead"].mean()) if "lead" in lead_excl.columns and len(lead_excl) else None
            comp = {
                "with_transient": {
                    "n_lead": lead_stats.get("n_lead"),
                    "n_coincident": lead_stats.get("n_coincident"),
                    "n_lag": lead_stats.get("n_lag"),
                    "mean_lead": lead_stats.get("mean_lead"),
                },
                "exclude_transient_instability": {
                    "n_lead": n_lead,
                    "n_coincident": n_coincident,
                    "n_lag": n_lag,
                    "mean_lead": mean_lead,
                    "n_excluded": len(transient_ids),
                },
            }
            comp_path = results_dir / "lead_comparison_with_without_transient.json"
            with open(comp_path, "w", encoding="utf-8") as f:
                json.dump(comp, f, indent=2)
            print(f"Lead comparison (with vs without transient): {comp_path}")
            print(f"  With transient:    n_lead={comp['with_transient']['n_lead']}, n_coincident={comp['with_transient']['n_coincident']}, n_lag={comp['with_transient']['n_lag']}")
            print(f"  Excl. transient:   n_lead={n_lead}, n_coincident={n_coincident}, n_lag={n_lag}, mean_lead={mean_lead}")

    print()
    print("Done.")


if __name__ == "__main__":
    main()
