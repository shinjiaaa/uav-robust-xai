# User Study Pipeline: FastCAV Pseudo-Heatmap vs Grad-CAM (Academic Appendix)

This document describes a **reproducible** comparison pipeline for object detection under corruptions (fog, lowlight, motion blur). It is aligned with the implementation in `src/user_study/` and `scripts/12_user_study_export.py`.

**Terminology (critical):**

| Name | Meaning in this repository |
|------|------------------------------|
| **FastCAV** | Global concept scores per detection (e.g. `tiny_object_visibility`, fog-related concepts). **No native pixel-level map.** |
| **Pseudo-heatmap** | `H = normalize(B · g(c))`: fixed spatial prior `B` (default: Sobel magnitude + Gaussian blur on the **same** corrupted image) times scalar stress `g` from concepts. **Not** a claim of FastCAV localization. |
| **`gradcampp`** (`heatmap_samples/gradcampp/`; legacy disk folder `fastcam` is aliased in the app) | **Grad-CAM++** (`YOLOGradCAMPP`). **This is not FastCAV.** |
| **Grad-CAM arm** | Class-discriminative saliency from `heatmap_samples/{gradcam|gradcampp|layercam}/...` (choose one method folder for fair single comparison). |

**Constraints honored:** no bounding boxes in user-facing explanations; no change to model internals; post-hoc visualization and LLM text only.

---

## 1. FastCAV pseudo-heatmap generation

### Method

1. **Surrogate spatial prior** `B`: same resolution as the input image. Default: edge-energy map from Sobel gradients on the corrupted frame, then Gaussian blur (`gaussian_sigma` in `configs/experiment.yaml` → `user_study.pseudo_heatmap`).
2. **Scalar stress** `g ∈ (0,1]`: from global concept scores via `scalar_stress_from_concepts()` (default emphasizes low `tiny_object_visibility`).
3. **Pseudo-heatmap** `H = normalize(B_01 · g)`; JET colormap + semi-transparent overlay on the original corrupted image.

### API (implementation)

- **Function:** `generate_fastcav_pseudo_heatmap(image_bgr, concept_scores, base_map=None, gaussian_sigma=..., stress_fn=...)`
- **File:** `src/user_study/pseudo_heatmap.py`
- **Optional** `base_map`: replace `B` with any H×W float (e.g. external prior), still scaled by `g`.

### Pseudocode

```
B ← Sobel(|∇I|) then GaussianBlur(σ)   # or user-supplied base_map
B_01 ← minmax(B)
g ← stress_fn(concept_scores)         # default: scalar_stress_from_concepts
H ← minmax(B_01 * g)
overlay ← alpha * ColorMap(JET, H) + (1-alpha) * I
```

### Disclosure

Stored in JSON: `disclaimer` field and `README_user_study.txt` — pseudo-heatmap is **not** FastCAV spatial explanation.

---

## 2. Data alignment for fair comparison

### Organization

| Source | Role |
|--------|------|
| `results/detection_records.csv` | Same image path, corruption, severity, prediction metadata for each unit. |
| `results/fastcav_tiny_concept_scores.csv` | `concept_*` columns keyed by `(image_id, object_uid, corruption, severity)`. |
| `results/heatmap_samples/<xai_method>/<model_id>/<corruption>/L<sev>/<image_id>_<object_uid>.png` | Grad-CAM family overlay (copy into user-study bundle). |

**Join:** inner merge on `(image_id, object_uid, corruption, severity)` so FastCAV and Grad-CAM refer to the **same** detection event. Only `concept_*` columns are merged from FastCAV to avoid duplicate score columns.

### Selecting representative samples (suggested stratification)

- **Severity 0:** clean reference for the same object track.
- **Onset:** first severity where a predefined metric crosses threshold (e.g. from `tiny_curves` / DASC onset).
- **Failure:** severity where miss or low confidence occurs.

Export script does not enforce stratification; filter `detection_records` / merged table **before** export or post-filter `manifest.csv`.

---

## 3. LLM explanation generation (identical template)

### Structure (both methods)

Same system prompt and user template with **four sections** (English in code; translate for localized studies if needed):

```
[Status]
[Location]
[Cause]
[Risk]
```

- **Grad-CAM block:** approximate spatial summary from the saved overlay PNG (`summarize_heatmap_regions` on grayscale).
- **FastCAV block:** concept scores + pseudo spatial summary + explicit non-localization note.

### Code

- `SYSTEM_PROMPT_STRICT`, `build_strict_user_study_user_message`, `narrate_gradcam_visualization_summary`, `narrate_fastcav_visualization_summary`, `format_concept_signals_with_trends`, `format_performance_signals_block` — `src/user_study/llm_user_study.py`
- **API:** `generate_explanation_openai(user_message, model=...)`; model from `user_study.llm.model` in YAML. Export builds inputs from overlay summaries vs L0, concept trends vs L0, and confidence/miss trends vs L0.

### Example formatted output (illustrative)

```
[Status]
The detector reports class "pedestrian" with moderate confidence under fog corruption.

[Location]
The saliency emphasis appears strongest in the upper-right quadrant of the frame.

[Cause]
Fog reduces contrast; the model may rely on diffuse edge patterns rather than crisp object boundaries.

[Risk]
Trusting this detection without verification could be unsafe if the object is small or partially occluded.
```

(FastCAV arm would differ in [Location]/[Cause] to reflect global concepts + pseudo-heatmap disclaimer.)

---

## 4. User study sample layout

Each **unit** folder (`results/user_study_bundles/unit_XXXXX/`):

| File | Content |
|------|---------|
| `original.png` | Same corrupted frame as detection/CAM. |
| `gradcam_overlay.png` | Copy of Grad-CAM **family** overlay (path controlled by `--gradcam-xai`, default from `user_study.gradcam_overlay_method`). |
| `fastcav_pseudo_jet.png` | JET-only pseudo-heatmap. |
| `fastcav_pseudo_overlay.png` | **FastCAV pseudo-heatmap overlaid on image** (this is the FastCAV visual condition). |
| `fastcav_pseudo_meta.json` | `stress_g`, disclaimer, concepts, pseudo spatial summary. |
| `prompt_gradcam.txt` / `prompt_fastcav.txt` | Same-template prompts. |
| `explanation_gradcam.txt` / `explanation_fastcav.txt` | LLM outputs (if not `--skip-llm`). |

**Conditions:**

- **A:** `original.png` + `gradcam_overlay.png` + `explanation_gradcam.txt`
- **B:** `original.png` + `fastcav_pseudo_overlay.png` + `explanation_fastcav.txt`
- **C:** Visualization only: same PNGs as A or B **without** showing the text file (or blank instruction sheet).

Root: `manifest.csv`, `evaluation_questions_likert.txt`, `README_user_study.txt`.

---

## 5. Evaluation questions (Likert)

Generated file: `evaluation_questions_likert.txt` (1 = Strongly disagree … 5 = Strongly agree).

**Dimensions:**

1. **Spatial alignment:** Q1–Q2 (text vs visualization).
2. **Interpretability:** Q3–Q4.
3. **Usefulness:** Q5–Q6.
4. **Manipulation check (FastCAV pseudo):** Q7 (participants acknowledge pseudo-heatmap is not pixel-level concept causation).

Optional free-text for mismatch notes.

---

## Reproduction commands

```bash
# 1) Detection + FastCAV scores + Grad-CAM overlays (existing pipelines; not repeated here)
# 2) Export user-study bundles (pseudo-heatmap + prompts + optional LLM)
python scripts/12_user_study_export.py --max-units 50
# Grad-CAM++ overlays live under heatmap_samples/gradcampp/ (or legacy fastcam/) — use:
python scripts/12_user_study_export.py --gradcam-xai gradcampp --max-units 50
```

Configure defaults in `configs/experiment.yaml` under `user_study:`.

---

## References (code)

- `src/user_study/pseudo_heatmap.py` — pseudo-heatmap
- `src/user_study/heatmap_summary.py` — grid summary for overlays
- `src/user_study/llm_user_study.py` — prompts and questionnaire
- `scripts/12_user_study_export.py` — bundle export
