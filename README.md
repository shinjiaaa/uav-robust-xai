# UAV Robust XAI: VisDrone Robustness Evaluation Pipeline

End-to-end reproducible experiment pipeline for robustness evaluation of object detectors on VisDrone with controlled image corruptions and XAI (Explainable AI), focusing on **single-image evaluation** and **failure-event based Grad-CAM analysis**.

## DASC 실험 모드

**연구 질문**: 객체 탐지 모델의 성능 저하가 일어나기 전, XAI(Grad-CAM)가 성능 저하의 전조 붕괴를 잡을 수 있는가?

- **정량적 평가**: IoU curve, mAP, 모델 저하 단계 vs Grad-CAM 패턴 붕괴 단계, 노이즈별 일관성
- **정성적 평가**: Heatmap 시각화로 인지적 도움 평가
- **모델**: YOLO base (yolo_generic)
- **변조**: 안개(fog), 모션 블러(motion_blur), 저조도(lowlight) | 단계 0(원본)~4
- **프로토타입**: `prototype/index.html` — Input/Output 확인용

## Research Question (RQ)

**RQ**: In single-image object detection, when image corruption severity increases, does the resulting detection performance degradation co-occur with consistent changes in Grad-CAM distribution patterns? Can these CAM-based changes serve as generalizable risk signals across different corruption types and models?

## Key Features

- **Single Image Evaluation**: Each image is processed independently (no frame sequences)
- **Failure-Event Based CAM Analysis**: Grad-CAM analysis focused on failure events
- **Automatic Risk Region Detection**: Identifies performance cliffs and instability regions
- **CAM Distribution Metrics**: Analysis of CAM patterns across severity levels
- **Multi-Model Evaluation**: Generic YOLO, Fine-tuned YOLO, RT-DETR

## Experiment Design

### 1. Dataset
- **VisDrone2019-DET**: Drone-view object detection dataset
- **Single Images**: Each image is processed independently (no frame sequences)
- **Tiny Objects**: Focus on small-scale objects (area ≤ 60 px², width ≤ 6px, height ≤ 9px)
- **Sampling**: N=100 tiny objects, one per image (no duplicate images)

### 2. Image Corruptions
- **Fog**: Atmospheric scattering model (alpha: 0.0-0.60, extreme: 0.95)
- **Low-light**: Gamma correction + brightness scaling + noise (extreme: gamma=5.0, brightness=0.1)
- **Motion Blur**: Linear kernel convolution (kernel length: 7-25, extreme: 100)
- **Severity Levels**: 
  - **Pilot Mode**: [0, 50] (extreme severity for initial testing)
  - **Full Mode**: [0, 1, 2, 3, 4] (standard 5 levels)
- **Dynamic Refinement**: Failure regions are subdivided into 10 steps for detailed analysis

### 3. Models
- **Generic YOLO**: COCO-pretrained YOLOv8s (currently used for pilot experiment)
- ~~**Aviation Fine-tuned YOLO**: VisDrone fine-tuned YOLOv8s~~ (disabled for pilot)
- ~~**RT-DETR**: Architecture-different baseline~~ (disabled for pilot)

### 4. Evaluation Metrics

#### Global (Dataset-wide)
- mAP@0.5, mAP@0.5:0.95
- Precision, Recall

#### Local (Tiny Object Analysis)
- **Miss Rate Curve**: Failure rate across severity levels
- **Score/IoU Drop**: Confidence and accuracy degradation
- **Instability Detection**: Variance-based instability regions (across severities)

#### CAM Distribution Metrics
- **Energy in Bbox**: Activation concentration in GT region
- **Activation Spread**: Spatial dispersion of activations
- **Entropy**: Distribution entropy
- **Center Shift**: Movement of activation center

### 5. Failure Event Detection
- **First Miss**: Severity/frame where detection first fails
- **Score Drop**: Significant confidence decrease
- **IoU Drop**: Significant accuracy decrease
- **Instability**: High variance regions

### 6. Risk Region Identification
- **mAP Drop**: Severity where mAP drops ≥15% from baseline
- **Miss Rate Threshold**: Severity where miss rate ≥50%
- **Automatic Detection**: Programmatic identification of risk regions

## Installation

```bash
# Clone repository
git clone <repository-url>
cd uav-robust-xai

# Install dependencies
pip install -r requirements.txt

# Setup environment
python scripts/00_setup_env.py
```

## Usage

### Full Pipeline

```bash
# Step 1: Sample tiny objects from single images
python scripts/01_sample_tiny_objects.py

# Step 2: Generate corruption sequences
python scripts/02_generate_corruption_sequences.py

# Step 3: Detect tiny objects
python scripts/03_detect_tiny_objects_timeseries.py

# Step 4: Detect failure events and risk regions
python scripts/04_detect_failure_events.py

# Step 5: Grad-CAM failure analysis (with dynamic refinement)
python scripts/05_gradcam_failure_analysis.py

# Step 6: Generate LLM report
python scripts/06_llm_report.py
```

### Pilot Experiment (Recommended First)

Run a small-scale pilot experiment with extreme severities:

```bash
# Run pilot experiment (extreme severity 50, N=100 samples, one per image)
python scripts/00_pilot_experiment.py
```

This will test the pipeline with:
- **Sample size**: 100 tiny objects (one per image, no duplicates)
- **Extreme severity**: 50 (to test degradation limits)
- **Dynamic refinement**: Failure regions automatically subdivided into 10 steps

## Configuration

Edit `configs/experiment.yaml` to customize:
- Frame sequence settings (clip length, stride)
- Tiny object thresholds
- Corruption parameters
- Model configurations
- Risk detection thresholds
- CAM analysis settings

## Outputs

Results are saved in `results/` directory:

- `tiny_objects_samples.json`: Sampled tiny objects
- `tiny_records_timeseries.csv`: Detection records (across severities)
- `failure_events.csv`: Detected failure events
- `risk_regions.csv`: Identified risk regions
- `gradcam_metrics_timeseries.csv`: CAM distribution metrics
- `report.md`: LLM-generated report

## Reproducibility

- Fixed random seed (42)
- Deterministic corruptions
- Cached corrupted images
- Structured results format

## License

[Your License Here]
