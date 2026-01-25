# UAV Robust XAI: VisDrone Robustness Evaluation Pipeline

End-to-end reproducible experiment pipeline for robustness evaluation of object detectors on VisDrone with controlled image corruptions and XAI (Explainable AI), focusing on **continuous frame sequences** and **failure-event based Grad-CAM analysis**.

## Research Question (RQ)

**RQ**: In continuous frame-based object detection, when image corruption severity increases, does the resulting detection performance degradation co-occur with consistent changes in Grad-CAM distribution patterns? Can these CAM-based changes serve as generalizable risk signals across different corruption types and models?

## Key Features

- **Continuous Frame Sequences**: Video sequence-based evaluation (not single images)
- **Failure-Event Based CAM Analysis**: Grad-CAM analysis focused on failure events
- **Automatic Risk Region Detection**: Identifies performance cliffs and instability regions
- **CAM Distribution Metrics**: Time-series analysis of CAM patterns
- **Multi-Model Evaluation**: Generic YOLO, Fine-tuned YOLO, RT-DETR

## Experiment Design

### 1. Dataset
- **VisDrone2019-DET**: Drone-view object detection dataset
- **Frame Sequences**: Continuous frame clips extracted from video sequences
- **Tiny Objects**: Focus on small-scale objects (area ≤ 60 px², width ≤ 6px, height ≤ 9px)

### 2. Image Corruptions
- **Fog**: Atmospheric scattering model (alpha: 0.0-0.60)
- **Low-light**: Gamma correction + brightness scaling + noise
- **Motion Blur**: Linear kernel convolution (kernel length: 7-25)
- **Severity Levels**: 0-4 (5 levels per corruption type)

### 3. Models
- **Generic YOLO**: COCO-pretrained YOLOv8s
- **Aviation Fine-tuned YOLO**: VisDrone fine-tuned YOLOv8s
- **RT-DETR**: Architecture-different baseline

### 4. Evaluation Metrics

#### Global (Dataset-wide)
- mAP@0.5, mAP@0.5:0.95
- Precision, Recall

#### Local (Tiny Object Tracking)
- **Miss Rate Curve**: Failure rate across severity
- **Score/IoU Drop**: Confidence and accuracy degradation
- **Instability Detection**: Variance-based instability regions

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
# Step 1: Extract frame clips with tiny objects
python scripts/01_extract_frame_clips.py

# Step 2: Generate corruption sequences
python scripts/02_generate_corruption_sequences.py

# Step 3: Detect tiny objects (time-series)
python scripts/03_detect_tiny_objects_timeseries.py

# Step 4: Detect failure events and risk regions
python scripts/04_detect_failure_events.py

# Step 5: Grad-CAM failure analysis
python scripts/05_gradcam_failure_analysis.py

# Step 6: Generate LLM report
python scripts/06_llm_report.py
```

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

- `frame_clips.json`: Extracted frame clips
- `tiny_objects_samples.json`: Sampled tiny objects
- `tiny_records_timeseries.csv`: Time-series detection records
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
