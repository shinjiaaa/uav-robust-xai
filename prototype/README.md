# DASC 실험 프로토타입

Input/Output 확인용 간단한 프로토타입 (HTML/CSS/JS).

## 실행 방법

1. 파이프라인 실행 후:
   ```bash
   cd c:\uav-robust-xai
   python scripts/00_full_experiment.py
   ```

2. 프로토타입 서버 실행 (프로젝트 루트에서):
   ```bash
   python -m http.server 8080
   ```

3. 브라우저에서:
   ```
   http://localhost:8080/prototype/
   ```

## 표시 항목

- **Input**: 원본 이미지 업로드
- **Output**: 변조별 이미지 + Grad-CAM Heatmap (results/heatmap_samples/)
- **IoU / Miss Rate 곡선**
- **성능 저하 단계 vs Grad-CAM 붕괴 단계**
- **노이즈별 일관성**
