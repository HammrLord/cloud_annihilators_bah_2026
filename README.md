# Cloud Annihilators

**Generative AI-Based Cloud Removal and Reconstruction for LISS-IV Satellite Imagery**

ISRO BAH 2026 — Delhi Technological University

## Architecture

### Stage 1: MSKT (Multimodal Spectral Knowledge Transformer)
- Pretrained on SEN12MS-CR to reconstruct 13-band Sentinel-2 from LISS-IV (3 bands) + SAR (2 bands)
- Learns rich multispectral representations using L1, SSIM, and SAM losses

### Stage 2: Physics-guided Cloud Removal
1. **PPE** (Physical Property Estimator) → physics-guided initialization
2. **MSKT** (frozen/pretrained) → augmented spectral knowledge  
3. **Physics-conditioned AdaLN** → adapts features per atmospheric conditions
4. **U-shaped ECRFormer** → fuses optical + SAR + physics features → predicts residual ΔR

**Result:** R_final = clamp(R_init + ΔR, 0, 1)

### Key Features
- **Spectral Channel Augmentation** — MSKT bridges LISS-IV's 3-band constraint
- **Physics-conditioned AdaLN** — dynamic feature adaptation
- **Modality Detection & Routing** — automatic sensor identification
- **Downstream Geospatial Insights** — NDVI, change detection, vegetation monitoring
- **Quality Assessment Dashboard** — confidence heatmaps, uncertainty visualization

## Results

| Model | PSNR (dB) | SSIM |
|-------|-----------|------|
| ECRFormer (direct transfer) | 9.77 | 0.5199 |
| **MSKT + ECRFormer (Ours)** | **25.11** | **0.9067** |

## Quick Start

```bash
pip install -r requirements.txt

# Stage 1: MSKT pretraining
python scripts/train_stage1.py --config configs/stage1_mskt.yaml

# Stage 2: Full model training
python scripts/train_stage2.py --config configs/stage2.yaml

# Inference
python scripts/inference.py --checkpoint experiments/stage2/best.ckpt \
                            --input data/cloudy.tif --output data/output.tif

# Web demo
python web_demo/app.py --checkpoint experiments/stage2/best.ckpt
```

## Tech Stack
- **Framework:** PyTorch, PyTorch Lightning
- **Co-registration:** AROSICS, ESA SNAP
- **Data:** Bhoonidhi (LISS-IV), Copernicus (Sentinel-1/2)
- **Frontend:** Gradio, React/Next.js
- **Visualization:** Plotly, D3.js, Leaflet.js
- **GIS:** GDAL, rasterio, QGIS

## Team
- Kartik Sharma (Leader)
- Dhruv Kashyap
- Bibek Sanjeev
- Harshit Nayak
