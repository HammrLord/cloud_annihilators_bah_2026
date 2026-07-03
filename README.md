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
- **Generalizable Training Framework** — sensor-agnostic backbone, extensible beyond LISS-IV


## UI/UX

The frontend is built as a single interactive web app (React/Next.js + Leaflet.js) with three core views:

### 1. Comparison Viewer
- Side-by-side toggle between **Cloudy** and **Reconstructed** imagery on an interactive map canvas
- **Layer Control panel** to switch between Optical (raw), Cloud-Free (reconstructed), SAR (VV/VH), and DEM layers
- Derived-product overlays: **NDVI** and **Confidence Map**, toggled independently
- Live reconstruction-quality badge (e.g. HIGH) with a short vegetation-health summary generated from NDVI stats

### 2. Core Intelligence Hub
- **Model Recommendation Engine** — auto-selects the best pipeline (e.g. Cloud Removal, Change Detection) based on scene conditions, with a confidence score shown per recommendation
- **Cloud Density** readout (% cover) with a quick preview thumbnail
- **Data Availability** tracker for Optical / SAR / DEM sources (online / syncing status)
- **Recent Processing History** table — run ID, pipeline, mode (Quality/Speed), status, timestamp

### 3. Analytics Dashboard
- **Spectral signature plots** and **pixel histograms** (Plotly/D3.js) comparing cloudy vs. reconstructed bands
- **NDVI distribution** and **elevation profile (DEM)** charts
- **Reconstruction Confidence** heatmap overlay with a High/Mid/Low legend
- **Key statistics** panel: total cloud coverage, mean NDVI, water-body area, etc.
- **One-click export**: GeoTIFF raster, CSV statistics, PDF summary report (ReportLab)

<img height="250" alt="image" src="https://github.com/user-attachments/assets/c4492b27-1b74-47a3-9584-cd1561b3828b" />
<img height="250" alt="image" src="https://github.com/user-attachments/assets/e726d127-6352-4f4d-97c0-772712432516" />
<img height="250" alt="image" src="https://github.com/user-attachments/assets/79874b27-91a5-4c03-baf4-e6137ea30cba" />


## Results

| Model | PSNR (dB) | SSIM |
|-------|-----------|------|
| ECRFormer (direct transfer) | 9.77 | 0.5199 |
| **MSKT + ECRFormer (Ours)** | **25.11** | **0.9067** |

<img width="256" height="256" alt="image" src="https://github.com/user-attachments/assets/b624521d-c08a-4385-b253-8863504bcf67" /> <img width="256" height="256" alt="image" src="https://github.com/user-attachments/assets/99963479-7d53-4bdf-bd15-176155089728" />  

<img width="256" height="256" alt="image" src="https://github.com/user-attachments/assets/75f2fb14-5aba-4f46-b151-f2e367558a99" /> <img width="256" height="256" alt="image" src="https://github.com/user-attachments/assets/4222e0f3-4ea8-48f0-b6cf-53b8600a06e7" />




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
