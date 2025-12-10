  # CLIP-Based Image Understanding  
**Text→Image Retrieval + Crop-Aware Reranking + Zero-Shot Image Classification (CIFAR-10 demo)**

This project implements a compact vision–language pipeline using **CLIP** embeddings for:

1) **Text-to-Image Retrieval**  
   - Input: a natural-language prompt (e.g., `"dog"`, `"airplane"`)  
   - Output: Top-K images ranked by **cosine similarity** in CLIP embedding space

2) **Crop-Aware Retrieval (Add-on)**  
   - Uses **Mask R-CNN** to propose object regions (crops)  
   - Computes CLIP embeddings for crops  
   - Reranks with **max fusion** between full-image score and best-crop score

3) **Zero-Shot Image Classification**  
   - Input: an uploaded image (even low-res / blurry)  
   - Output: predicted label + Top-3 candidates with confidence scores  
   - Uses CLIP text prompts like `"a photo of a dog"` for each class

> Note: This system **does not generate images**. It retrieves and ranks from a fixed image database.

---

## Why this project
- Demonstrates **feature embeddings**, **similarity search**, **ranking metrics**, and **interpretability**
- Relevant to real applications like **searching CCTV frames** by concept or triaging large image streams
- Built as an academic prototype: emphasize **responsible use** (not for bypassing security systems)

---

## Features
- ✅ Baseline CLIP retrieval with Top-K visualization + printed labels/scores  
- ✅ Interactive prompt loop (`exit` to stop)  
- ✅ Out-of-domain query handling (“No match” via similarity threshold)  
- ✅ Crop-aware reranking with Mask R-CNN crops  
- ✅ Recall@K + MRR evaluation  
- ✅ Image→Label zero-shot classification (Top-1 and Top-3)

---

## Example outputs (what you should see)
### Retrieval (Baseline)
- **Step 6**: encodes text prompt → prints embedding dimension  
- **Step 7**: ranks database images → shows Top-K grid + similarity scores  

### Crop-aware retrieval
- **Step 9**: extracts crops + encodes crops → prints crop stats  
- **Step 10**: reranks using max fusion → shows new Top-K grid  

### Classification
- Upload an image → prints Top-1 label + Top-3 candidates with probabilities

---

## Dataset
Default demo uses **CIFAR-10** (small, fast, reproducible).

- CIFAR-10 images are only **32×32**, so images may appear **blurry when enlarged**.  
  This is expected and due to dataset resolution.

> For higher-quality visuals and richer prompts, future work can switch to COCO/Flickr8k style datasets.

---

## Installation
### Option A: Google Colab (recommended)
Open the notebook(s) in `notebooks/` and run cells top-to-bottom.

### Option B: Local (Python ≥ 3.9)
```bash
python -m venv .venv
source .venv/bin/activate   # Windows: .venv\Scripts\activate
pip install -r requirements.txt
