# Computer Vision Portfolio Project
## Semantic Segmentation · ViT Classification · Zero-Shot CLIP Retrieval

**Stack:** Python 3.10 · PyTorch 2.1 · timm · open-clip-torch · albumentations · FAISS

---

## Installation
```bash
pip install -r requirements.txt
```

---

## Part 1 — Semantic Segmentation with DeepLabV3+

### Architecture
- **Encoder**: ResNet-101 (ImageNet pretrained) with dilated convolutions
- **ASPP**: Atrous Spatial Pyramid Pooling (rates 6, 12, 18, 24)
- **Decoder**: bilinear 4× up-sample + 3×3 conv projection
- **Loss**: CrossEntropyLoss + 0.4 × auxiliary loss (ignore_index=255)
- **Optimizer**: SGD (momentum=0.9) + polynomial LR decay
- **Dataset**: Cityscapes fine annotations → 19 trainId classes

### Augmentation Ablation Configs
| Config | Transforms |
|--------|-----------|
| A0 | Resize only (baseline) |
| A1 | A0 + HorizontalFlip |
| A2 | A1 + RandomCrop (0.5–2× scale) |
| A3 | A2 + ColorJitter |
| A4 (full) | A3 + GaussNoise + GaussianBlur + GridDistortion |

### Commands
```bash
# Train with full augmentation
python Part1_train.py --data_root /path/to/cityscapes --aug_config A4 --epochs 80

# Run all ablation configs
python Part1_train.py --data_root /path/to/cityscapes --ablation

# Evaluate per-class mIoU
python Part1_evaluate.py --data_root /path/to/cityscapes \
    --checkpoint checkpoints/seg/A4/best.pth

# Visualize predictions
python Part1_visualize.py --mode predict \
    --checkpoint checkpoints/seg/A4/best.pth --image sample.png

# Plot ablation bar chart
python Part1_visualize.py --mode ablation \
    --ablation_json checkpoints/seg/ablation_results.json
```

---

## Part 2 — Vision Transformer Classification

### Models
| Model | Params | ImageNet Top-1 |
|-------|--------|---------------|
| ViT-B/16 | 86M | ~81.8% |
| ResNet-50 | 25M | ~76.1% |
| EfficientNet-B4 | 19M | ~82.9% |

### Training Strategy
1. **Warm-up phase** (5 epochs): freeze backbone, train head only (AdamW, lr=1e-3)
2. **Fine-tune phase**: unfreeze all layers with **differential LR** (backbone=1e-5, head=1e-3) + CosineAnnealingLR
3. **Label smoothing** = 0.1 to prevent overconfidence
4. **Early stopping** (patience=7) on validation accuracy

### Commands
```bash
# Train ViT-B/16
python Part2_train.py --data_root /path/to/dataset --model_name vit_b16 --epochs 30

# Compare all three models
python Part2_train.py --data_root /path/to/dataset --compare_all

# Evaluate + confusion matrix
python Part2_evaluate.py --data_root /path/to/dataset \
    --checkpoint checkpoints/cls/vit_b16/best.pth --model_name vit_b16

# Attention map visualization
python Part2_attention_viz.py \
    --checkpoint checkpoints/cls/vit_b16/best.pth \
    --image path/to/image.jpg --output attention_maps.png
```

---

## Part 3 — Zero-Shot CLIP Image Retrieval

### Architecture
- **CLIP** (ViT-B/32 default): dual encoder maps images and text into a shared 512-d embedding space
- **FAISS IndexFlatIP**: exact inner-product search (≡ cosine similarity on L2-normalized vectors)
- Supports **text-to-image** and **image-to-image** search
- Evaluation via **Recall@K** (K = 1, 5, 10)

### Commands
```bash
# Step 1: Build the gallery index
python Part3_build_index.py \
    --gallery_dir /path/to/images --index_dir clip_index

# Step 2: Text-to-image retrieval
python Part3_demo.py --index_dir clip_index \
    --mode text --query "a red car on a rainy street"

# Step 3: Image-to-image retrieval
python Part3_demo.py --index_dir clip_index \
    --mode image --query /path/to/query.jpg

# Step 4: Recall@K evaluation
python Part3_demo.py --index_dir clip_index \
    --mode eval --eval_queries queries.json
```

### queries.json format
```json
[
  {"query": "a cat sitting on a chair", "label": "cat"},
  {"query": "a dog running on grass",   "label": "dog"}
]
```

---

## File Map
| File | Description |
|------|------------|
| `Part1_dataset.py` | Cityscapes loader, 35→19 class remapping |
| `Part1_transforms.py` | Albumentations augmentation configs (A0–A4) |
| `Part1_model.py` | DeepLabV3+ factory, SegmentationLoss |
| `Part1_train.py` | Training loop + ablation runner |
| `Part1_evaluate.py` | Per-class mIoU, pixel accuracy, confusion matrix |
| `Part1_visualize.py` | Color overlay visualization + ablation chart |
| `Part2_dataset.py` | Generic ImageFolder dataset |
| `Part2_model.py` | ViT-B/16, ResNet-50, EfficientNet-B4 factory |
| `Part2_train.py` | 2-phase fine-tuning, early stopping |
| `Part2_evaluate.py` | Top-1/5 accuracy, per-class F1, confusion matrix |
| `Part2_attention_viz.py` | Attention rollout + per-head maps |
| `Part3_build_index.py` | CLIP encoding + FAISS index builder |
| `Part3_retrieval.py` | CLIPRetrieval engine (text↔image) |
| `Part3_demo.py` | Visual demo + Recall@K evaluator |
| `requirements.txt` | All dependencies |
