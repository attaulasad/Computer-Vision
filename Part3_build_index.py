"""
Part 3 – Zero-Shot CLIP Retrieval
build_index.py: Encode an image gallery into a CLIP embedding index.

Usage:
    python Part3_build_index.py \
        --gallery_dir /path/to/images \
        --index_dir   clip_index \
        --model_name  ViT-B/32
"""

import os
import argparse
import json
import pickle

import numpy as np
import torch
import faiss
from PIL import Image
from tqdm import tqdm
import open_clip


SUPPORTED_CLIP_MODELS = {
    "ViT-B/32":  ("ViT-B-32",  "openai"),
    "ViT-B/16":  ("ViT-B-16",  "openai"),
    "ViT-L/14":  ("ViT-L-14",  "openai"),
}


def load_clip_model(model_name: str = "ViT-B/32", device: str = "cpu"):
    """Load CLIP model + preprocessor via open_clip."""
    arch, pretrained = SUPPORTED_CLIP_MODELS[model_name]
    model, _, preprocess = open_clip.create_model_and_transforms(
        arch, pretrained=pretrained
    )
    model = model.to(device).eval()
    return model, preprocess


@torch.no_grad()
def encode_images(image_paths: list, model, preprocess, device: str,
                   batch_size: int = 64) -> np.ndarray:
    """Encode a list of image paths → L2-normalized embeddings."""
    all_embeddings = []
    for i in tqdm(range(0, len(image_paths), batch_size), desc="Encoding images"):
        batch_paths = image_paths[i: i + batch_size]
        images = []
        for p in batch_paths:
            try:
                img = Image.open(p).convert("RGB")
                images.append(preprocess(img))
            except Exception as e:
                print(f"  Warning: could not open {p}: {e}")
                images.append(torch.zeros(3, 224, 224))

        batch   = torch.stack(images).to(device)
        feats   = model.encode_image(batch)
        feats   = feats / feats.norm(dim=-1, keepdim=True)
        all_embeddings.append(feats.cpu().numpy())

    return np.vstack(all_embeddings).astype(np.float32)


def build_faiss_index(embeddings: np.ndarray) -> faiss.IndexFlatIP:
    """
    Build an inner-product FAISS index.
    Since embeddings are L2-normalized, inner product == cosine similarity.
    """
    dim = embeddings.shape[1]
    index = faiss.IndexFlatIP(dim)
    index.add(embeddings)
    return index


def collect_image_paths(gallery_dir: str):
    exts = {".jpg", ".jpeg", ".png", ".bmp", ".webp", ".tiff"}
    paths, labels = [], []
    for root, _, files in os.walk(gallery_dir):
        for f in sorted(files):
            if os.path.splitext(f)[1].lower() in exts:
                paths.append(os.path.join(root, f))
                # Use sub-folder name as label (if any)
                rel  = os.path.relpath(root, gallery_dir)
                labels.append(rel if rel != "." else "unknown")
    return paths, labels


def get_args():
    p = argparse.ArgumentParser()
    p.add_argument("--gallery_dir", type=str, required=True)
    p.add_argument("--index_dir",   type=str, default="clip_index")
    p.add_argument("--model_name",  type=str, default="ViT-B/32",
                   choices=list(SUPPORTED_CLIP_MODELS.keys()))
    p.add_argument("--batch_size",  type=int, default=64)
    return p.parse_args()


def main():
    args   = get_args()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Building CLIP index | model={args.model_name} | device={device}")

    model, preprocess = load_clip_model(args.model_name, device)
    paths, labels     = collect_image_paths(args.gallery_dir)
    print(f"Found {len(paths)} images in gallery.")

    embeddings = encode_images(paths, model, preprocess, device, args.batch_size)
    print(f"Embedding shape: {embeddings.shape}")

    index = build_faiss_index(embeddings)

    os.makedirs(args.index_dir, exist_ok=True)
    faiss.write_index(index, os.path.join(args.index_dir, "index.faiss"))

    metadata = {"paths": paths, "labels": labels, "model": args.model_name,
                "dim": int(embeddings.shape[1])}
    with open(os.path.join(args.index_dir, "metadata.json"), "w") as f:
        json.dump(metadata, f, indent=2)

    np.save(os.path.join(args.index_dir, "embeddings.npy"), embeddings)
    print(f"\nIndex saved → {args.index_dir}/")
    print(f"  index.faiss    ({len(paths)} vectors, dim={embeddings.shape[1]})")
    print(f"  metadata.json")
    print(f"  embeddings.npy")


if __name__ == "__main__":
    main()
