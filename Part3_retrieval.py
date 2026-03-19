"""
Part 3 – Zero-Shot CLIP Retrieval
retrieval.py: Core retrieval engine.

Supports:
  - text_to_image(query: str)  → top-K images
  - image_to_image(img_path)   → top-K similar images
  - batch evaluation with Recall@K
"""

import os
import json
import numpy as np
import torch
import faiss
from PIL import Image
import open_clip

from Part3_build_index import load_clip_model, SUPPORTED_CLIP_MODELS


class CLIPRetrieval:
    """
    Zero-shot cross-modal retrieval engine backed by CLIP + FAISS.

    Args:
        index_dir  : directory produced by Part3_build_index.py
        model_name : CLIP variant (must match index)
        top_k      : number of results to return
    """

    def __init__(self, index_dir: str, model_name: str = "ViT-B/32",
                 top_k: int = 10):
        self.top_k      = top_k
        self.index_dir  = index_dir
        self.device     = "cuda" if torch.cuda.is_available() else "cpu"

        # Load FAISS index + metadata
        self.index = faiss.read_index(os.path.join(index_dir, "index.faiss"))
        with open(os.path.join(index_dir, "metadata.json")) as f:
            meta = json.load(f)
        self.paths   = meta["paths"]
        self.labels  = meta["labels"]
        self.dim     = meta["dim"]

        # Load CLIP model
        self.model, self.preprocess = load_clip_model(model_name, self.device)
        self.tokenizer = open_clip.get_tokenizer(
            SUPPORTED_CLIP_MODELS[model_name][0]
        )
        print(f"CLIPRetrieval ready | {len(self.paths)} images | "
              f"dim={self.dim} | device={self.device}")

    # ──────────────────────────────────────────
    #  Text → Image
    # ──────────────────────────────────────────
    @torch.no_grad()
    def text_to_image(self, query: str) -> list:
        """
        Encode a text query and retrieve the top-K most similar images.

        Returns:
            list of dicts: {"path", "label", "score", "rank"}
        """
        tokens = self.tokenizer([query]).to(self.device)
        feat   = self.model.encode_text(tokens)
        feat   = feat / feat.norm(dim=-1, keepdim=True)
        feat_np = feat.cpu().numpy().astype(np.float32)

        scores, indices = self.index.search(feat_np, self.top_k)
        return self._format_results(indices[0], scores[0])

    # ──────────────────────────────────────────
    #  Image → Image
    # ──────────────────────────────────────────
    @torch.no_grad()
    def image_to_image(self, image_path: str,
                        exclude_self: bool = True) -> list:
        """
        Encode a query image and retrieve the top-K most similar gallery images.

        Returns:
            list of dicts: {"path", "label", "score", "rank"}
        """
        img   = Image.open(image_path).convert("RGB")
        tensor = self.preprocess(img).unsqueeze(0).to(self.device)
        feat   = self.model.encode_image(tensor)
        feat   = feat / feat.norm(dim=-1, keepdim=True)
        feat_np = feat.cpu().numpy().astype(np.float32)

        k = self.top_k + 1 if exclude_self else self.top_k
        scores, indices = self.index.search(feat_np, k)
        results = self._format_results(indices[0], scores[0])

        if exclude_self:
            results = [r for r in results if r["path"] != image_path][:self.top_k]

        return results

    # ──────────────────────────────────────────
    #  Batch Recall@K evaluation
    # ──────────────────────────────────────────
    @torch.no_grad()
    def evaluate_recall(self, queries: list, ground_truth_labels: list,
                         k_values: list = [1, 5, 10]) -> dict:
        """
        Compute Recall@K for a list of text queries.

        Args:
            queries            : list of text strings
            ground_truth_labels: list of true class labels (matching gallery labels)
            k_values           : list of K values

        Returns:
            dict: {k: recall_at_k}
        """
        tokens = self.tokenizer(queries).to(self.device)
        feats  = self.model.encode_text(tokens)
        feats  = feats / feats.norm(dim=-1, keepdim=True)
        feats_np = feats.cpu().numpy().astype(np.float32)

        max_k = max(k_values)
        _, indices = self.index.search(feats_np, max_k)
        recalls = {}
        for k in k_values:
            hits = 0
            for i, gt_label in enumerate(ground_truth_labels):
                retrieved_labels = [self.labels[idx] for idx in indices[i, :k]]
                if gt_label in retrieved_labels:
                    hits += 1
            recalls[f"R@{k}"] = hits / len(queries)
        return recalls

    def _format_results(self, indices, scores):
        results = []
        for rank, (idx, score) in enumerate(zip(indices, scores), start=1):
            if 0 <= idx < len(self.paths):
                results.append({
                    "rank":  rank,
                    "path":  self.paths[idx],
                    "label": self.labels[idx],
                    "score": float(score),
                })
        return results
