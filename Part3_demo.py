"""
Part 3 – Zero-Shot CLIP Retrieval
demo.py: Visual demo for text-to-image and image-to-image retrieval.

Usage:
    # Text query
    python Part3_demo.py \
        --index_dir clip_index \
        --query "a dog running in a park" \
        --mode text

    # Image query
    python Part3_demo.py \
        --index_dir clip_index \
        --query /path/to/query_image.jpg \
        --mode image

    # Evaluate Recall@K
    python Part3_demo.py \
        --index_dir clip_index \
        --mode eval \
        --eval_queries queries.json
"""

import argparse
import json
import os
import math

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from PIL import Image

from Part3_retrieval import CLIPRetrieval


def load_image_safe(path: str, size: tuple = (224, 224)):
    try:
        return Image.open(path).convert("RGB").resize(size)
    except Exception:
        return Image.fromarray(np.zeros((*size, 3), dtype=np.uint8))


def plot_results(results: list, query, mode: str,
                 output_path: str = "retrieval_results.png",
                 top_k: int = 10):
    """
    Plot top-K retrieval results in a grid.
    For text queries: title = query string.
    For image queries: first panel = query image.
    """
    cols    = min(5, top_k)
    rows    = math.ceil(top_k / cols)
    n_extra = 1 if mode == "image" else 0

    total_panels = top_k + n_extra
    total_cols   = min(cols + n_extra, 6)
    total_rows   = math.ceil(total_panels / total_cols)

    fig, axes = plt.subplots(total_rows, total_cols,
                              figsize=(4 * total_cols, 4 * total_rows))
    axes = np.array(axes).flatten()

    panel = 0
    if mode == "image":
        query_img = load_image_safe(query)
        axes[0].imshow(query_img)
        axes[0].set_title("QUERY", fontsize=11, fontweight="bold",
                           color="red")
        axes[0].axis("off")
        panel = 1

    for res in results[:top_k]:
        img = load_image_safe(res["path"])
        axes[panel].imshow(img)
        title = f"#{res['rank']}  {res['label']}\n{res['score']:.3f}"
        axes[panel].set_title(title, fontsize=8)
        axes[panel].axis("off")
        panel += 1

    for ax in axes[panel:]:
        ax.axis("off")

    query_str = f'"{query}"' if mode == "text" else os.path.basename(query)
    fig.suptitle(f"CLIP {'Text' if mode=='text' else 'Image'}-to-Image Retrieval\n"
                  f"Query: {query_str}", fontsize=13, fontweight="bold")
    plt.tight_layout(rect=[0, 0, 1, 0.95])
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Results saved → {output_path}")


def print_results_table(results: list):
    print(f"\n{'Rank':<6} {'Score':<8} {'Label':<20} {'Path'}")
    print("-" * 75)
    for r in results:
        path_short = r["path"] if len(r["path"]) < 45 else "..." + r["path"][-42:]
        print(f"{r['rank']:<6} {r['score']:<8.4f} {r['label']:<20} {path_short}")


def get_args():
    p = argparse.ArgumentParser()
    p.add_argument("--index_dir",   type=str, required=True)
    p.add_argument("--model_name",  type=str, default="ViT-B/32")
    p.add_argument("--mode",        type=str, default="text",
                   choices=["text", "image", "eval"])
    p.add_argument("--query",       type=str, default=None,
                   help="Text string (mode=text) or image path (mode=image)")
    p.add_argument("--top_k",       type=int, default=10)
    p.add_argument("--output",      type=str, default="retrieval_results.png")
    p.add_argument("--eval_queries",type=str, default=None,
                   help="JSON file: [{'query': str, 'label': str}, ...]")
    return p.parse_args()


def main():
    args   = get_args()
    engine = CLIPRetrieval(args.index_dir, args.model_name, top_k=args.top_k)

    if args.mode == "text":
        assert args.query, "--query text string required"
        print(f"\nText query: \"{args.query}\"")
        results = engine.text_to_image(args.query)
        print_results_table(results)
        plot_results(results, args.query, mode="text",
                     output_path=args.output, top_k=args.top_k)

    elif args.mode == "image":
        assert args.query, "--query image path required"
        print(f"\nImage query: {args.query}")
        results = engine.image_to_image(args.query)
        print_results_table(results)
        plot_results(results, args.query, mode="image",
                     output_path=args.output, top_k=args.top_k)

    elif args.mode == "eval":
        assert args.eval_queries, "--eval_queries JSON file required"
        with open(args.eval_queries) as f:
            pairs = json.load(f)
        queries = [p["query"] for p in pairs]
        labels  = [p["label"] for p in pairs]
        recalls = engine.evaluate_recall(queries, labels, k_values=[1, 5, 10])
        print("\n===== Recall@K Evaluation =====")
        for k, v in recalls.items():
            print(f"  {k}: {v*100:.2f}%")
        with open("recall_results.json", "w") as f:
            json.dump(recalls, f, indent=2)


if __name__ == "__main__":
    main()
