import os
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from PIL import Image
from datasets import load_dataset
from pycocotools import mask as coco_mask

from src.sam_proposer import SamPointProposer
from src.clip_ranker import ClipRanker
from src.loop_runner import LoopRunner

def decode_rle(rle_dict):
    return coco_mask.decode(rle_dict).astype(bool)

def show_mask(ax, mask, color=(1,0,0,0.4)):
    m = mask.astype(np.float32)
    rgba = np.zeros((m.shape[0], m.shape[1], 4), dtype=np.float32)
    rgba[...,0] = color[0]
    rgba[...,1] = color[1]
    rgba[...,2] = color[2]
    rgba[...,3] = color[3] * m
    ax.imshow(rgba)

def show_box(ax, bbox_xyxy, edgecolor="r"):
    if bbox_xyxy is None:
        return
    x1, y1, x2, y2 = bbox_xyxy
    rect = patches.Rectangle((x1, y1), x2-x1, y2-y1, linewidth=2, edgecolor=edgecolor, facecolor="none")
    ax.add_patch(rect)

def run_once(ds, idx, runner, max_iter, use_vlm):
    item = ds[idx]
    image = item["image"]
    sample = item["samples"][0]
    text = sample["sentences"][0]
    gt = decode_rle(sample["mask"])
    runner.max_iter = max_iter
    runner.use_vlm = use_vlm
    hist = runner.run(image, text, debug=False)
    pred = hist[-1]["pred"]["mask"].astype(bool)
    bbox = hist[-1]["pred"].get("bbox_xyxy", None)
    conf = float(hist[-1]["conf"])
    gap = float(hist[-1]["gap"])
    iters = len(hist)
    return image, text, gt, pred, bbox, conf, gap, iters

def make_figure(save_path, image, text, gt, pred, bbox, title_suffix):
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))

    axes[0].imshow(image)
    axes[0].set_title(f"Input\n{text}", fontsize=10)
    axes[0].axis("off")

    axes[1].imshow(image)
    show_mask(axes[1], gt, color=(0,1,0,0.4))
    axes[1].set_title("GT (green)")
    axes[1].axis("off")

    axes[2].imshow(image)
    show_mask(axes[2], pred, color=(1,0,0,0.4))
    show_box(axes[2], bbox, edgecolor="r")
    axes[2].set_title(f"Pred (red)\n{title_suffix}", fontsize=10)
    axes[2].axis("off")

    plt.tight_layout()
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.savefig(save_path, dpi=200)
    plt.close()

def main():
    ds = load_dataset("moondream/refcoco-m", split="validation")

    proposer = SamPointProposer(
        checkpoint_path="checkpoints/sam_vit_b_01ec64.pth",
        model_type="vit_b",
        device="cpu",
        points_per_side=4,
        multimask=True,
    )
    ranker = ClipRanker(device="cpu")
    runner = LoopRunner(
        proposer, ranker,
        max_iter=2,
        conf_thr=0.25,
        gap_thr=0.03,
        cache_proposals=True,
        use_vlm=True,
    )

    indices = [33, 24, 5, 10, 6, 9]

    outdir = "figures/app_refcocom"
    for idx in indices:
        image, text, gt, pred, bbox, conf, gap, iters = run_once(ds, idx, runner, max_iter=1, use_vlm=False)
        make_figure(
            os.path.join(outdir, f"idx{idx}_oneshot.png"),
            image, text, gt, pred, bbox,
            title_suffix=f"one-shot | iters={iters} | conf={conf:.3f} | gap={gap:.3f}"
        )

        image, text, gt, pred, bbox, conf, gap, iters = run_once(ds, idx, runner, max_iter=2, use_vlm=True)
        make_figure(
            os.path.join(outdir, f"idx{idx}_loop_iter2.png"),
            image, text, gt, pred, bbox,
            title_suffix=f"loop+VLM | iters={iters} | conf={conf:.3f} | gap={gap:.3f}"
        )

        print("saved:", idx)

if __name__ == "__main__":
    main()
