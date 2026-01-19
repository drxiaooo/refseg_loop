import os
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches

from datasets import load_dataset
from pycocotools import mask as coco_mask
from PIL import Image

from src.sam_proposer import SamPointProposer
from src.clip_ranker import ClipRanker
from src.loop_runner import LoopRunner


def decode_rle(rle_dict):
    return coco_mask.decode(rle_dict).astype(bool)


def draw_mask(ax, mask, color):
    h, w = mask.shape
    rgba = np.zeros((h, w, 4))
    rgba[..., :3] = color[:3]
    rgba[..., 3] = mask * color[3]
    ax.imshow(rgba)


def visualize(image, gt, pred1, pred2, save_path, title):
    fig, axes = plt.subplots(1, 4, figsize=(18, 5))

    axes[0].imshow(image)
    axes[0].set_title("Image")
    axes[0].axis("off")

    axes[1].imshow(image)
    draw_mask(axes[1], gt, (0, 1, 0, 0.5))
    axes[1].set_title("GT")
    axes[1].axis("off")

    axes[2].imshow(image)
    draw_mask(axes[2], pred1, (1, 0, 0, 0.5))
    axes[2].set_title("One-shot")
    axes[2].axis("off")

    axes[3].imshow(image)
    draw_mask(axes[3], pred2, (1, 0, 0, 0.5))
    axes[3].set_title("Loop + VLM")
    axes[3].axis("off")

    plt.suptitle(title)
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()


def main(num_cases=10):
    os.makedirs("figures", exist_ok=True)

    ds = load_dataset("moondream/refcoco-m", split="validation")

    proposer = SamPointProposer(
        checkpoint_path="checkpoints/sam_vit_b_01ec64.pth",
        model_type="vit_b",
        device="cpu",
        points_per_side=4,
        multimask=True,
    )

    ranker = ClipRanker(device="cpu")

    # one-shot
    runner_base = LoopRunner(
        proposer,
        ranker,
        max_iter=1,
        use_vlm=False,
    )

    # loop + VLM 
    runner_loop = LoopRunner(
        proposer,
        ranker,
        max_iter=2,
        use_vlm=True,
    )

    saved = 0

    for i in range(len(ds)):
        item = ds[i]
        image = item["image"]
        sample = item["samples"][0]
        text = sample["sentences"][0]
        gt = decode_rle(sample["mask"])

        hist1 = runner_base.run(image, text)
        pred1 = hist1[-1]["pred"]["mask"].astype(bool)

        hist2 = runner_loop.run(image, text)
        pred2 = hist2[-1]["pred"]["mask"].astype(bool)

     
        iou1 = (pred1 & gt).sum() / ((pred1 | gt).sum() + 1e-6)
        iou2 = (pred2 & gt).sum() / ((pred2 | gt).sum() + 1e-6)

        if abs(iou2 - iou1) < 0.1:
            continue

        save_path = f"figures/case_{saved}.png"
        title = f"text: '{text}' | one-shot={iou1:.2f}, loop={iou2:.2f}"
        visualize(image, gt, pred1, pred2, save_path, title)

        print("saved:", save_path)

        saved += 1
        if saved >= num_cases:
            break


if __name__ == "__main__":
    main(num_cases=5)
