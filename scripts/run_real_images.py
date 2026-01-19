import os
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image

from src.sam_proposer import SamPointProposer
from src.clip_ranker import ClipRanker
from src.loop_runner import LoopRunner

def show_mask(ax, mask, color=(1,0,0,0.4)):
    m = mask.astype(np.float32)
    rgba = np.zeros((m.shape[0], m.shape[1], 4), dtype=np.float32)
    rgba[...,0] = color[0]
    rgba[...,1] = color[1]
    rgba[...,2] = color[2]
    rgba[...,3] = color[3] * m
    ax.imshow(rgba)

def save_overlay(save_path, image_pil, text, pred_mask, title):
    fig, ax = plt.subplots(1, 1, figsize=(7, 5))
    ax.imshow(image_pil)
    show_mask(ax, pred_mask, color=(1,0,0,0.4))
    ax.set_title(f"{title}\n{text}", fontsize=10)
    ax.axis("off")
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.tight_layout()
    plt.savefig(save_path, dpi=200)
    plt.close()

def run_one(image_pil, text, runner, max_iter, use_vlm):
    runner.max_iter = max_iter
    runner.use_vlm = use_vlm
    hist = runner.run(image_pil, text, debug=True)
    pred = hist[-1]["pred"]["mask"].astype(bool)
    conf = float(hist[-1]["conf"])
    gap = float(hist[-1]["gap"])
    iters = len(hist)
    return pred, conf, gap, iters

def main():
    cases = [
        ("assets/real/real1.png", "leftmost dog"),
        ("assets/real/real2.png", "the cat in black "),
        ("assets/real/real3.png", "the green ball "),
    ]

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

    outdir = "figures/app_real"
    for path, text in cases:
        image = Image.open(path).convert("RGB")

        pred1, conf1, gap1, it1 = run_one(image, text, runner, max_iter=1, use_vlm=False)
        save_overlay(
            os.path.join(outdir, os.path.splitext(os.path.basename(path))[0] + "_oneshot.png"),
            image, text, pred1,
            title=f"one-shot | iters={it1} | conf={conf1:.3f} | gap={gap1:.3f}"
        )

        pred2, conf2, gap2, it2 = run_one(image, text, runner, max_iter=2, use_vlm=True)
        save_overlay(
            os.path.join(outdir, os.path.splitext(os.path.basename(path))[0] + "_loop_iter2.png"),
            image, text, pred2,
            title=f"loop+VLM | iters={it2} | conf={conf2:.3f} | gap={gap2:.3f}"
        )

        print("saved:", path)

if __name__ == "__main__":
    main()
