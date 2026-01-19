import numpy as np
from datasets import load_dataset
from pycocotools import mask as coco_mask
from tqdm import tqdm

from src.sam_proposer import SamPointProposer
from src.clip_ranker import ClipRanker
from src.loop_runner import LoopRunner
from src.eval_metrics import iou_mask


def decode_rle(rle_dict):
    return coco_mask.decode(rle_dict).astype(bool)


def bucket_of(area_ratio: float):
    if area_ratio < 0.05:
        return "small"
    elif area_ratio < 0.20:
        return "medium"
    else:
        return "large"


def main(n=200):
    ds = load_dataset("moondream/refcoco-m", split="validation")

    proposer = SamPointProposer(
        checkpoint_path="checkpoints/sam_vit_b_01ec64.pth",
        model_type="vit_b",
        device="cpu",
        points_per_side=4,
        multimask=True,
    )
    ranker = ClipRanker(device="cpu")

    # One-shot baseline
    runner_base = LoopRunner(
        proposer, ranker,
        max_iter=1,
        cache_proposals=True,
        use_vlm=False,
    )

    # Best loop setting : iter=2 + VLM
    runner_loop = LoopRunner(
        proposer, ranker,
        max_iter=2,
        cache_proposals=True,
        use_vlm=True,
    )

    stats = {
        "small": {"n": 0, "sum_iou_base": 0.0, "sum_iou_loop": 0.0},
        "medium": {"n": 0, "sum_iou_base": 0.0, "sum_iou_loop": 0.0},
        "large": {"n": 0, "sum_iou_base": 0.0, "sum_iou_loop": 0.0},
    }

    for i in tqdm(range(min(len(ds), n))):
        item = ds[i]
        image = item["image"]
        sample = item["samples"][0]
        text = sample["sentences"][0]
        gt = decode_rle(sample["mask"])

        H, W = gt.shape
        area_ratio = float(gt.sum()) / float(H * W + 1e-6)
        b = bucket_of(area_ratio)

        # one-shot
        hist_b = runner_base.run(image, text, debug=(i == 0))
        pred_b = hist_b[-1]["pred"]["mask"].astype(bool)
        iou_b = float(iou_mask(pred_b, gt))

        # loop+VLM
        hist_l = runner_loop.run(image, text, debug=False)
        pred_l = hist_l[-1]["pred"]["mask"].astype(bool)
        iou_l = float(iou_mask(pred_l, gt))

        stats[b]["n"] += 1
        stats[b]["sum_iou_base"] += iou_b
        stats[b]["sum_iou_loop"] += iou_l

    print("\n=== Size buckets (mIoU) ===")
    for b in ["small", "medium", "large"]:
        n_b = stats[b]["n"]
        if n_b == 0:
            print(f"{b:6s}: n=0")
            continue
        miou_base = stats[b]["sum_iou_base"] / n_b
        miou_loop = stats[b]["sum_iou_loop"] / n_b
        print(f"{b:6s}: n={n_b:3d} | one-shot={miou_base:.4f} | loop+VLM(iter2)={miou_loop:.4f}")


if __name__ == "__main__":
    main(n=200)
