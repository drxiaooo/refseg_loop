import os
import time
import numpy as np
from datasets import load_dataset
from pycocotools import mask as coco_mask
from tqdm import tqdm

from src.sam_proposer import SamPointProposer
from src.clip_ranker import ClipRanker
from src.eval_metrics import iou_mask
from src.loop_runner import LoopRunner


def decode_rle(rle_dict):
    return coco_mask.decode(rle_dict).astype(bool)


def main(n=50, max_iter=3, use_vlm=True):
    ds = load_dataset("moondream/refcoco-m", split="validation")

    proposer = SamPointProposer(
        checkpoint_path="checkpoints/sam_vit_b_01ec64.pth",
        model_type="vit_b",
        device="cpu",
        points_per_side=4,
        multimask=True,
    )
    print("PROPOSER_CLASS =", proposer.__class__.__name__)

    ranker = ClipRanker(device="cpu")
    runner = LoopRunner(
        proposer,
        ranker,
        max_iter=max_iter,
        conf_thr=0.25,
        gap_thr=0.03,
        cache_proposals=True,
        use_vlm=use_vlm,
    )

    
    total_iou = 0.0
    total_iters = 0
    ious = []  


    sam_calls_before = getattr(proposer, "calls", 0)
    clip_roi_before = getattr(ranker, "roi_calls", 0)

    total_vlm_calls = 0  

   
    t0 = time.perf_counter()

    for i in tqdm(range(min(len(ds), n))):
        item = ds[i]
        image = item["image"]
        sample = item["samples"][0]
        text = sample["sentences"][0]
        gt = decode_rle(sample["mask"])

        hist = runner.run(image, text, debug=(i == 0))
        pred = hist[-1]["pred"]["mask"].astype(bool)

        iou = float(iou_mask(pred, gt))
        ious.append(iou)
        total_iou += iou
        total_iters += len(hist)

        
        if use_vlm:
            total_vlm_calls += max(0, len(hist) - 1)

    t1 = time.perf_counter()

   
    used_n = min(len(ds), n)
    mean_iou = total_iou / max(used_n, 1)
    avg_iters = total_iters / max(used_n, 1)

    total_time_sec = t1 - t0
    avg_time_per_img = total_time_sec / max(used_n, 1)

    print(f"n={used_n}, mIoU={mean_iou:.4f}, avg_iters={avg_iters:.2f}")
    print(f"total_time_sec={total_time_sec:.2f}, avg_time_per_img={avg_time_per_img:.2f}")

    
    sam_calls_after = getattr(proposer, "calls", None)
    clip_roi_after = getattr(ranker, "roi_calls", None)

   
    total_sam_calls = None
    sam_calls_per_img = None
    if sam_calls_after is not None:
        total_sam_calls = sam_calls_after - sam_calls_before
        sam_calls_per_img = total_sam_calls / max(used_n, 1)

    
    total_clip_roi = None
    clip_roi_per_img = None
    if clip_roi_after is not None:
        total_clip_roi = clip_roi_after - clip_roi_before
        clip_roi_per_img = total_clip_roi / max(used_n, 1)

    vlm_calls_per_img = total_vlm_calls / max(used_n, 1)

    print("=== Tool cost (counts) ===")
    print(f"use_vlm={int(use_vlm)}, max_iter={max_iter}")
    print(f"sam_calls_total={total_sam_calls}, sam_calls_per_img={sam_calls_per_img}")
    print(f"clip_roi_calls_total={total_clip_roi}, clip_roi_per_img={clip_roi_per_img:.1f}")
    print(f"vlm_calls_total={total_vlm_calls}, vlm_calls_per_img={vlm_calls_per_img:.2f}")

    
    os.makedirs("metrics", exist_ok=True)
    tag = f"loop_vlm{int(use_vlm)}_iter{max_iter}_n{used_n}"
    save_path = os.path.join("metrics", f"ious_{tag}.npy")
    np.save(save_path, np.array(ious, dtype=np.float32))
    print("saved:", save_path)


if __name__ == "__main__":
    import argparse

    p = argparse.ArgumentParser()
    p.add_argument("--n", type=int, default=50)
    p.add_argument("--max_iter", type=int, default=3)
    p.add_argument("--use_vlm", type=int, default=1)  # 1/0
    args = p.parse_args()

    main(n=args.n, max_iter=args.max_iter, use_vlm=bool(args.use_vlm))
