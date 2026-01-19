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

def main(n=50):
    ds = load_dataset("moondream/refcoco-m", split="validation")

    proposer = SamPointProposer(
        checkpoint_path="checkpoints/sam_vit_b_01ec64.pth",
        model_type="vit_b",
        device="cpu",
        points_per_side=4,
        multimask=True
    )
    ranker = ClipRanker(device="cpu")

    runner = LoopRunner(
        proposer,
        ranker,
        max_iter=3,
        conf_thr=0.25,
        gap_thr=0.03,
        cache_proposals=True,
        use_vlm=False,   # ✅ 关键：关闭 VLM
    )

    total_iou = 0.0
    total_iters = 0

    for i in tqdm(range(min(len(ds), n))):
        item = ds[i]
        image = item["image"]
        sample = item["samples"][0]
        text = sample["sentences"][0]
        gt = decode_rle(sample["mask"])

        hist = runner.run(image, text, debug=(i == 0))
        pred = hist[-1]["pred"]["mask"].astype(bool)

        total_iou += iou_mask(pred, gt)
        total_iters += len(hist)

    print(f"n={n}, mIoU={total_iou/n:.4f}, avg_iters={total_iters/n:.2f} (no-VLM loop)")

if __name__ == "__main__":
    main()
