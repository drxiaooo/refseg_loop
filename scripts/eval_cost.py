import time
import numpy as np
from datasets import load_dataset
from pycocotools import mask as coco_mask
from tqdm import tqdm

from src.sam_proposer import GridProposer
from src.clip_ranker import ClipRanker
from src.loop_runner import LoopRunner

def decode_rle(rle_dict):
    return coco_mask.decode(rle_dict).astype(bool)

def main(n=50):
    ds = load_dataset("moondream/refcoco-m", split="validation")
    proposer = GridProposer(grid=6)
    ranker = ClipRanker(device="cpu")
    runner = LoopRunner(proposer, ranker, max_iter=3, conf_thr=0.25, gap_thr=0.03)

    t0 = time.time()
    start_calls = ranker.roi_calls
    total_iters = 0

    for i in tqdm(range(min(len(ds), n))):
        item = ds[i]
        image = item["image"]
        sample = item["samples"][0]
        text = sample["sentences"][0]
        hist = runner.run(image, text)
        total_iters += len(hist)

    dt = time.time() - t0
    roi_calls = ranker.roi_calls - start_calls

    print(f"n={n}")
    print(f"total_time_sec={dt:.2f}, avg_time_per_img={dt/n:.2f}")
    print(f"avg_iters={total_iters/n:.2f}")
    print(f"clip_roi_calls={roi_calls}, avg_roi_per_img={roi_calls/n:.1f}")

if __name__ == "__main__":
    main()
