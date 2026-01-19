import numpy as np
from datasets import load_dataset
from pycocotools import mask as coco_mask
from tqdm import tqdm

from src.sam_proposer import SamPointProposer
from src.clip_ranker import ClipRanker
from src.eval_metrics import iou_mask

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

    print("PROPOSER_CLASS =", proposer.__class__.__name__)

    total = 0.0
    count = 0

    for i in tqdm(range(min(len(ds), n))):
        item = ds[i]
        image = item["image"]
        sample = item["samples"][0]
        text = sample["sentences"][0]
        gt = decode_rle(sample["mask"])

        image_np = np.array(image)
        props = proposer.propose(image_np)
        scored = ranker.rank(image_np, props, text)
        pred = scored[0]["mask"].astype(bool)

        total += iou_mask(pred, gt)
        count += 1

    print(f"n={count}, proposer={proposer.__class__.__name__}, mIoU={total/count:.4f}")

if __name__ == "__main__":
    main()
