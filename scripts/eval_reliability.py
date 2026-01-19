import numpy as np
from datasets import load_dataset
from pycocotools import mask as coco_mask
from tqdm import tqdm

from src.sam_proposer import SamPointProposer
from src.clip_ranker import ClipRanker
from src.loop_runner import LoopRunner
from src.eval_metrics import iou_mask
from src.perturb import add_gaussian_noise, gaussian_blur


def decode_rle(rle_dict):
    return coco_mask.decode(rle_dict).astype(bool)


def area_bucket_by_ratio(mask: np.ndarray):
    H, W = mask.shape
    r = float(mask.sum()) / float(H * W + 1e-6)
    if r < 0.05:
        return "small"
    elif r < 0.20:
        return "medium"
    else:
        return "large"


def run_method(runner: LoopRunner, img_np: np.ndarray, image_pil, text: str):
    from PIL import Image
    pil = Image.fromarray(img_np)
    hist = runner.run(pil, text, debug=False)
    pred = hist[-1]["pred"]["mask"].astype(bool)
    return pred


def main(n=100):
    ds = load_dataset("moondream/refcoco-m", split="validation")

    proposer = SamPointProposer(
        checkpoint_path="checkpoints/sam_vit_b_01ec64.pth",
        model_type="vit_b",
        device="cpu",
        points_per_side=4,
        multimask=True,
    )
    ranker = ClipRanker(device="cpu")

    runner_oneshot = LoopRunner(
        proposer, ranker,
        max_iter=1,
        conf_thr=0.25,
        gap_thr=0.03,
        cache_proposals=True,
        use_vlm=False,
    )
    runner_loop = LoopRunner(
        proposer, ranker,
        max_iter=2,
        conf_thr=0.25,
        gap_thr=0.03,
        cache_proposals=True,
        use_vlm=True,
    )

    settings = [
        ("clean", None),
        ("noise10", lambda x: add_gaussian_noise(x, sigma=10)),
        ("noise20", lambda x: add_gaussian_noise(x, sigma=20)),
        ("blur5", lambda x: gaussian_blur(x, k=5)),
        ("blur9", lambda x: gaussian_blur(x, k=9)),
    ]

    rob = {
        "one-shot": {name: [] for name, _ in settings},
        "loop+VLM": {name: [] for name, _ in settings},
    }

    buckets = {
        "one-shot": {"small": [], "medium": [], "large": []},
        "loop+VLM": {"small": [], "medium": [], "large": []},
    }

    for i in tqdm(range(min(len(ds), n))):
        item = ds[i]
        image = item["image"]
        sample = item["samples"][0]
        text = sample["sentences"][0]
        gt = decode_rle(sample["mask"])

        img_np = np.array(image)
        b = area_bucket_by_ratio(gt)

        pred_clean_one = run_method(runner_oneshot, img_np, image, text)
        buckets["one-shot"][b].append(float(iou_mask(pred_clean_one, gt)))

        pred_clean_loop = run_method(runner_loop, img_np, image, text)
        buckets["loop+VLM"][b].append(float(iou_mask(pred_clean_loop, gt)))

        for name, tfm in settings:
            test_np = img_np if tfm is None else tfm(img_np)

            pred1 = run_method(runner_oneshot, test_np, image, text)
            rob["one-shot"][name].append(float(iou_mask(pred1, gt)))

            pred2 = run_method(runner_loop, test_np, image, text)
            rob["loop+VLM"][name].append(float(iou_mask(pred2, gt)))

    print("\n=== Robustness (mIoU) ===")
    for method in ["one-shot", "loop+VLM"]:
        print(f"\n[{method}]")
        for name, _ in settings:
            v = float(np.mean(rob[method][name])) if len(rob[method][name]) else 0.0
            print(f"{name:8s}: {v:.4f}")

    print("\n=== Size buckets on clean (mIoU) ===")
    for method in ["one-shot", "loop+VLM"]:
        print(f"\n[{method}]")
        for k in ["small", "medium", "large"]:
            arr = buckets[method][k]
            v = float(np.mean(arr)) if len(arr) else 0.0
            print(f"{k:6s}: {v:.4f} (n={len(arr)})")


if __name__ == "__main__":
    main(n=50)
