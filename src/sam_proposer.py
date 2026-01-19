import numpy as np
import torch
from segment_anything import sam_model_registry, SamPredictor

def _mask_to_bbox_xyxy(mask: np.ndarray):
    ys, xs = np.where(mask > 0)
    if len(xs) == 0 or len(ys) == 0:
        return None
    x1, x2 = int(xs.min()), int(xs.max())
    y1, y2 = int(ys.min()), int(ys.max())
    return (x1, y1, x2, y2)

class SamPointProposer:
    def __init__(self, checkpoint_path: str, model_type="vit_b", device="cpu", points_per_side=4, multimask=True):
        self.device = device
        self.points_per_side = points_per_side
        self.multimask = multimask
        self.calls = 0


        sam = sam_model_registry[model_type](checkpoint=checkpoint_path)
        sam.to(device=device)
        sam.eval()
        self.predictor = SamPredictor(sam)

    @torch.no_grad()
    def propose(self, image_np: np.ndarray):
        self.calls += 1

        H, W, _ = image_np.shape
        self.predictor.set_image(image_np)

        xs = np.linspace(0.1 * W, 0.9 * W, self.points_per_side)
        ys = np.linspace(0.1 * H, 0.9 * H, self.points_per_side)

        proposals = []
        for y in ys:
            for x in xs:
                point = np.array([[x, y]], dtype=np.float32)
                label = np.array([1], dtype=np.int32)

                masks, _, _ = self.predictor.predict(
                    point_coords=point,
                    point_labels=label,
                    multimask_output=self.multimask,
                )

                for m in masks:
                    m = (m > 0).astype(np.uint8)
                    bbox = _mask_to_bbox_xyxy(m)
                    if bbox is None:
                        continue
                    x1, y1, x2, y2 = bbox
                    area = int(m.sum())
                    if area < 50:
                        continue
                    proposals.append({
                        "mask": m,
                        "bbox_xyxy": bbox,
                        "centroid": ((x1 + x2) / 2.0, (y1 + y2) / 2.0),
                        "area": area,
                    })

        uniq = {}
        for p in proposals:
            key = (p["bbox_xyxy"], p["area"] // 100)
            if key not in uniq:
                uniq[key] = p
        out = list(uniq.values())

        if len(out) == 0:
            out = [{
                "mask": np.ones((H, W), dtype=np.uint8),
                "bbox_xyxy": (0, 0, W - 1, H - 1),
                "centroid": (W / 2.0, H / 2.0),
                "area": H * W
            }]

        return out
