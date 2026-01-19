import torch
import open_clip
from PIL import Image

class ClipRanker:
    def __init__(self, device="cpu"):
        self.device = device

        self.roi_calls = 0

        self.model, _, self.preprocess = open_clip.create_model_and_transforms(
            "ViT-B-32", pretrained="laion2b_s34b_b79k"
        )
        self.tokenizer = open_clip.get_tokenizer("ViT-B-32")
        self.model = self.model.to(device).eval()

    @torch.no_grad()
    def rank(self, image_np, proposals, text):
        self.roi_calls += len(proposals)

        text_tokens = self.tokenizer([text]).to(self.device)
        text_feat = self.model.encode_text(text_tokens)
        text_feat = text_feat / text_feat.norm(dim=-1, keepdim=True)

        scored = []
        for p in proposals:
            x1, y1, x2, y2 = map(int, p["bbox_xyxy"])
            roi = image_np[y1:y2, x1:x2, :]
            if roi.size == 0:
                score = -1e9
            else:
                img_in = self.preprocess(Image.fromarray(roi)).unsqueeze(0).to(self.device)
                img_feat = self.model.encode_image(img_in)
                img_feat = img_feat / img_feat.norm(dim=-1, keepdim=True)
                score = float((img_feat @ text_feat.T).squeeze().cpu().item())

            q = dict(p)
            q["score"] = score
            scored.append(q)

        scored.sort(key=lambda d: d["score"], reverse=True)
        return scored
