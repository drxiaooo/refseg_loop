import os
from openai import OpenAI


def _pos_bucket(cx, cy, w, h):
    if cx < w * 0.33:
        horiz = "left"
    elif cx > w * 0.66:
        horiz = "right"
    else:
        horiz = "center"

    if cy < h * 0.33:
        vert = "top"
    elif cy > h * 0.66:
        vert = "bottom"
    else:
        vert = "middle"
    return f"{horiz}-{vert}"


class VLMPolicy:
    def __init__(
        self,
        model="Qwen/Qwen2.5-VL-72B-Instruct",
        base_url="https://api.siliconflow.cn/v1",
        api_key=None,
    ):
        api_key = api_key or os.environ.get("SILICONFLOW_API_KEY")
        if not api_key:
            raise RuntimeError("Missing SILICONFLOW_API_KEY (env var).")
        self.client = OpenAI(api_key=api_key, base_url=base_url)
        self.model = model

    def refine_query(self, query, scored_props, img_size, topk=3):
        w, h = img_size
        lines = []
        for i, c in enumerate(scored_props[:topk]):
            pos = _pos_bucket(c["centroid"][0], c["centroid"][1], w, h)
            area_ratio = c["area"] / (w * h)
            lines.append(f"{i+1}. score={c['score']:.3f}, area={area_ratio*100:.1f}%, pos={pos}")

        prompt = (
            "You are assisting a referring image segmentation system.\n\n"
            f'Original query: "{query}"\n\n'
            "Top candidate regions (ranked by CLIP):\n"
            + "\n".join(lines)
            + "\n\n"
            "The scores are close, meaning the query may be ambiguous.\n"
            "Please refine the query to better distinguish the target object.\n"
            "Return ONLY the refined query text (no quotes, no explanations)."
        )

        resp = self.client.chat.completions.create(
            model=self.model,
            messages=[
                {"role": "system", "content": "You are a careful assistant that outputs only the refined query."},
                {"role": "user", "content": prompt},
            ],
            temperature=0.2,
            max_tokens=32,
        )
        new_q = resp.choices[0].message.content.strip()

        if not new_q or len(new_q) > 80:
            return query
        return new_q
