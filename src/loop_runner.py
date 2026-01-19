import os
import numpy as np
from src.vlm_policy import VLMPolicy


class LoopRunner:
    def __init__(
        self,
        proposer,
        ranker,
        max_iter=3,
        conf_thr=0.25,
        gap_thr=0.03,
        cache_proposals=True,
        vlm_topk=3,
        use_vlm=True,
    ):
        self.proposer = proposer
        self.ranker = ranker
        self.max_iter = max_iter
        self.conf_thr = conf_thr
        self.gap_thr = gap_thr
        self.cache_proposals = cache_proposals
        self.vlm_topk = vlm_topk
        self.use_vlm = use_vlm

        self.vlm = None
        if self.use_vlm:
            self.vlm = VLMPolicy(api_key=os.environ.get("SILICONFLOW_API_KEY"))

    def run(self, image_pil, query, debug=False):
        image_np = np.array(image_pil)
        W, H = image_pil.size

        cur_query = query
        history = []

        if debug:
            print("cache_proposals =", self.cache_proposals)

        cached_props = None
        if self.cache_proposals:
            cached_props = self.proposer.propose(image_np)
            if debug and hasattr(self.proposer, "calls"):
                print("after cache propose calls =", self.proposer.calls)

        for it in range(self.max_iter):
            props = cached_props if cached_props is not None else self.proposer.propose(image_np)

            if debug and hasattr(self.proposer, "calls"):
                print(f"iter {it}: proposer.calls =", self.proposer.calls)

            scored = self.ranker.rank(image_np, props, cur_query)
            top1 = scored[0]
            top2 = scored[1] if len(scored) > 1 else None

            conf = float(top1["score"])
            gap = conf - (float(top2["score"]) if top2 is not None else -1e9)

            history.append(
                {
                    "iter": it,
                    "query": cur_query,
                    "conf": conf,
                    "gap": gap,
                    "pred": top1,
                }
            )

            if debug:
                print(f"iter {it} | query='{cur_query}' | conf={conf:.3f} | gap={gap:.3f}")

            if conf >= self.conf_thr and gap >= self.gap_thr:
                break

            if it == self.max_iter - 1:
                break

            if self.use_vlm and self.vlm is not None:
                try:
                    new_query = self.vlm.refine_query(
                        cur_query,
                        scored[: self.vlm_topk],
                        (W, H),
                        topk=self.vlm_topk,
                    )
                except Exception as e:
                    if debug:
                        print("VLM error:", repr(e))
                    new_query = cur_query

                if debug:
                    if new_query != cur_query:
                        print(f"VLM refined -> '{new_query}'")
                    else:
                        print("VLM refined -> (no change)")

                cur_query = new_query
            else:
                if debug:
                    print("VLM disabled -> keep query")

        return history
