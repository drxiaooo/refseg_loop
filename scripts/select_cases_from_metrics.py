import numpy as np

def topk(a, k=5):
    idx = np.argsort(a)[::-1][:k]
    return [(int(i), float(a[i])) for i in idx]

def bottomk(a, k=5):
    idx = np.argsort(a)[:k]
    return [(int(i), float(a[i])) for i in idx]

iou1 = np.load("metrics/ious_loop_vlm1_iter1_n50.npy")
iou2 = np.load("metrics/ious_loop_vlm1_iter2_n50.npy")
iou3 = np.load("metrics/ious_loop_vlm1_iter3_n50.npy")

d21 = iou2 - iou1
d32 = iou3 - iou2

print("=== best improvements (iter2 - iter1) ===")
print(topk(d21, k=8))

print("\n=== worst degradations (iter3 - iter2) ===")
print(bottomk(d32, k=8))

print("\n=== near no-change (abs(iter2-iter1)) ===")
absd = np.abs(d21)
idx = np.argsort(absd)[:8]
print([(int(i), float(absd[i]), float(iou1[i]), float(iou2[i])) for i in idx])
