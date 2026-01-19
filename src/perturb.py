import numpy as np
import cv2

def add_gaussian_noise(image_np, sigma=10.0):
    noise = np.random.normal(0, sigma, image_np.shape).astype(np.float32)
    out = np.clip(image_np.astype(np.float32) + noise, 0, 255).astype(np.uint8)
    return out

def gaussian_blur(image_np, k=5):
    k = int(k)
    if k % 2 == 0:
        k += 1
    return cv2.GaussianBlur(image_np, (k, k), 0)
