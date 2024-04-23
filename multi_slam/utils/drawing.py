import time

import cv2
import matplotlib.cm as cm
import numpy as np
from einops import *
from imageio.v3 import imwrite

GUTTER = 20

def make_matching_plot(path, image1, image2, pts1, pts2, conf, sel, text=[], showing_gt=False):
    H, W, _ = image1.shape
    N, _ = pts1.shape
    bit = np.full(2, 3, dtype=np.int64)
    combined = np.zeros((H, W*2 + GUTTER, 3), dtype=np.uint8)
    combined[:, :W] = image1
    combined[:, -W:] = image2
    offset = np.array([W+GUTTER, 0])
    for j, (p1, p2) in enumerate(zip(pts1.astype(np.int64), pts2.astype(np.int64))):
        if j not in sel:
            continue
        if not showing_gt:
            color = np.asarray(cm.jet(conf[j]))[:3] * 255
        elif conf[j]:
            color = np.asarray(cm.gist_rainbow(np.random.uniform()))[:3] * 255
        else:
            continue
        combined = cv2.line(combined, p1, p2+offset, color, 1)
        for n,p in enumerate([p1, p2+offset]):
            if (j < (N//2)) == (n == 0):
                combined = cv2.rectangle(combined, p-bit, p+bit, color=color, thickness=-1)
            else:
                combined = cv2.circle(combined, p, radius=3, color=color, thickness=-1)
    for i, l in enumerate(text):
        combined = cv2.putText(combined, l, (5,30+(40*i)), None, 1.0, (0, 100, 255), 2, cv2.LINE_AA)
    
    if path is None:
        return combined
    imwrite(path, combined)
