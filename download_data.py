import cv2 as cv
from sklearn.datasets import fetch_olivetti_faces
import numpy as np
import os

# Download Olivetti faces dataset
olivetti = fetch_olivetti_faces()
x = olivetti.images
y = olivetti.target

if not os.path.exists("images"):
    os.mkdir("images")
    for i in set(y):
        os.mkdir(f"images/{i}")

for img, label in zip(x,y):
    count = len(os.listdir(f"images/{label}"))
    img = np.round(np.reshape(img, (64,64,1)) * 255).astype(np.uint8)
    img_bgr = cv.cvtColor(img, cv.COLOR_GRAY2BGR)
    cv.imwrite(f"images/{label}/image_{count}.jpg", img_bgr)