"""
Test classes HogExtractor(), LbpExtractor()
"""
import os
import random

import cv2
import glob
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

from feature_extraction import HogExtractor, LbpExtractor

# ----
# Test single image HOG feature extraction
# ----

image = random.choice(glob.glob('data/vehicles/KITTI_extracted/*.png'))
img = cv2.imread(image)

extractor = HogExtractor(visual=True, colorspace='YCrCb', cell_per_block=(1, 1))
hog_features, hog_images = extractor.extract(img)

fig = plt.figure(figsize=(9, 6))
gs = gridspec.GridSpec(2, 3)
ax1 = fig.add_subplot(gs[0, 0])
ax2 = fig.add_subplot(gs[0, 1:3])

ax1.imshow(img)
ax2.plot(hog_features)
for i in range(3):
    try:
        fig.add_subplot(gs[1, i]).imshow(hog_images[i])
    except:
        pass

plt.suptitle("HOG features and images (colorspace: {})".
             format(extractor._colorspace))
plt.show()

# ----
# Test single image LBP feature extraction
# ----

image = "data/vehicles/KITTI_extracted/1.png"
img = cv2.imread(image)

extractor = LbpExtractor(colorspace='YCrCb')
lbp_features = extractor.extract(img)
extractor._visual = True
lbp_images = extractor.extract(img)

fig = plt.figure(figsize=(9, 6))
gs = gridspec.GridSpec(2, 3)
ax1 = fig.add_subplot(gs[0, 0])
ax2 = fig.add_subplot(gs[0, 1:3])

ax1.imshow(img)
ax2.plot(lbp_features)
for i in range(3):
    try:
        fig.add_subplot(gs[1, i]).imshow(lbp_images[i])
    except:
        pass

plt.suptitle("LBP features and images (colorspace: {})".
             format(extractor._colorspace))
plt.show()

# ----
# Test sliding window feature extraction
# ----

extractor = HogExtractor(colorspace='YCrCb', cell_per_block=(1, 1))
title = "HOG features"

# extractor = LbpExtractor(colorspace='YCrCb')
# title = "LBP features"

image = "test_images/test_image00.png"

if os.path.isfile(image):
    img = cv2.imread(image)
else:
    raise OSError("{} does not exist!".format(image))

b, g, r = cv2.split(img)  # get b,g,r
plt.imshow(cv2.merge([r, g, b]))
plt.title('Original image')
plt.show()

features, windows = \
    extractor.sliding_window_extract(img, step_size=(64, 64))

fig1, axs1 = plt.subplots(6, 6, figsize=(8, 8))
i = 120
for ax in axs1.flatten():
    if i > len(features) - 1:
        break
    ax.imshow(img[windows[i][0][1]:windows[i][1][1],
                  windows[i][0][0]:windows[i][1][0]])
    ax.set_axis_off()
    i += 1
plt.subplots_adjust(wspace=0.05)
plt.suptitle("sliding windows")
plt.show()

fig2, axs2 = plt.subplots(6, 6, figsize=(8, 8))
i = 100
for ax in axs2.flatten():
    if i > len(features) - 1:
        break
    ax.plot(features[i])
    ax.set_axis_off()
    i += 1
plt.subplots_adjust(wspace=0.05)

plt.suptitle(title)
plt.show()
