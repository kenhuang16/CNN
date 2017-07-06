"""
test class Threshold
"""
import matplotlib.pyplot as plt
import cv2

from threshold import Threshold
from utilities import two_plots


test_image = "./test_images/test_image01.png"

thresh_params = [
    {'color_space': 'hls', 'channel': 2, 'direction': 'x', 'thresh': (20, 100)},
    {'color_space': 'hls', 'channel': 2, 'direction': None, 'thresh': (100, 255)},
    {'color_space': 'gray', 'channel': None, 'direction': None, 'thresh': (190, 255)}
]

img = cv2.imread(test_image)
img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

binary = None
for param in thresh_params:
    th = Threshold(img, param['color_space'], param['channel'])

    th.transform(param['direction'], thresh=param['thresh'])
    if binary is None:
        binary = th.binary
    else:
        binary |= th.binary

    # Visualize the result in each step
    title1 = param['color_space'].upper() + '-' + str(param['channel'])
    if param['direction'] is None:
        title2 = 'color thresh ' + str(param['thresh'])
    else:
        title2 = param['direction'] + ' gradient thresh ' + str(param['thresh'])

    two_plots(th.img, th.binary, titles=(title1, title2, ''))

plt.imshow(binary, cmap='gray')
plt.title("Combination of different threshold", fontsize=18)
plt.show()