"""
Define parameters
"""
import numpy as np
import glob


INF = 1.0e21
Y_METER_PER_PIXEL = 30.0 / 720  # meters per pixel in y dimension
X_METER_PER_PIXEL = 3.7 / 900  # meters per pixel in x dimension

# The length (in pixel) of the front of the car in the original image.
CAR_FRONT_LENGTH = 40

car_files = []
car_files.extend(glob.glob("data/vehicles/KITTI_extracted/*.png"))
car_files.extend(glob.glob("data/vehicles/GTI_Far/*.png"))
car_files.extend(glob.glob("data/vehicles/GTI_Left/*.png"))
car_files.extend(glob.glob("data/vehicles/GTI_Right/*.png"))
car_files.extend(glob.glob("data/vehicles/GTI_MiddleClose/*.png"))

noncar_files = []
noncar_files.extend(glob.glob("data/non-vehicles/Extras/*.png"))
noncar_files.extend(glob.glob("data/non-vehicles/GTI/*.png"))

w_img = 1280
h_img = 720

test_video = {
    'input': './videos/test_video.mp4',
    'output': './videos/test_video_processed.mp4',
    'src': np.float32([[40, 640], [560, 450], [720, 450], [1240, 640]]),
    'dst': np.float32([[0, h_img], [0, 0], [w_img, 0], [w_img, h_img]])
}

project_video = {
    'input': './videos/project_video.mp4',
    'output': './videos/project_video_processed.mp4',
    'src': np.float32([[0, 660], [555, 450], [715, 450], [1280, 665]]),
    'dst': np.float32([[0, h_img], [0, 0], [w_img, 0], [w_img, h_img]])
}

challenge_video = {
    'input': './videos/challenge_video.mp4',
    'output': './videos/challenge_video_processed.mp4',
    'src': np.float32([[0, 660], [555, 450], [715, 450], [1280, 665]]),
    'dst': np.float32([[270, 720], [270, 0], [1050, 0], [1050, 720]])
}

harder_challenge_video = {
    'input': './videos/harder_challenge_video.mp4',
    'output': './videos/harder_challenge_video_processed.mp4',
    'src': np.float32([[240, 700], [415, 580], [840, 580], [985, 700]]),
    'dst': np.float32([[240, 700], [240, 300], [985, 300], [985, 700]])
}

# Parameters for thresh-hold operations
# 'l' channel for white lines and 's' channel for yellow lines
thresh_params = [
    {'color_space': 'hls', 'channel': 1, 'direction': None, 'thresh': (180, 255)},
    {'color_space': 'hls', 'channel': 2, 'direction': None, 'thresh': (120, 255)}
]

# It is suggested to choose significantly different box sizes
car_search_params = {'scales': (0.50, 0.75),
                     'confidence_thresh': 0.6,
                     'overlap_thresh': 0.2,
                     'heat_thresh': 3.0,
                     'step_size': (0.125, 0.0625),
                     'regions': (((0.40, 0.50), (1.00, 0.95)),
                                 ((0.40, 0.50), (0.90, 0.85)))}
