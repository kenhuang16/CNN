"""
Define parameters
"""
import numpy as np
import glob


INF = 1.0e21
Y_METER_PER_PIXEL = 30.0 / 720  # meters per pixel in y dimension
X_METER_PER_PIXEL = 3.7 / 900  # meters per pixel in x dimension


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
    'frame': 500,
    'src': np.float32([[0, h_img], [570, 450], [708, 450], [w_img, h_img]]),
    'dst': np.float32([[0, h_img], [0, 0], [w_img, 0], [w_img, h_img]])
}

project_video = {
    'input': './videos/project_video.mp4',
    'output': './videos/project_video_processed.mp4',
    'frame': 500,
    'src': np.float32([[0, h_img], [570, 450], [708, 450], [w_img, h_img]]),
    'dst': np.float32([[0, h_img], [0, 0], [w_img, 0], [w_img, h_img]])
}

challenge_video = {
    'input': './videos/challenge_video.mp4',
    'output': './videos/challenge_video_processed.mp4',
    'frame': 354,
    'src': np.float32([[270, 720], [585, 500], [750, 500], [1050, 720]]),
    'dst': np.float32([[270, 720], [270, 0], [1050, 0], [1050, 720]])
}

harder_challenge_video = {
    'input': './videos/harder_challenge_video.mp4',
    'output': './videos/harder_challenge_video_processed.mp4',
    'frame': 354,
    'src': np.float32([[240, 700], [415, 580], [840, 580], [985, 700]]),
    'dst': np.float32([[240, 700], [240, 300], [985, 300], [985, 700]])
}

# Parameters for thresh-hold operations
thresh_params = [
    {'color_space': 'hls', 'channel': 2, 'direction': 'x', 'thresh': (20, 100)},
    {'color_space': 'hls', 'channel': 2, 'direction': None, 'thresh': (100, 255)},
    {'color_space': 'gray', 'channel': None, 'direction': None, 'thresh': (190, 255)}
]

# It is suggested to choose significantly different box sizes
car_search_params = {'scales': (0.50, 0.75),
                     'confidence_thresh': 0.3,
                     'overlap_thresh': 0.2,
                     'heat_thresh': 3.0,
                     'step_size': (0.125, 0.0625),
                     'regions': (((0.30, 0.50), (1.00, 0.95)),
                                 ((0.50, 0.50), (0.90, 0.80)))}