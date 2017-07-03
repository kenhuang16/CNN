"""
This is a combination of the advanced lane line finding and
the vehicle detection projects.

Run main.py to process the input video
"""
import matplotlib.pyplot as plt

from traffic import TrafficVideo
from parameters import project_video, test_video, thresh_params


# my_video = project_video
video = test_video

# Process the video
ppt_trans_params = (video['frame'], video['src'], video['dst'])
f1 = TrafficVideo(video['input'], camera_cali_file='camera_cali.pkl',
                  perspective_transform_params=ppt_trans_params,
                  thresh_params=thresh_params,
                  car_classifier="car_classifier.pkl",
                  search_laneline=False,
                  search_car=True)

f1.show_perspective_transform()

f1.process(video['output'])

# for i in range(100, 2500, 100):
#     print("Frame {}".format(i))
#     image = f1.get_video_image(i)
#     plt.imsave('test_images/test_image{:02d}.png'.format(int(i/100)), image)
#     fig, ax = plt.subplots()
#     ax.imshow(image)
#     plt.tight_layout()
#     plt.show()
