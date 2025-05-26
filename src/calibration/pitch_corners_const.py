# pitch corners in game 1, hard coded. This will later be the output of a model
import numpy as np

net_left_bottom = np.array([840, 705])
net_right_bottom = np.array([2955, 751])
net_center_bottom = np.array([1906, 731])
far_left_corner = np.array([1372, 485])
far_right_corner = np.array([2451, 510])
close_white_line_center_left_far = np.array([1863, 1624])
close_white_line_center_left_close = np.array([1862, 1647])
close_white_line_center_right_far = np.array([1900, 1624])
close_white_line_center_right_close = np.array([1900, 1649])
net_left_top  = np.array([825, 507])
net_right_top = np.array([2980, 555])
net_center_top = np.array([1926, 543])

image_width = 3840
image_height = 2160