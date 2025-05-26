import numpy as np

# units are meters
# right-handed system: x = toward camera, y = right, z = up (height)
far_left_corner_3d = np.array([0.0, 0.0, 0.0])
far_right_corner_3d = np.array([0.0, 10.0, 0.0])
net_left_bottom_3d = np.array([10.0, 0.0, 0.0])
net_right_bottom_3d = np.array([10.0, 10.0, 0.0])
net_center_bottom_3d = np.array([10.0, 5.0, 0.0])
close_white_line_center_left_far_3d = np.array([16.95, 4.975, 0.0])
close_white_line_center_left_close_3d = np.array([17.0, 4.975, 0.0])
close_white_line_center_right_far_3d = np.array([16.95, 5.025, 0.0])
close_white_line_center_right_close_3d = np.array([17.0, 5.025, 0.0])
net_left_top_3d = np.array([10.0, 0.0, 0.92])
net_right_top_3d = np.array([10.0, 10.0, 0.92])
net_center_top_3d = np.array([10.0, 5.0, 0.88])