#!/usr/bin/env python3
#	⭾	TABS ONLY – copy / paste exactly

import numpy as np
import cv2
from scipy.optimize import least_squares
from utils.paths import calculated_data_path
import plotly.graph_objects as go

# ─── 2-D image points ────────────────────────────────────────────────
from calibration.pitch_corners_const import (
	net_left_bottom, net_right_bottom, net_center_bottom,
	far_left_corner, far_right_corner,
	image_width, image_height, net_left_top, net_right_top, net_center_top,
	close_white_line_center_right_far, close_white_line_center_right_close,
	close_white_line_center_left_close, close_white_line_center_left_far,
)

# ─── 3-D pitch points (metres) ───────────────────────────────────────
from calibration.pitch_keypoints_3d import (
	far_left_corner_3d, far_right_corner_3d,
	net_left_bottom_3d, net_right_bottom_3d, net_center_bottom_3d,
	net_left_top_3d, net_right_top_3d, net_center_top_3d,
	close_white_line_center_left_close_3d, close_white_line_center_left_far_3d,
	close_white_line_center_right_close_3d, close_white_line_center_right_far_3d
)

# ─── Build correspondences ───────────────────────────────────────────
image_points_2d = np.asarray([
	far_left_corner, far_right_corner,
	net_left_bottom, net_right_bottom, net_center_bottom,
	close_white_line_center_left_far, close_white_line_center_right_far,
	close_white_line_center_left_close, close_white_line_center_right_close,
	net_left_top, net_right_top, net_center_top
], dtype=np.float32)

object_points_3d = np.asarray([
	far_left_corner_3d, far_right_corner_3d,
	net_left_bottom_3d, net_right_bottom_3d, net_center_bottom_3d,
	close_white_line_center_left_far_3d, close_white_line_center_right_far_3d,
	close_white_line_center_left_close_3d, close_white_line_center_right_close_3d,
	net_left_top_3d, net_right_top_3d, net_center_top_3d
], dtype=np.float32)

# ─── Intrinsics guess (pinhole) ───────────────────────────────────────
f_guess	= float(image_width)					# ≈ diagonal
cx, cy	= image_width / 2.0, image_height / 2.0
K = np.array([[f_guess, 0.0,  cx],
			  [0.0,		f_guess, cy],
			  [0.0,		0.0,	1.0]], dtype=np.float32)
dist = np.zeros((4,1), np.float32)

# ─── Warm-start pose with EPNP ────────────────────────────────────────
_, rvec0, tvec0 = cv2.solvePnP(
	objectPoints	= object_points_3d.reshape(-1,1,3),
	imagePoints		= image_points_2d.reshape(-1,1,2),
	cameraMatrix	= K,
	distCoeffs		= dist,
	flags			= cv2.SOLVEPNP_EPNP
)
R0, _ = cv2.Rodrigues(rvec0)
C0 = (-R0.T @ tvec0).flatten()					# world camera centre

# state vector: [rx, ry, rz, Cx, Cz]  with Cy fixed = 20 m
FIXED_CY = 20.0
x0 = np.hstack([rvec0.flatten(), C0[[0,2]]])

def residuals(x):
	rvec = x[:3].reshape(3,1).astype(np.float32)
	Cx, Cz = x[3:]
	Cw = np.array([Cx, FIXED_CY, Cz], dtype=np.float32).reshape(3,1)
	R, _ = cv2.Rodrigues(rvec)
	tvec = -R @ Cw										# convert to camera coords
	proj, _ = cv2.projectPoints(object_points_3d, rvec, tvec, K, dist)
	return (proj.reshape(-1,2) - image_points_2d).ravel()

opt = least_squares(residuals, x0, method="lm", xtol=1e-10, ftol=1e-10, max_nfev=400)

# ─── Extract refined pose ────────────────────────────────────────────
rvec = opt.x[:3].reshape(3,1).astype(np.float32)
Cx, Cz = opt.x[3:]
Cw  = np.array([Cx, FIXED_CY, Cz], dtype=np.float32).reshape(3,1)
R, _ = cv2.Rodrigues(rvec)
tvec = -R @ Cw
P = K @ np.hstack((R, tvec))

# ─── Diagnostics ─────────────────────────────────────────────────────
proj_pts, _ = cv2.projectPoints(object_points_3d, rvec, tvec, K, dist)
proj_pts = proj_pts.reshape(-1,2)
reproj_err = np.linalg.norm(proj_pts - image_points_2d, axis=1)

print("\n[INFO] Camera centre C_w:", Cw.flatten())
print("[INFO] Rotation vector   :", rvec.flatten())
print("[INFO] Translation vec   :", tvec.flatten())
print("[INFO] Mean reproj error :", reproj_err.mean(), "px")

# ─── 2-D overlay (scaled) ────────────────────────────────────────────
frame_path = calculated_data_path / "game1_3.mp4" / "average_frame.bmp"
img = cv2.imread(str(frame_path)); assert img is not None
SCALE = 0.25
vis = cv2.resize(img, (0,0), fx=SCALE, fy=SCALE)
for (xo,yo),(xp,yp) in zip(image_points_2d, proj_pts):
	cv2.drawMarker(vis,(int(xo*SCALE),int(yo*SCALE)),(0,255,0),cv2.MARKER_CROSS,15,2)
	cv2.drawMarker(vis,(int(xp*SCALE),int(yp*SCALE)),(255,0,0),cv2.MARKER_CROSS,15,2)
	cv2.line(vis,(int(xo*SCALE),int(yo*SCALE)),(int(xp*SCALE),int(yp*SCALE)),(0,0,255),1,cv2.LINE_AA)

out_png = calculated_data_path / "reprojection_overlay.scaled.png"
cv2.imwrite(str(out_png), vis)
cv2.imshow("2D Points vs Projections (scaled)", vis)
cv2.waitKey(0)
print(f"[INFO] Saved overlay → {out_png}")

# ─── 3-D Plotly visualisation ────────────────────────────────────────
cam_dir = (R.T @ np.array([[0],[0],[1]],dtype=float)).flatten()
arrow_end = Cw.flatten() + cam_dir
pts3d = object_points_3d.astype(float)

fig = go.Figure()
fig.add_trace(go.Scatter3d(x=pts3d[:,0],y=pts3d[:,1],z=pts3d[:,2],
	mode='markers+text',text=[f"pt{i}" for i in range(len(pts3d))],
	marker=dict(size=5,color='red'),name='3D Keypoints'))
fig.add_trace(go.Scatter3d(x=[Cw[0,0]],y=[Cw[1,0]],z=[Cw[2,0]],
	mode='markers+text',text=["Camera"],marker=dict(size=9,color='blue'),name='Camera'))
fig.add_trace(go.Scatter3d(x=[Cw[0,0],arrow_end[0]],y=[Cw[1,0],arrow_end[1]],
	z=[Cw[2,0],arrow_end[2]],mode='lines',
	line=dict(color='green',width=4),name='View Direction'))
fig.update_layout(
	title='3D Pitch Keypoints & Camera (C_y = 20 m)',
	scene=dict(xaxis_title='X',yaxis_title='Y',zaxis_title='Z',aspectmode='data'),
	margin=dict(l=0,r=0,b=0,t=30)
)
fig.show()
