#!/usr/bin/env python3
#	⭾	TABS ONLY – copy / paste exactly

from collections import OrderedDict, defaultdict
import \
	numpy as np
import \
	pandas as pd
import \
	cv2
from pathlib import \
	Path

# ─── Project paths ───────────────────────────────────────────────────
from utils.paths import \
	(
	raw_data_path,
	tracked_detections_csv_path, calculated_data_path,
)
from acquisition.VideoReader import \
	VideoReader

from calibration.pitch_keypoints_3d import \
	(
	far_left_corner_3d,
	far_right_corner_3d,
	net_left_bottom_3d,
	net_right_bottom_3d,
)
from calibration.pitch_corners_const import \
	(
	far_left_corner,
	far_right_corner,
	net_left_bottom,
	net_right_bottom,
)

# ─── Constants ───────────────────────────────────────────────────────
image_scale_factor			= 0.25
pitch_inner_size_px			= (600, 300)		# before orientation swap
buffer_px					= 100

player_colors_bgr = [
	(255,   0,   0),
	(  0, 255,   0),
	(  0,   0, 255),
	(255, 255,   0),
]
player_color_array = np.array(player_colors_bgr, dtype=np.uint8)

# ════════════════════════════════════════════════════════════════════
#                               PITCH
# ════════════════════════════════════════════════════════════════════
class Pitch2d:
	def __init__(
			self,
			length_m: float							= 20.0,
			width_m: float							= 10.0,
			canvas_inner_px: tuple[int,int]			= pitch_inner_size_px,
			pitch_2d_corners_4x2: np.ndarray | None	= None,
			img_2d_corners_4x2: np.ndarray | None	= None,
			orientation: str						= "vertical",
	):
		self.len_m, self.wid_m = float(length_m), float(width_m)
		self.orientation = orientation.lower()
		if self.orientation not in ("vertical","horizontal"):
			raise ValueError("orientation must be 'vertical' or 'horizontal'.")

		inner_w,inner_h = canvas_inner_px if self.orientation=="horizontal" else canvas_inner_px[::-1]
		self.inner_w_px, self.inner_h_px = inner_w, inner_h
		self.canvas_w_px, self.canvas_h_px = inner_w+2*buffer_px, inner_h+2*buffer_px

		if pitch_2d_corners_4x2 is None or img_2d_corners_4x2 is None:
			self.homography_3x3 = self.inv_homography_3x3 = None
		else:
			self.homography_3x3, _ = cv2.findHomography(
				pitch_2d_corners_4x2,
				img_2d_corners_4x2,
				method = cv2.RANSAC,
				ransacReprojThreshold = 3.0,
			)
			self.inv_homography_3x3 = np.linalg.inv(self.homography_3x3)

	def blank_canvas(self) -> np.ndarray:
		img = np.zeros((self.canvas_h_px, self.canvas_w_px, 3), np.uint8)
		cv2.rectangle(
			img,
			(buffer_px, buffer_px),
			(buffer_px+self.inner_w_px-1, buffer_px+self.inner_h_px-1),
			(255,255,255),2,
		)
		return img

	def pitch_to_canvas(self,p_xy:tuple[float,float])->tuple[int,int]:
		x_m,y_m = p_xy
		if self.orientation=="horizontal":
			cx_i = int((x_m/self.len_m)*self.inner_w_px)
			cy_i = int((1-y_m/self.wid_m)*self.inner_h_px)
		else:
			cx_i = int((y_m/self.wid_m)*self.inner_w_px)
			cy_i = int((x_m/self.len_m)*self.inner_h_px)
		return cx_i+buffer_px, cy_i+buffer_px

	def image_to_pitch_batch(self,img_xy:np.ndarray)->np.ndarray:
		return self._apply_homography(img_xy,self.inv_homography_3x3)

	def _apply_homography(self,pts:np.ndarray,H:np.ndarray)->np.ndarray:
		if H is None: raise ValueError("Homography not initialised.")
		if pts.ndim==1: pts = pts[np.newaxis,:]
		ones = np.ones((pts.shape[0],1),np.float32)
		dst = (H @ np.hstack([pts,ones]).T).T
		return dst[:,:2]/dst[:,2:3]

	def draw_polyline(self,canvas,seq,col,t=2):
		for p0,p1 in zip(seq[:-1],seq[1:]):
			cv2.line(canvas,self.pitch_to_canvas(p0),self.pitch_to_canvas(p1),col,t)
		return canvas

	def draw_points(self,canvas,pt_specs,font=0.5):
		for (px,py),col,lbl in pt_specs:
			if np.isnan(px): continue
			cx,cy=self.pitch_to_canvas((px,py))
			cv2.circle(canvas,(cx,cy),6,col,-1)
			if lbl:
				cv2.putText(canvas,lbl,(cx+8,cy-8),cv2.FONT_HERSHEY_SIMPLEX,font,col,1,cv2.LINE_AA)
		return canvas

# ════════════════════════════════════════════════════════════════════
#                       STATIC PITCH BACKGROUND
# ════════════════════════════════════════════════════════════════════
def build_static_canvas(pitch:Pitch2d)->np.ndarray:
	img_pts = OrderedDict([
		("deadzone0",			np.array([0,1055])),
		("deadzone1",			np.array([0,2159])),
		("deadzone2",			np.array([3839,2159])),
		("deadzone3",			np.array([3839,1174])),
		("far_left_corner",		far_left_corner),
		("far_right_corner",	far_right_corner),
		("net_right_bottom",	net_right_bottom),
		("net_left_bottom",		net_left_bottom),
	])
	pitch_pts = pitch.image_to_pitch_batch(np.array(list(img_pts.values()),np.float32))

	canvas = pitch.blank_canvas()

	poly_order = [
		"far_left_corner","far_right_corner","net_right_bottom",
		"deadzone3","deadzone2","deadzone1","deadzone0",
		"net_left_bottom","far_left_corner",
	]
	pitch_pts_poly = [pitch_pts[list(img_pts.keys()).index(n)] for n in poly_order]
	pitch.draw_polyline(canvas,pitch_pts_poly,(255,255,255),2)

	# net line
	pitch.draw_polyline(canvas,[(pitch.len_m/2,0),(pitch.len_m/2,pitch.wid_m)],(255,255,255),2)
	return canvas

# ════════════════════════════════════════════════════════════════════
#         STEP-1  ︳  CONVERT / CLAMP  →  PITCH-COORDS CSV
# ════════════════════════════════════════════════════════════════════
def convert_to_pitch(src:Path,dst:Path,pitch:Pitch2d)->None:
	df = pd.read_csv(src)
	centre_img = np.stack([((df.x1+df.x2)/2).to_numpy(np.float32), df.y2.to_numpy(np.float32)],1)
	centre_pitch = pitch.image_to_pitch_batch(centre_img)
	df["p_x"],df["p_y"] = centre_pitch[:,0],centre_pitch[:,1]

	inside = lambda x,y: (0<=x) & (x<=pitch.len_m) & (0<=y) & (y<=pitch.wid_m)
	last_valid = defaultdict(lambda:(np.nan,np.nan))

	rows=[]
	for idx,row in df.sort_values(["player_id","frame_ind"]).iterrows():
		pid = int(row.player_id)
		px,py = row.p_x,row.p_y

		if inside(px,py):
			last_valid[pid]=(px,py)
			rows.append(row)
		else:
			lx,ly=last_valid[pid]
			if np.isnan(lx):
				continue						# drop until we see first inside
			row.p_x,row.p_y = lx,ly			# snap back
			row.is_valid   = False			# mark as unreliable
			rows.append(row)

	pd.DataFrame(rows).to_csv(dst,index=False)
	print("→",dst)

# ════════════════════════════════════════════════════════════════════
#                    STEP-2  LIVE VISUALISER
# ════════════════════════════════════════════════════════════════════
class Visualiser:
	def __init__(self,video:Path,df:pd.DataFrame,pitch:Pitch2d,base:np.ndarray):
		self.df=df; self.pitch=pitch; self.base=base; self.reader=VideoReader(video)

	def show(self):
		for fi,frame in self.reader.video_frames_generator():
			sub=self.df[self.df.frame_ind==fi]
			if sub.empty: continue
			pids=sub.player_id.to_numpy(np.int32); val=sub.is_valid.astype(bool).to_numpy()
			pts=sub[["p_x","p_y"]].to_numpy(np.float32)

			# draw players on pitch
			pt_specs=[]
			for pid,(px,py),v in zip(pids,pts,val):
				if not v: continue
				col=player_color_array[pid%len(player_color_array)].tolist()
				pt_specs.append(((float(px),float(py)),col,f"P{pid}"))

			pitch_img=self.pitch.draw_points(self.base.copy(),pt_specs,0.5)

			# camera frame + boxes
			small=cv2.resize(frame,(int(frame.shape[1]*image_scale_factor),int(frame.shape[0]*image_scale_factor)))
			for pid,(x1,y1,x2,y2),v in zip(pids,sub[["x1","y1","x2","y2"]].values,val):
				if not v: continue
				col=player_color_array[pid%len(player_color_array)].tolist()
				sx1,sy1,sx2,sy2=[int(c*image_scale_factor) for c in (x1,y1,x2,y2)]
				cv2.rectangle(small,(sx1,sy1),(sx2,sy2),col,2)
				cv2.putText(small,f"P{pid}",(sx1,sy1-8),cv2.FONT_HERSHEY_SIMPLEX,0.6,col,2,cv2.LINE_AA)

			h_target=small.shape[0]
			pitch_res=cv2.resize(pitch_img,(int(pitch_img.shape[1]*h_target/pitch_img.shape[0]),h_target))
			cv2.imshow("Tracking",np.hstack([small,pitch_res]))
			if cv2.waitKey(1)==ord("q"):break
		cv2.destroyAllWindows()

# ════════════════════════════════════════════════════════════════════
#                       STEP-3  HEATMAPS
# ════════════════════════════════════════════════════════════════════
def build_heatmaps(df: pd.DataFrame, pitch: Pitch2d, base: np.ndarray, output_dir: Path):
	h, w = pitch.canvas_h_px, pitch.canvas_w_px
	acc = {pid: np.zeros((h, w), np.float32) for pid in range(4)}

	for pid, g in df[df.is_valid.astype(bool)].groupby("player_id"):
		g = g.sort_values("frame_ind")
		if len(g) < 2:
			continue
		p = g[["p_x", "p_y"]].to_numpy(np.float32)
		for (x0, y0), (x1, y1) in zip(p[:-1], p[1:]):
			d = np.hypot(x1 - x0, y1 - y0)
			d = max(d, 1e-3)
			inten = 1.0 / d
			cv2.line(acc[pid], pitch.pitch_to_canvas((x0, y0)), pitch.pitch_to_canvas((x1, y1)), inten, 1)

	colored = []
	output_dir.mkdir(parents=True, exist_ok=True)

	for pid in range(4):
		hm = acc[pid]
		hm = cv2.GaussianBlur(hm, (11, 11), 0)

		# Apply non-linear enhancement to heat
		hm = np.power(hm, 0.5)

		hm_norm = cv2.normalize(hm, None, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX)
		hm_u8 = hm_norm.astype(np.uint8)
		hm_color = cv2.applyColorMap(hm_u8, cv2.COLORMAP_JET)

		overlay = cv2.addWeighted(base, 0.3, hm_color, 0.7, 0)

		cv2.putText(overlay, f"Player {pid}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)

		cv2.imwrite(str(output_dir / f"heatmap_player_{pid}.png"), overlay)
		colored.append(overlay)

	cv2.imshow("Player Heatmaps 0-3", np.hstack(colored))
	cv2.waitKey(0)
	cv2.destroyAllWindows()

# ════════════════════════════════════════════════════════════════════
#                                MAIN
# ════════════════════════════════════════════════════════════════════
def main():
	pitch_pts = np.stack([
		far_left_corner_3d[:2],
		far_right_corner_3d[:2],
		net_right_bottom_3d[:2],
		net_left_bottom_3d[:2],
	])
	img_pts = np.stack([far_left_corner,far_right_corner,net_right_bottom,net_left_bottom])

	pitch = Pitch2d(pitch_2d_corners_4x2=pitch_pts,img_2d_corners_4x2=img_pts)
	static_canvas = build_static_canvas(pitch)

	pitch_csv = tracked_detections_csv_path.with_stem(tracked_detections_csv_path.stem+"_pitch")

	run_convert=False; run_vis=False; run_heat=True

	if run_convert:
		convert_to_pitch(tracked_detections_csv_path,pitch_csv,pitch)

	df=None
	if run_vis or run_heat:
		df = pd.read_csv(pitch_csv)

	if run_vis:
		game_data = raw_data_path/"game1_3.mp4"
		Visualiser(game_data,df,pitch,static_canvas).show()

	if run_heat:
		game_data_dir = calculated_data_path/"game1_3.mp4"
		build_heatmaps(df,pitch,static_canvas, output_dir=game_data_dir)

if __name__=="__main__":
	main()
