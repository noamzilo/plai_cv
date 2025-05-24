#!/usr/bin/env python3
#	⭾	TABS ONLY – copy / paste exactly

import cv2, math, argparse, random, numpy as np
from pathlib import Path
random.seed(0)

# ────────────────── quick helpers ──────────────────
def log(m): print(f"[PHASE] {m}")
def resize_if(img,s): return img if s==1 else cv2.resize(img,(0,0),fx=s,fy=s)
def l_angle(l): x1,y1,x2,y2=l; return (math.atan2(y2-y1,x2-x1)+math.pi)%math.pi
def l_len(l):   x1,y1,x2,y2=l; return math.hypot(x2-x1,y2-y1)
def l_abc(l):   x1,y1,x2,y2=l; return np.array([y1-y2,x2-x1,x1*y2-x2*y1],float)
def intersect(l1,l2):
	a,b=l_abc(l1),l_abc(l2); p=np.cross(a,b)
	return None if abs(p[2])<1e-6 else p[:2]/p[2]
def is_horizontal(angle,deg=10):
	return min(angle,math.pi-angle) < math.radians(deg)

# ───────── frame sampling ─────────
def collect_frames(video,out,step=10,max_n=30,scale=0.5):
	cache=sorted(out.glob("frame_*.*"))
	if cache:
		log(f"loading {len(cache)} cached frames")
		return [resize_if(cv2.imread(str(p)),scale) for p in cache[:max_n]]
	cap=cv2.VideoCapture(str(video)); N=int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
	f=[]
	for i in range(0,N,step):
		if len(f)==max_n: break
		cap.set(cv2.CAP_PROP_POS_FRAMES,i); r,frm=cap.read()
		if not r: break
		frm=resize_if(frm,scale); f.append(frm)
		out.mkdir(parents=True,exist_ok=True)
		cv2.imwrite(str(out/f"frame_{len(f):03d}.jpg"),frm)
	cap.release(); return f

# ───────── Hough with variable length ─────────
def hough_long(img,min_frac):
	h,w=img.shape[:2]; m=int(min_frac*min(h,w))
	edges=cv2.Canny(cv2.cvtColor(img,cv2.COLOR_BGR2GRAY),50,150)
	H=cv2.HoughLinesP(edges,1,np.pi/180,180,minLineLength=m,maxLineGap=10)
	if H is None: return [],edges
	# ← convert each ndarray to an immutable tuple
	return [tuple(map(int,l)) for l in H[:,0,:] if l_len(l)>=m], edges

# ───────── RANSAC helpers ─────────
def vp_from_pair(l1,l2): return intersect(l1,l2)
def angle_between(v1,v2):
	cos=np.dot(v1,v2)/(np.linalg.norm(v1)*np.linalg.norm(v2))
	return math.acos(max(-1,min(1,cos)))
def inliers_to_vp(vp,lines,th_deg=10):
	if vp is None: return []
	thr=math.radians(th_deg); v=np.array(vp,float); ok=[]
	for l in lines:
		x1,y1,x2,y2=l
		mid=np.array([(x1+x2)/2,(y1+y2)/2],float)
		dir_vec=np.array([x2-x1,y2-y1],float)
		to_vp=v-mid
		if np.linalg.norm(to_vp)<1: continue
		if angle_between(dir_vec,to_vp) < thr: ok.append(l)
	return ok
def two_vps_ransac(lines,iters=300,th_deg=10):
	best_vp1,best_in1=None,[]
	for _ in range(iters):
		l1,l2=random.sample(lines,2)
		vp=vp_from_pair(l1,l2)
		inl=inliers_to_vp(vp,lines,th_deg)
		if len(inl)>len(best_in1):
			best_vp1,best_in1=vp,inl
	if len(best_in1)<2: return None,None,[],[]
	rem=[l for l in lines if l not in best_in1]	# now safe, lines are tuples
	if len(rem)<2: return best_vp1,None,best_in1,rem
	best_vp2,best_in2=None,[]
	for _ in range(iters):
		l1,l2=random.sample(rem,2)
		vp=vp_from_pair(l1,l2)
		inl=inliers_to_vp(vp,rem,th_deg)
		if len(inl)>len(best_in2):
			best_vp2,best_in2=vp,inl
	return best_vp1,best_vp2,best_in1,best_in2

# ───────── main ─────────
def main():
	ap=argparse.ArgumentParser()
	ap.add_argument('--video',required=True); ap.add_argument('--out',default='debug')
	ap.add_argument('--scale',type=float,default=0.5); ap.add_argument('--frames',type=int,default=30)
	a=ap.parse_args(); out=Path(a.out)

	frames=collect_frames(Path(a.video),out,10,a.frames,a.scale)
	if len(frames)<5: log("too few frames"); return
	bg=np.median(np.stack(frames),0).astype(np.uint8)

	min_fracs=[0.15,0.10,0.07,0.05,0.04,0.03,0.02]
	for frac in min_fracs:
		lines,edges=hough_long(bg,frac)

		img_raw=bg.copy()
		for l in lines: cv2.line(img_raw,(l[0],l[1]),(l[2],l[3]),(0,255,0),1)
		cv2.imwrite(str(out/f'hough_lines_{int(frac*100)}p.jpg'),img_raw)

		if len(lines)<6: continue

		horiz_all=[l for l in lines if is_horizontal(l_angle(l))]
		others=[l for l in lines if not is_horizontal(l_angle(l))]
		if len(others)<4: continue

		vp1,vp2,in1,in2=two_vps_ransac(others)
		if vp1 is None or vp2 is None: continue

		if vp1[1] < vp2[1]:
			vert_all,depth_all,vp_vert,vp_depth=in1,in2,vp1,vp2
		else:
			vert_all,depth_all,vp_vert,vp_depth=in2,in1,vp2,vp1

		log(f"depth={len(depth_all)}, horiz={len(horiz_all)}, vert={len(vert_all)}")

		# ───────── dumps ─────────
		out.mkdir(parents=True,exist_ok=True)
		img=bg.copy()
		for l in horiz_all: cv2.line(img,(l[0],l[1]),(l[2],l[3]),(0,0,255),2)	# Red
		for l in vert_all:  cv2.line(img,(l[0],l[1]),(l[2],l[3]),(0,255,0),2)	# Green
		for l in depth_all: cv2.line(img,(l[0],l[1]),(l[2],l[3]),(255,0,0),2)	# Blue
		for vp in (vp_vert,vp_depth):
			cv2.circle(img,(int(vp[0]),int(vp[1])),7,(0,255,0),-1)
		cv2.imwrite(str(out/'clusters_rgb.jpg'),img)
		cv2.imwrite(str(out/'edges.jpg'),edges)

		inter=bg.copy()
		for ld in depth_all:
			for lh in horiz_all:
				p=intersect(ld,lh)
				if p is not None and np.all(np.isfinite(p)):
					cv2.drawMarker(inter,(int(p[0]),int(p[1])),(0,255,0),
					               cv2.MARKER_CROSS,20,3)
		cv2.imwrite(str(out/'depth×horiz_intersections.jpg'),inter)
		log(f"debug images → {out}")
		return

	log("could not get robust VP clustering at any length threshold")

if __name__=='__main__':
	main()
