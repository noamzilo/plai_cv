#!/usr/bin/env python3
#	⭾ TABS ONLY – copy / paste

import cv2, math, argparse, numpy as np
from pathlib import Path

# ---------- logging ----------
def log(msg): print(f"[PHASE] {msg}")

# ---------- basic helpers ----------
def resize_if(img, s): return img if s == 1 else cv2.resize(
	img, (0,0), fx=s, fy=s, interpolation=cv2.INTER_AREA)

def line_angle(l):
	x1,y1,x2,y2 = l
	return (math.atan2(y2 - y1, x2 - x1) + math.pi) % math.pi

def line_len(l):
	x1,y1,x2,y2 = l
	return math.hypot(x2 - x1, y2 - y1)

def line_abc(l):
	x1,y1,x2,y2 = l
	return np.array([y1 - y2, x2 - x1, x1*y2 - x2*y1], float)

# ---------- frame sampling ----------
def collect_frames(video, out_dir, step=10, max_n=30, scale=0.5):
	cached = sorted(out_dir.glob("frame_*.*"))
	if cached:
		log(f"loading {len(cached)} cached frames")
		return [resize_if(cv2.imread(str(p)), scale) for p in cached[:max_n]]

	log("sampling video")
	cap = cv2.VideoCapture(video); N = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
	frames=[]
	for i in range(0,N,step):
		if len(frames)==max_n: break
		cap.set(cv2.CAP_PROP_POS_FRAMES,i)
		r,f = cap.read()
		if not r: break
		f=resize_if(f,scale)
		frames.append(f)
		cv2.imwrite(str(out_dir/f"frame_{len(frames):03d}.jpg"),f)
		if len(frames)%5==0: log(f"collected {len(frames)} frames")
	cap.release(); return frames

def median_bg(frames):
	log("median background"); return np.median(np.stack(frames),0).astype(np.uint8)

def motion_mask(frames,bg,thr=25):
	log("motion mask")
	acc=np.zeros(bg.shape[:2],np.uint8)
	for f in frames:
		g=cv2.cvtColor(cv2.absdiff(f,bg),cv2.COLOR_BGR2GRAY)
		_,m=cv2.threshold(g,thr,255,cv2.THRESH_BINARY)
		acc=cv2.bitwise_or(acc,m)
	k=cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(5,5))
	return cv2.dilate(acc,k,2)

# ---------- Hough + filtering ----------
def detect_lines(img, keep_mask):
	log("Canny + Hough (long lines only)")
	g=cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
	edges=cv2.Canny(g,50,150,apertureSize=3)
	edges=cv2.bitwise_and(edges,edges,mask=keep_mask)		# keep motion pixels

	h,w=img.shape[:2]; min_len=int(0.25*min(h,w))			# 25 % of min dim
	raw=cv2.HoughLinesP(edges,1,np.pi/180,180,				# ↑ threshold
						minLineLength=min_len,maxLineGap=10)
	if raw is None: return edges,[]
	lines=[l for l in raw[:,0,:] if line_len(l)>=min_len]	# extra safeguard
	return edges,lines

# ---------- clustering & VP ----------
def kmeans_orient(lines,k=3):
	ang=np.float32([[line_angle(l)] for l in lines])
	_,lab,_=cv2.kmeans(ang,k,None,
		(cv2.TERM_CRITERIA_EPS+cv2.TERM_CRITERIA_MAX_ITER,30,0.1),
		10,cv2.KMEANS_PP_CENTERS)
	cl=[[] for _ in range(k)]
	for l_idx,l in zip(lab.ravel(),lines): cl[l_idx].append(l)
	return cl

def ransac_trim(cluster,deg=8):
	if len(cluster)<4: return cluster
	a=np.array([line_angle(l) for l in cluster])
	med=np.median(a)
	inl=[l for l,ang in zip(cluster,a)
		if abs(((ang-med+math.pi/2)%math.pi)-math.pi/2)<math.radians(deg)]
	return inl if len(inl)>=3 else inl

def vp_least_squares(cluster):
	if len(cluster)<2: return None
	A=np.stack([line_abc(l) for l in cluster])
	_,_,vh=np.linalg.svd(A); v=vh[-1]
	return None if abs(v[2])<1e-6 else np.array([v[0]/v[2], v[1]/v[2]])

def focal(v1,v2,cx,cy):
	nx1,ny1=v1[0]-cx,v1[1]-cy
	nx2,ny2=v2[0]-cx,v2[1]-cy
	f2=-(nx1*nx2+ny1*ny2)
	return None if f2<=0 else math.sqrt(f2)

# ---------- drawing ----------
def dump_debug(bg,edges,cls,vps,keep,corners,out):
	log("dumping debug")
	out.mkdir(parents=True,exist_ok=True)
	cv2.imwrite(str(out/'01_bg.jpg'),bg)
	cv2.imwrite(str(out/'02_edges.jpg'),edges)
	col=[(0,0,255),(255,0,0),(0,255,255)]
	im=bg.copy()
	for i,c in enumerate(cls):
		for x1,y1,x2,y2 in c: cv2.line(im,(x1,y1),(x2,y2),col[i],2)
	for i in keep:
		cv2.circle(im,(int(vps[i][0]),int(vps[i][1])),6,(0,255,0),-1)
	cv2.imwrite(str(out/'03_lines_vps.jpg'),im)
	fin=bg.copy()
	for x,y in corners:
		cv2.drawMarker(fin,(x,y),(0,255,0),cv2.MARKER_CROSS,30,4)
	cv2.imwrite(str(out/'04_final.jpg'),fin)

# ---------- main ----------
def main():
	ap=argparse.ArgumentParser()
	ap.add_argument('--video',required=True)
	ap.add_argument('--out',default='debug')
	ap.add_argument('--scale',type=float,default=0.5)
	ap.add_argument('--frames',type=int,default=30)
	args=ap.parse_args(); out=Path(args.out)

	frames=collect_frames(args.video,out,10,args.frames,args.scale)
	if len(frames)<5: log("too few frames"); return
	bg = median_bg(frames)
	msk= motion_mask(frames,bg)
	edges,lines = detect_lines(bg,msk)
	if len(lines)<6: log("too few long lines"); return

	cls=[ransac_trim(c) for c in kmeans_orient(lines,3)]
	med=[np.median([line_angle(l) for l in c]) if c else 999 for c in cls]
	vert_idx=int(np.argmin([abs(a-math.pi/2) for a in med]))
	keep=[i for i in range(3) if i!=vert_idx]

	vps=[vp_least_squares(c) for c in cls]
	if any(vps[i] is None for i in keep):
		log("VP failure"); return
	h,w=bg.shape[:2]; f=focal(vps[keep[0]],vps[keep[1]],w/2,h/2)
	print(f"[RESULT] focal ≈ {f:.1f}, vertical_cluster = {vert_idx}")

	# simple two corners demo
	all_keep=cls[keep[0]]+cls[keep[1]]
	top=min(all_keep,key=lambda l:(l[1]+l[3])*0.5)
	left=min(all_keep,key=lambda l:min(l[0],l[2]))
	right=max(all_keep,key=lambda l:max(l[0],l[2]))
	c1=vp_least_squares([top,left])
	c2=vp_least_squares([top,right])
	corners=[tuple(map(int,p)) for p in (c1,c2) if p is not None]

	dump_debug(bg,edges,cls,vps,keep,corners,out)
	log(f"done → {out}")

if __name__=='__main__':
	main()
