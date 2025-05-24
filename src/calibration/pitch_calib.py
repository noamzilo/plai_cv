#!/usr/bin/env python3
# tabs-only – copy/paste

import cv2, math, argparse, numpy as np
from pathlib import Path

# ---------- tiny utils ----------
def log(msg): print(f"[PHASE] {msg}")

def resize_if(img, s): return img if s == 1 else cv2.resize(
	img, (0,0), fx=s, fy=s, interpolation=cv2.INTER_AREA)

def angle(line):						# 0..π  (radians)
	x1,y1,x2,y2 = line
	return (math.atan2(y2-y1, x2-x1) + math.pi) % math.pi

def cross(a,b): return np.cross(a,b)

def intersect(l1,l2):
	p1,p2 = (*l1[:2],1),(*l1[2:],1)
	p3,p4 = (*l2[:2],1),(*l2[2:],1)
	x = cross(cross(p1,p2), cross(p3,p4))
	return None if abs(x[2])<1e-6 else (x[0]/x[2], x[1]/x[2])

# ---------- frame IO ----------
def collect_frames(video, out_dir, every=10, max_f=30, scale=0.5):
	cached = sorted(out_dir.glob("frame_*.*"))
	if cached:
		log(f"loading {len(cached)} cached frames")
		return [resize_if(cv2.imread(str(p)), scale) for p in cached[:max_f]]

	log("sampling video")
	cap = cv2.VideoCapture(video)
	N   = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
	frames=[]
	for i in range(0,N,every):
		if len(frames)==max_f: break
		cap.set(cv2.CAP_PROP_POS_FRAMES,i)
		r,f = cap.read()
		if not r: break
		f=resize_if(f,scale)
		frames.append(f)
		cv2.imwrite(str(out_dir/f"frame_{len(frames):03d}.jpg"),f)
		if len(frames)%5==0: log(f"collected {len(frames)} frames")
	cap.release()
	return frames

def median_bg(frames):
	log("median background")
	return np.median(np.stack(frames),axis=0).astype(np.uint8)

def motion_mask(frames,bg,thr=25):
	log("motion mask")
	acc=np.zeros(bg.shape[:2],np.uint8)
	for f in frames:
		g=cv2.cvtColor(cv2.absdiff(f,bg),cv2.COLOR_BGR2GRAY)
		_,m=cv2.threshold(g,thr,255,cv2.THRESH_BINARY)
		acc=cv2.bitwise_or(acc,m)
	k=cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(5,5))
	return cv2.dilate(acc,k,2)

# ---------- Hough + clustering ----------
def detect_lines(img,mask):
	log("Canny + Hough")
	e=cv2.Canny(cv2.cvtColor(img,cv2.COLOR_BGR2GRAY),50,150,apertureSize=3)
	e=cv2.bitwise_and(e,e,mask=cv2.bitwise_not(mask))
	L=cv2.HoughLinesP(e,1,np.pi/180,120,minLineLength=60,maxLineGap=12)
	return e,[] if L is None else L[:,0,:]

def kmeans_orient(lines,k=3):
	ang=np.float32([[angle(l)] for l in lines])
	_,lab,_=cv2.kmeans(ang,k,None,
		(cv2.TERM_CRITERIA_EPS+cv2.TERM_CRITERIA_MAX_ITER,30,0.1),
		10,cv2.KMEANS_PP_CENTERS)
	cl=[[] for _ in range(k)]
	for l_idx,l in zip(lab.ravel(),lines): cl[l_idx].append(l)
	return cl

def trim_cluster(cl,deg=8,min_inliers=3):
	if len(cl)<min_inliers: return []
	a=np.array([angle(l) for l in cl])
	med=np.median(a)
	inl=[l for l,ang in zip(cl,a)
		if abs(((ang-med+math.pi/2)%math.pi)-math.pi/2) < math.radians(deg)]
	return inl if len(inl)>=min_inliers else []

def vp_from_cluster(cl):
	if len(cl)<2: return None
	pts=[p for i in range(len(cl))
		 for j in range(i+1,len(cl))
		 if (p:=intersect(cl[i],cl[j]))]
	return None if len(pts)<3 else np.median(np.array(pts),axis=0)

def focal(v1,v2,cx,cy):
	nx1,ny1 = v1[0]-cx, v1[1]-cy
	nx2,ny2 = v2[0]-cx, v2[1]-cy
	f2 = -(nx1*nx2 + ny1*ny2)
	return None if f2<=0 else math.sqrt(f2)

# ---------- drawing ----------
def dump(bg,e,cls,keep_idx,vps,corners,out):
	out.mkdir(parents=True,exist_ok=True)
	cv2.imwrite(str(out/'01_bg.jpg'),bg)
	cv2.imwrite(str(out/'02_edges.jpg'),e)
	col=[(0,0,255),(255,0,0),(0,255,255)]
	im=bg.copy()
	for i,c in enumerate(cls):
		for x1,y1,x2,y2 in c:
			cv2.line(im,(x1,y1),(x2,y2),col[i],2)
	for vp in keep_idx:
		cv2.circle(im,(int(vps[vp][0]),int(vps[vp][1])),6,(0,255,0),-1)
	cv2.imwrite(str(out/'03_lines_vps.jpg'),im)
	fin=bg.copy()
	for x,y in corners:
		cv2.drawMarker(fin,(x,y),(0,255,0),cv2.MARKER_CROSS,28,4)
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
	bg=median_bg(frames)
	mmask=motion_mask(frames,bg)
	edges,lines=detect_lines(bg,mmask)
	if len(lines)<6: log("too few lines"); return

	cls=kmeans_orient(lines,3)
	cls=[trim_cluster(c) for c in cls]			# RANSAC trim

	# identify vertical cluster = median angle closest to 90°
	med_angles=[np.median([angle(l) for l in c]) if c else 999 for c in cls]
	vert_idx=int(np.argmin([abs(a-math.pi/2) for a in med_angles]))
	keep=[i for i in range(3) if i!=vert_idx]

	vps=[vp_from_cluster(c) for c in cls]
	if any(vps[i] is None for i in keep):
		log("could not compute VPs – check clusters"); return

	h,w=bg.shape[:2]; f=focal(vps[keep[0]],vps[keep[1]],w/2,h/2)
	print(f"[RESULT] focal ≈ {f:.1f},  vertical_cluster = {vert_idx}")

	# naïve two-corner demo: intersect topmost horizontal with extremes
	all_keep=cls[keep[0]]+cls[keep[1]]
	top=min(all_keep,key=lambda l:(l[1]+l[3])*0.5)
	left=min(all_keep,key=lambda l:min(l[0],l[2]))
	right=max(all_keep,key=lambda l:max(l[0],l[2]))
	c1,c2=intersect(top,left),intersect(top,right)
	corners=[tuple(map(int,c)) for c in (c1,c2) if c]

	dump(bg,edges,cls,keep,vps,corners,out)
	log(f"done → {out}")

if __name__=='__main__':
	main()
