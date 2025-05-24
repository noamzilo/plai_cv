#!/usr/bin/env python3
#	⭾	TABS ONLY – copy / paste

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
	return [l for l in H[:,0,:] if l_len(l)>=m],edges

# ───────── clustering & trimming ─────────
def kmeans(lines,k):
	ang=np.float32([[l_angle(l)] for l in lines])
	_,lab,_=cv2.kmeans(ang,k,None,(cv2.TERM_CRITERIA_EPS+cv2.TERM_CRITERIA_MAX_ITER,30,0.1),10,cv2.KMEANS_PP_CENTERS)
	C=[[] for _ in range(k)]
	for idx,l in zip(lab.ravel(),lines): C[idx].append(l)
	return C

def trim(cl,deg=8):
	if len(cl)<4: return cl
	a=np.array([l_angle(l) for l in cl]); med=np.median(a)
	return [l for l,ang in zip(cl,a)
			if abs(((ang-med+math.pi/2)%math.pi)-math.pi/2)<math.radians(deg)]

# ───────── VP (SVD + random fallback) ─────────
def vp_svd(cl):
	if len(cl)<2: return None
	A=np.stack([l_abc(l) for l in cl]); _,_,vh=np.linalg.svd(A)
	v=vh[-1]; return None if abs(v[2])<1e-6 else v[:2]/v[2]

def vp_random(cl,tries=200):
	if len(cl)<2: return None
	pts=[]
	for _ in range(tries):
		l1,l2=random.sample(cl,2)
		p=intersect(l1,l2)
		if p is not None and np.all(np.isfinite(p)): pts.append(p)
	return None if len(pts)<5 else np.median(np.array(pts),0)

def robust_vp(cl):
	vp=vp_svd(cl)
	return vp if vp is not None else vp_random(cl)

def focal(v1,v2,cx,cy):
	n1=v1-np.array([cx,cy]); n2=v2-np.array([cx,cy])
	f2=-(n1[0]*n2[0]+n1[1]*n2[1])
	return None if f2<=0 else math.sqrt(f2)

# ───────── main ─────────
def main():
	ap=argparse.ArgumentParser()
	ap.add_argument('--video',required=True); ap.add_argument('--out',default='debug')
	ap.add_argument('--scale',type=float,default=0.5); ap.add_argument('--frames',type=int,default=30)
	a=ap.parse_args(); out=Path(a.out)

	frames=collect_frames(Path(a.video),out,10,a.frames,a.scale)
	if len(frames)<5: log("too few frames"); return
	bg=np.median(np.stack(frames),0).astype(np.uint8)

	min_fracs=[0.15,0.10,0.07,0.05,0.04,0.03,0.02]		# progressively shorter
	for frac in min_fracs:
		lines,edges=hough_long(bg,frac)
		img_raw = bg.copy()
		for l in lines:
			cv2.line(img_raw, (l[0], l[1]), (l[2], l[3]), (0, 255, 0), 1)
		cv2.imwrite(str(out/f'hough_lines_{int(frac*100)}p.jpg'), img_raw)
		if len(lines)<6: continue
		cl=[trim(c) for c in kmeans(lines,3)]
		# identify clusters
		med=[np.median([l_angle(l) for l in c]) if c else 999 for c in cl]
		vert_idx=int(np.argmin([abs(a-math.pi/2) for a in med]))
		other=[i for i in range(3) if i!=vert_idx]
		if len(other)!=2: continue
		depth_idx=max(other,key=lambda i:med[i]); horiz_idx=[i for i in other if i!=depth_idx][0]
		depth,horiz,vert=cl[depth_idx],cl[horiz_idx],cl[vert_idx]

		if len(vert)>=2:					# finally have enough vertical lines
			break
		log(f"vertical lines {len(vert)} < 2 → relax length to {min_fracs[min_fracs.index(frac)+1]*100:.1f}%")
	else:
		log("never got enough vertical lines"); return

	log(f"depth={len(depth)}, horiz={len(horiz)}, vert={len(vert)}")
	vp_depth,vp_vert=robust_vp(depth),robust_vp(vert)
	if vp_depth is None or vp_vert is None:
		log("still no VP"); return

	h,w=bg.shape[:2]; f=focal(vp_depth,vp_vert,w/2,h/2)
	print(f"[RESULT] f≈{f:.1f}")

	# dumps
	out.mkdir(parents=True,exist_ok=True)
	# coloured lines + VPs
	img=bg.copy()
	for l in depth: cv2.line(img,(l[0],l[1]),(l[2],l[3]),(255,0,0),2)
	for l in horiz: cv2.line(img,(l[0],l[1]),(l[2],l[3]),(0,0,255),2)
	for l in vert:  cv2.line(img,(l[0],l[1]),(l[2],l[3]),(0,255,255),2)
	for vp in (vp_depth,vp_vert): cv2.circle(img,(int(vp[0]),int(vp[1])),7,(0,255,0),-1)
	cv2.imwrite(str(out/'lines_vps_coloured.jpg'),img)
	cv2.imwrite(str(out/'edges.jpg'),edges)

	# intersections depth × horiz
	inter=bg.copy()
	for ld in depth:
		for lh in horiz:
			p=intersect(ld,lh)
			if p is not None and np.all(np.isfinite(p)):
				cv2.drawMarker(inter,(int(p[0]),int(p[1])),(0,255,0),cv2.MARKER_CROSS,20,3)
	cv2.imwrite(str(out/'depth×horiz_intersections.jpg'),inter)

	log(f"debug images → {out}")

if __name__=='__main__':
	main()
