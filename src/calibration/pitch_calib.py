#!/usr/bin/env python3
#	⭾	TABS ONLY – copy / paste exactly

import cv2, math, argparse, random, numpy as np
from pathlib import Path

random.seed(0)

# ────────────────── helpers ──────────────────
def log(m): print(f"[PHASE] {m}")
def resize_if(img,s): return img if s==1 else cv2.resize(img,(0,0),fx=s,fy=s)
def l_angle(l): x1,y1,x2,y2=l; return (math.atan2(y2-y1,x2-x1)+math.pi)%math.pi
def l_len(l):   x1,y1,x2,y2=l; return math.hypot(x2-x1,y2-y1)
def l_abc(l):   x1,y1,x2,y2=l; return np.array([y1-y2,x2-x1,x1*y2-x2*y1],float)
def intersect(l1,l2):
	a,b=l_abc(l1),l_abc(l2); p=np.cross(a,b)
	return None if abs(p[2])<1e-6 else p[:2]/p[2]

def is_horizontal(angle,deg=10):		# only for horizontals
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

# ───────── Hough lines (two-pass) ─────────
def hough_long(img,min_frac):
	h,w=img.shape[:2]
	edges=cv2.Canny(cv2.cvtColor(img,cv2.COLOR_BGR2GRAY),50,150)

	# pass 1 – generic long lines
	m1=int(min_frac*min(h,w))
	H1=cv2.HoughLinesP(edges,1,np.pi/180,140,minLineLength=m1,maxLineGap=10)
	lines=[]
	if H1 is not None:
		lines+=[tuple(map(int,l)) for l in H1[:,0,:] if l_len(l)>=m1]

	# pass 2 – near-vertical shorter lines
	m2=int(0.05*min(h,w))
	H2=cv2.HoughLinesP(edges,1,np.pi/180,90,minLineLength=m2,maxLineGap=10)
	if H2 is not None:
		for l in H2[:,0,:]:
			l=tuple(map(int,l))
			if  abs(l_angle(l)-math.pi/2)<math.radians(20):		# near vertical
				lines.append(l)

	return lines, edges

# ───────── VP helpers & RANSAC ─────────
def robust_vp(lines):
	if len(lines)<2: return None
	A=np.stack([l_abc(l) for l in lines]); _,_,vh=np.linalg.svd(A)
	v=vh[-1]; return None if abs(v[2])<1e-6 else v[:2]/v[2]

def pt_line_dist2(pt,l):
	x0,y0=pt; x1,y1,x2,y2=l
	dx,dy=x2-x1,y2-y1; num=abs(dy*x0-dx*y0+x2*y1-y2*x1); den=math.hypot(dx,dy)
	return 1e9 if den==0 else (num/den)**2

def inliers(vp,lines,th=60):
	if vp is None: return []
	th2=th*th; return [l for l in lines if pt_line_dist2(vp,l)<=th2]

def two_vps_ransac(lines,iters=400,th=60):
	best_v1,best_in1=None,[]
	for _ in range(iters):
		l1,l2=random.sample(lines,2)
		vp=intersect(l1,l2); inl=inliers(vp,lines,th)
		if len(inl)>len(best_in1):
			best_v1,best_in1=vp,inl
	if len(best_in1)<2: return None,None,[],[]
	rem=[l for l in lines if l not in best_in1]
	if len(rem)<2: return best_v1,None,best_in1,rem
	best_v2,best_in2=None,[]
	for _ in range(iters):
		l1,l2=random.sample(rem,2)
		vp=intersect(l1,l2); inl=inliers(vp,rem,th)
		if len(inl)>len(best_in2):
			best_v2,best_in2=vp,inl
	return best_v1,best_v2,best_in1,best_in2

# ───────── main ─────────
def main():
	ap=argparse.ArgumentParser()
	ap.add_argument('--video',required=True); ap.add_argument('--out',default='debug')
	ap.add_argument('--scale',type=float,default=0.5); ap.add_argument('--frames',type=int,default=30)
	args=ap.parse_args(); out=Path(args.out)

	frames=collect_frames(Path(args.video),out,10,args.frames,args.scale)
	if len(frames)<5:
		log("too few frames"); return
	bg=np.median(np.stack(frames),0).astype(np.uint8)

	min_fracs=[0.20,0.15,0.12,0.10,0.08,0.06,0.04]
	for frac in min_fracs:
		lines,edges=hough_long(bg,frac)

		# dump ALL Hough lines (white)
		raw=bg.copy()
		for l in lines: cv2.line(raw,(l[0],l[1]),(l[2],l[3]),(255,255,255),1)
		cv2.imwrite(str(out/f'hough_raw_{int(frac*100)}p.jpg'),raw)

		if len(lines)<6: continue
		horiz=[l for l in lines if is_horizontal(l_angle(l))]
		others=[l for l in lines if l not in horiz]
		if len(others)<4: continue

		v1,v2,in1,in2=two_vps_ransac(others)
		if v1 is None or v2 is None: continue
		if v1[1]<v2[1]:
			vert,depth,vp_vert,vp_depth=in1,in2,v1,v2
		else:
			vert,depth,vp_vert,vp_depth=in2,in1,v2,v1
		if len(vert)<2: continue

		vp_horiz=robust_vp(horiz)

		# ---------- LOG ----------
		log(f"VP-horiz (red)  : {None if vp_horiz is None else tuple(map(int,vp_horiz))}")
		log(f"VP-vert  (green) : {tuple(map(int,vp_vert))}")
		log(f"VP-depth (blue)  : {tuple(map(int,vp_depth))}")
		log(f"clusters — horiz:{len(horiz)}  vert:{len(vert)}  depth:{len(depth)}")

		# ---------- CANVAS ----------
		h,w=bg.shape[:2]
		px=[0,w-1,int(vp_depth[0])]; py=[0,h-1,int(vp_depth[1])]
		min_x,min_y,max_x,max_y=min(px),min(py),max(px),max(py)
		off_x=-min_x if min_x<0 else 0; off_y=-min_y if min_y<0 else 0
		canvas=np.zeros((max_y-min_y+1,max_x-min_x+1,3),np.uint8)
		canvas[off_y:off_y+h,off_x:off_x+w]=bg

		def draw(l,c): cv2.line(canvas,(l[0]+off_x,l[1]+off_y),
		                                (l[2]+off_x,l[3]+off_y),c,2)
		for l in horiz: draw(l,(0,0,255))
		for l in vert:  draw(l,(0,255,0))
		for l in depth: draw(l,(255,0,0))

		# ONLY depth & vert VP marks
		cv2.drawMarker(canvas,(int(vp_depth[0])+off_x,int(vp_depth[1])+off_y),
		              (255,0,0),markerType=cv2.MARKER_CROSS,markerSize=30,thickness=4)
		cv2.drawMarker(canvas,(int(vp_vert[0])+off_x,int(vp_vert[1])+off_y),
		              (0,255,0),markerType=cv2.MARKER_CROSS,markerSize=30,thickness=4)

		# legend
		font=cv2.FONT_HERSHEY_SIMPLEX; y=20
		cv2.line(canvas,(10,y),(60,y),(0,0,255),4); cv2.putText(canvas,"Horiz",(70,y+5),font,0.5,(0,0,255),1)
		y+=20; cv2.line(canvas,(10,y),(60,y),(0,255,0),4); cv2.putText(canvas,"Vert",(70,y+5),font,0.5,(0,255,0),1)
		y+=20; cv2.line(canvas,(10,y),(60,y),(255,0,0),4); cv2.putText(canvas,"Depth",(70,y+5),font,0.5,(255,0,0),1)

		cv2.imwrite(str(out/'clusters_rgb.jpg'),canvas)
		cv2.imwrite(str(out/'edges.jpg'),edges)
		return

	log("no robust VP clustering found")

if __name__=='__main__':
	main()
