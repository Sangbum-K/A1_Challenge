import re
"""scratch: a1.yaml <-> dvlc T_lidar_camera 변환과 투영 확인."""
import json, sys, math, numpy as np, cv2, yaml
A1 = "/home/ailab/A1-Challenge-2026-main/src/rte/a1_description/config/a1.yaml"
O = np.array([[0,-1,0],[0,0,-1],[1,0,0]], float)  # optical <- camera body
def rpy_m(r,p,y):
    cr,sr,cp,sp,cy,sy=map(float,(np.cos(r),np.sin(r),np.cos(p),np.sin(p),np.cos(y),np.sin(y)))
    return np.array([[cy*cp,cy*sp*sr-sy*cr,cy*sp*cr+sy*sr],[sy*cp,sy*sp*sr+cy*cr,sy*sp*cr-cy*sr],[-sp,cp*sr,cp*cr]])
def m_rpy(R):
    p=math.asin(max(-1,min(1,-R[2,0]))); return [math.atan2(R[2,1],R[2,2]),p,math.atan2(R[1,0],R[0,0])]
def q_from(R):
    from scipy.spatial.transform import Rotation; return Rotation.from_matrix(R).as_quat()  # x y z w
def R_from(q):
    from scipy.spatial.transform import Rotation; return Rotation.from_quat(q).as_matrix()
def body_from_lidar():
    l=yaml.safe_load(open(A1))["extrinsics"]["base_link_to_lidar"]; return rpy_m(*[math.radians(l[k]) for k in ("roll","pitch","yaw")])
def nominal_T_lidar_camera():
    c=yaml.safe_load(open(A1))["perception"]["camera_in_lidar_body"]; B=body_from_lidar()
    R=B.T@rpy_m(*np.radians(c["rpy_deg"]))@O.T; t=B.T@np.array(c["xyz_m"],float); return R,t
def to_a1(R,t):
    B=body_from_lidar(); return B@t, np.degrees(m_rpy(B@R@O))
def read_ply(path):
    data=open(path,"rb").read(); i=data.index(b"end_header\n")+len(b"end_header\n")
    n=int(re.search(rb"element vertex (\d+)",data[:i]).group(1))
    return np.frombuffer(data[i:i+n*16],np.float32).reshape(n,4).astype(float)
def overlay(pre, name, R_lc, t_lc, out):
    """LiDAR 점(R_lc,t_lc: camera->lidar)을 영상에 투영. 색 = intensity."""
    cal=json.load(open(f"{pre}/calib.json"))["camera"]; fx,fy,cx,cy=cal["intrinsics"]
    K=np.array([[fx,0,cx],[0,fy,cy],[0,0,1]]); D=np.array(cal["distortion_coeffs"])
    pts=read_ply(f"{pre}/{name}.ply"); img=cv2.imread(f"{pre}/{name}.png")
    if img.ndim==2 or img.shape[2]==1: img=cv2.cvtColor(img,cv2.COLOR_GRAY2BGR)
    Rcl=R_lc.T; tcl=-R_lc.T@t_lc; pc=pts[:,:3]@Rcl.T+tcl; keep=(pc[:,2]>0.5)&(pc[:,2]<40)
    uv,_=cv2.projectPoints(pts[keep,:3],cv2.Rodrigues(Rcl)[0],tcl,K,D); uv=uv.reshape(-1,2)
    z=pc[keep,2]; v=np.clip(1-(z-1)/14,0,1)
    col=cv2.applyColorMap((v*255).astype(np.uint8).reshape(-1,1),cv2.COLORMAP_JET).reshape(-1,3)
    h,w=img.shape[:2]; ok=(uv[:,0]>=0)&(uv[:,0]<w)&(uv[:,1]>=0)&(uv[:,1]<h)
    dim=(img*0.6).astype(np.uint8)
    for (u,vv),c in zip(uv[ok].astype(int),col[ok]): dim[vv,u]=c
    cv2.imwrite(out,dim); return int(ok.sum())
