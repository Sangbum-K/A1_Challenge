"""Render an offline audit with measurement gaps visible, never filled as truth."""
import argparse
import csv
import json
import pathlib
import subprocess
import numpy as np
import cv2
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.font_manager import FontProperties
from audit_lidar_lead import frames, rotation


def slope(t, values, valid, window):
    out = np.full_like(values, np.nan)
    for i in range(len(t)):
        ids = np.flatnonzero((t >= t[i]-window) & (t <= t[i]) & valid)
        if len(ids) < max(5, int(window*4)) or t[i]-t[ids[0]] < window*.8 or not valid[i]:
            continue
        x = t[ids]-np.mean(t[ids])
        out[i] = (x[:, None]*values[ids]).sum(0)/(x*x).sum()
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--out', required=True)
    ap.add_argument('--video-end', type=float, default=375.)
    args = ap.parse_args()
    out = pathlib.Path(args.out)
    rows = list(csv.DictReader((out/'track.csv').open()))
    n = len(rows)
    def col(k):
        return np.array([float(r[k]) for r in rows])
    t = col('video_s')
    tracked = np.array([r['status']!='lost' for r in rows])
    measured = np.array([r['status']=='measured' for r in rows])
    fresh = tracked & (col('measurement_age_s') <= .3) & (t >= t[0]+1.)
    vel = np.column_stack([col('vx_world'), col('vy_world')])
    a1 = slope(t, vel, fresh & measured, 1.)
    a2 = slope(t, vel, fresh & measured, 2.)
    speeds = np.linalg.norm(vel, axis=1)
    tangent = np.divide(vel, speeds[:, None], out=np.full_like(vel, np.nan), where=speeds[:, None]>.5)
    accel1 = np.sum(a1*tangent, axis=1)
    accel2 = np.sum(a2*tangent, axis=1)
    # Average acceleration over the preceding 2 seconds, not instantaneous truth.
    for i, r in enumerate(rows):
        r['fresh_estimate'] = bool(fresh[i])
        r['lead_tangential_accel_1s_mps2'] = accel1[i]
        r['lead_tangential_accel_2s_mps2'] = accel2[i]
    with (out/'metrics.csv').open('w') as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
    last_hit = t[np.flatnonzero(measured)[-1]]
    lost = t[np.flatnonzero(~tracked)[0]] if np.any(~tracked) else None
    active = t <= last_hit
    warm_measured = measured & (t >= t[0]+1.)
    def stats(v, mask=warm_measured):
        a = np.asarray(v)[mask & np.isfinite(v)]
        return dict(zip(['min', 'p05', 'median', 'p95', 'max'], map(float,np.percentile(a,[0,5,50,95,100])))) if len(a) else {}
    gaps = []
    start = None
    for i in range(n):
        if active[i] and not measured[i] and start is None:
            start = i
        if start is not None and (measured[i] or not active[i]):
            gaps.append(dict(start_s=float(t[start]),end_s=float(t[i]),duration_s=float(t[i]-t[start])))
            start = None
    summary = dict(scans=n, examined_start_video_s=float(t[0]), examined_end_video_s=float(t[-1]),
        last_measurement_video_s=float(last_hit), first_lost_video_s=None if lost is None else float(lost),
        retained_until_last_measurement_s=float(last_hit-t[0]), measured_scans=int(measured.sum()),
        scans_through_last_measurement=int(active.sum()), measured_fraction_through_last=float(measured[active].mean()),
        max_recovered_gap_s=max([g['duration_s'] for g in gaps],default=0), recovered_gaps=gaps,
        max_interval_between_associated_measurements_s=float(np.max(np.diff(t[measured]))),
        range_m=stats(col('range_m')), lead_speed_mps=stats(col('lead_speed_mps')),
        relative_long_mps=stats(col('relative_long_mps')), points=stats(col('points')),
        innovation_m=stats(col('innovation_m')),
        lead_tangential_accel_2s_mps2=stats(accel2),
        accel_window_disagreement_abs_mps2=stats(abs(accel1-accel2)),
        extraction_compute_ms_all=stats(col('compute_ms'),np.ones(n,dtype=bool)),
        sparse_measurement_fraction_lt10=float(np.mean(col('points')[measured]<10)),
        warnings=['Manual seed, geometric candidates, no trained vehicle classifier.',
          'No speed/distance/acceleration ground truth; innovation is not accuracy.',
          'Observed surface reference is not the full vehicle center or bumper clearance.',
          'Clock and INS mounting fitted offline; planar motion compensation.',
          '2s acceleration is a trailing-window trend; low-speed direction is undefined.',
          'Prediction-only frames are not new measurements; >0.3s flagged stale.',
          'After loss, no identity reassignment. Other visible candidates are unclassified.',
          'Compute timing includes extraction/deskew/clustering but excludes bag I/O, localization, video.'])
    (out/'summary.json').write_text(json.dumps(summary,indent=2,allow_nan=False))
    # Static export: all data before loss, plus a small visible lost interval.
    plt.rcParams['font.family'] = FontProperties(fname='/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc').get_name()
    plt.rcParams['axes.unicode_minus'] = False
    fig, ax = plt.subplots(5,1,figsize=(15,13),sharex=True)
    ax[0].plot(t,np.where(tracked,col('range_m'),np.nan),label='필터 거리: 라이다→관측 표면 대표점')
    rawrange=np.hypot(col('raw_surface_x_m'),col('raw_surface_y_m'))
    ax[0].scatter(t,rawrange,s=4,alpha=.4,label='실제 관측 대표점 거리');ax[0].set_ylabel('거리 [m]')
    ax[1].plot(t,np.where(fresh,col('lead_speed_mps')*3.6,np.nan),label='선행차 평면 속력 추정')
    ax[1].plot(t,col('ego_long_mps')*3.6,label='자차 전방 속도 INS');ax[1].set_ylabel('속도 [km/h]')
    ax[2].plot(t,np.where(fresh,col('relative_long_mps'),np.nan),label='차량 전방축 상대속도')
    ax[2].plot(t,np.where(fresh,col('range_rate_mps'),np.nan),alpha=.6,label='직선거리 변화율');ax[2].axhline(0,c='gray',lw=.7);ax[2].set_ylabel('상대속도 [m/s]')
    ax[3].plot(t,accel1,alpha=.35,label='과거 1초 속도 기울기')
    ax[3].plot(t,accel2,label='과거 2초 속도 기울기');ax[3].axhline(0,c='gray',lw=.7);ax[3].set_ylabel('선행차 가감속 [m/s²]')
    ax[4].plot(t,np.where(measured,col('points'),np.nan),label='후보 점 개수: 0.16m voxel 처리 후')
    ax[4].set_ylabel('점 개수');ax[4].set_xlabel('원본 라이다 영상 재생 시각 [초]')
    for a in ax:
        a.grid(alpha=.25);a.legend(loc='upper right',fontsize=9);a.set_xlim(t[0],min(args.video_end,t[-1]))
        if lost is not None:a.axvspan(lost,min(args.video_end,t[-1]),color='red',alpha=.10)
        if lost is not None:a.axvspan(last_hit,lost,color='orange',alpha=.12)
        for g in gaps:a.axvspan(g['start_s'],g['end_s'],color='orange',alpha=.12)
    fig.suptitle('4분 26초 지정 선행차 — 수동 초기화 / 라이다 추적 + GNSS·INS 운동 보정\n주황: 관측 공백 · 빨강: ID 소실 · 가속도는 후행 2초 추세값, 정답 검증 아님')
    fig.tight_layout();fig.savefig(out/'tracking_metrics.png',dpi=150);plt.close(fig)
    # Video: front is RIGHT, matching the user's original LiDAR top view.
    w,h,fps=1440,900,10
    proc=subprocess.Popen(['ffmpeg','-y','-loglevel','error','-f','rawvideo','-pix_fmt','bgr24','-s',f'{w}x{h}','-r',str(fps),'-i','-','-an','-c:v','libx264','-preset','fast','-crf','21','-pix_fmt','yuv420p','-movflags','+faststart',str(out/'lead_tracking_0426.mp4')],stdin=subprocess.PIPE)
    def text(im,s,x,y,color=(230,230,230),scale=.6):
        cv2.putText(im,s,(x,y),cv2.FONT_HERSHEY_SIMPLEX,scale,color,1,cv2.LINE_AA)
    def project(p):
        return np.rint(np.column_stack([300+p[:,0]*8,430-p[:,1]*8])).astype(int)
    lut=cv2.applyColorMap(np.arange(256,dtype=np.uint8)[:,None],cv2.COLORMAP_TURBO)[:,0,:]
    charts=[('Range [m]',col('range_m'),(0,45)),('Lead speed [km/h]',np.where(fresh,col('lead_speed_mps')*3.6,np.nan),(0,25)),('Relative forward [m/s]',np.where(fresh,col('relative_long_mps'),np.nan),(-4,6)),('Lead acceleration, past 2s [m/s2]',accel2,(-4,4))]
    snapshots={266,280,288,310,330,350,359,362}
    contacts=[];vf=0;previous=None
    for i,fr in enumerate(frames(out)):
        if i>=len(rows) or fr['video_s']>args.video_end:break
        im=np.full((h,w,3),(20,18,15),np.uint8)
        text(im,'Seeded lead tracking | 2026-10-08 | LiDAR + GNSS/INS | NO CAMERA',25,30,scale=.75)
        text(im,f"Original video {int(t[i]//60):02d}:{t[i]%60:04.1f} | elapsed {t[i]-t[0]:.1f}s",25,63)
        for x in range(-30,71,10):
            a,b=project(np.array([[x,-25],[x,40]]));cv2.line(im,tuple(a),tuple(b),(48,43,38),1)
            text(im,str(x),int(a[0]),660,scale=.4)
        for y in range(-20,41,10):
            a,b=project(np.array([[-30,y],[70,y]]));cv2.line(im,tuple(a),tuple(b),(48,43,38),1)
        p=fr['p'];uv=project(p);good=(uv[:,0]>10)&(uv[:,0]<870)&(uv[:,1]>95)&(uv[:,1]<650)
        idx=np.flatnonzero(good);im[uv[idx,1],uv[idx,0]]=lut[np.clip((p[idx,2]+2)/5*255,0,255).astype(int)]
        # Other candidates remain dim; they are never labelled seed after loss.
        for c in fr['cs']:
            xy=project(c['p']);ok=(xy[:,0]>10)&(xy[:,0]<870)&(xy[:,1]>95)&(xy[:,1]<650)
            im[xy[ok,1],xy[ok,0]]=(110,105,100)
        origin=tuple(project(np.array([[0,0]]))[0]);cv2.circle(im,origin,5,(255,255,255),2)
        cv2.arrowedLine(im,origin,(origin[0]+40,origin[1]),(255,255,255),2)
        text(im,'EGO / FRONT ->',origin[0]-35,origin[1]+25,scale=.45)
        color=(70,230,100) if measured[i] else (0,175,255)
        state=rows[i]['status'].upper()
        if not tracked[i]:color=(70,70,240)
        text(im,'SEED ID: '+state,25,710,color,scale=.9)
        if tracked[i]:
            center=np.array([col('x_surface_m')[i],col('y_surface_m')[i]])
            chosen=int(rows[i]['candidate_index'])
            if chosen>=0:
                c=fr['cs'][chosen];xy=project(c['p'])
                for q in xy:
                    if 10<q[0]<870 and 95<q[1]<650:cv2.circle(im,tuple(q),2,color,-1)
                lo,hi=c['lo'][:2]-.4,c['hi'][:2]+.4
            else:lo,hi=center-[1.,1.3],center+[1.,1.3]
            box=project(np.array([lo,hi]));cv2.rectangle(im,tuple(box[0]),tuple(box[1]),color,2)
            cc=project(center[None])[0];text(im,'LEAD',int(cc[0])+12,int(cc[1])-12,color,.5)
            text(im,f"Range {float(rows[i]['range_m']):.2f} m | return age {float(rows[i]['measurement_age_s']):.2f} s",25,749,color)
            if fresh[i]:
                text(im,f"Lead {speeds[i]*3.6:.1f} km/h | relative {float(rows[i]['relative_long_mps']):+.2f} m/s",25,782)
                av='N/A' if not np.isfinite(accel2[i]) else f'{accel2[i]:+.2f}'
                text(im,f'Acceleration trend {av} m/s2 (trailing 2 s)',25,815)
            else:text(im,'Motion estimate stale / warming up - do not treat as measurement',25,782,color,.5)
        else:
            text(im,'Identity lost. Later candidates are NOT assigned to this vehicle.',25,754,color,.55)
            text(im,'No fabricated distance, velocity or acceleration after loss.',25,790,color,.55)
        text(im,'Surface range from LiDAR origin; not bumper clearance. Estimates, not ground truth.',25,872,(160,160,160),.47)
        for j,(name,y,lims) in enumerate(charts):
            x0,y0,cw,ch=925,120+j*180,485,125
            cv2.rectangle(im,(x0,y0),(x0+cw,y0+ch),(80,75,65),1)
            text(im,name,x0,y0-12,scale=.5)
            left=max(t[0],t[i]-25);right=left+25
            ids=np.flatnonzero((t>=left)&(t<=t[i])&np.isfinite(y))
            for k in range(1,len(ids)):
                a,b=ids[k-1],ids[k]
                if b-a>1:continue
                uv1=(int(x0+(t[a]-left)/25*cw),int(y0+ch-(np.clip(y[a],*lims)-lims[0])/(lims[1]-lims[0])*ch))
                uv2=(int(x0+(t[b]-left)/25*cw),int(y0+ch-(np.clip(y[b],*lims)-lims[0])/(lims[1]-lims[0])*ch))
                cv2.line(im,uv1,uv2,(80,220,220),2,cv2.LINE_AA)
            text(im,f'{lims[1]}',x0-35,y0+8,scale=.38);text(im,f'{lims[0]}',x0-35,y0+ch,scale=.38)
            text(im,f'{left:.0f}s - {right:.0f}s',x0,y0+ch+20,scale=.4)
        target_frame=int(round((t[i]-t[0])*fps))
        while previous is not None and vf<target_frame:
            proc.stdin.write(previous.tobytes());vf+=1
        proc.stdin.write(im.tobytes());vf+=1;previous=im
        sec=int(t[i])
        if sec in snapshots:
            cv2.imwrite(str(out/f'frame_{sec:03d}s.jpg'),im)
            contacts.append(cv2.resize(im,(720,450)));snapshots.remove(sec)
    proc.stdin.close()
    if proc.wait()!=0:raise RuntimeError('ffmpeg failed')
    while len(contacts)%2:contacts.append(np.zeros_like(contacts[0]))
    cv2.imwrite(str(out/'tracking_contact_sheet.jpg'),np.vstack([np.hstack(contacts[i:i+2]) for i in range(0,len(contacts),2)]))
    print(json.dumps({k:v for k,v in summary.items() if k not in ['recovered_gaps','warnings']},indent=2),flush=True)


if __name__=='__main__':main()
