"""Conditional P5 replay using the manually seeded LiDAR audit, never CAN.

Untracked/missing/stale observations are NaN, not zero brake. Nominal distance
preserves the existing P5 planar-range input contract. Longitudinal-distance
sensitivity is recorded separately; neither is bumper clearance/path distance.
"""
import argparse
import csv
import html
import json
import pathlib
import pickle
import sys
import numpy as np
import yaml

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT/'src'))
from a1_brake.brake import config_from_dict
from a1_brake.p5 import p5_accel
from audit_lidar_lead import frames, pose_at, reader


def clock_text(t):
    return f'{int(t)//60:02d}:{int(t)%60:02d}'


def write_csv(path, rows):
    with path.open('w', newline='', encoding='utf-8-sig') as f:
        w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)


def main():
    ap=argparse.ArgumentParser()
    ap.add_argument('--audit', type=pathlib.Path, default=ROOT/'outputs/lidar_lead_audit_20261008_0426_sparse')
    ap.add_argument('--out', type=pathlib.Path, default=ROOT/'outputs/lidar_p5_20261008_0300_1200')
    args=ap.parse_args();out=args.out;out.mkdir(parents=True,exist_ok=True)
    cfg=config_from_dict(yaml.safe_load((ROOT/'configs/brake_p5.yaml').read_text()))
    ins=dict(np.load(args.audit/'ins.npz'))
    raw=list(csv.DictReader((args.audit/'metrics.csv').open()))
    mt=np.array([float(r['video_s']) for r in raw])
    with (args.audit/'frames.pkl').open('rb') as f:first=pickle.load(f)
    epoch_ns=first['recv_ns']-round(first['video_s']*1e9)
    # Read only previously unprocessed early scan timestamps. No invented lead
    # before the user's 04:26 seed and no backward, noncausal association.
    cache=out/'early_scan_times.npy'
    if cache.exists():early=np.load(cache)
    else:
        from rclpy.serialization import deserialize_message
        from sensor_msgs.msg import PointCloud2
        r=reader(['/lidar_points']);r.seek(epoch_ns+180_000_000_000);early=[]
        while r.has_next():
            _,b,ns=r.read_next();v=(ns-epoch_ns)*1e-9
            if v>=mt[0]:break
            m=deserialize_message(b,PointCloud2)
            times=np.ndarray((m.width*m.height,),dtype='<f8',buffer=m.data,offset=18,strides=(m.point_step,))
            early.append([v,(float(times.min())+float(times.max()))/2])
        early=np.array(early);np.save(cache,early)
    timeline=list(early)
    for f in frames(args.audit):
        if f['video_s']>720.06:break
        timeline.append([f['video_s'],f['t']])
    timeline=np.array(timeline)
    rows=[]
    for video_s in np.arange(180.,720.001,.1):
        ti=int(np.argmin(abs(timeline[:,0]-video_s)))
        scan_video,stamp=timeline[ti]
        _,ev,yaw=pose_at(ins,stamp)
        ego_long=float(ev[0] @ np.array([np.cos(yaw[0]),np.sin(yaw[0])]))
        ego=max(0.,ego_long)
        j=int(np.argmin(abs(mt-video_s)));r=raw[j]
        same=abs(mt[j]-video_s)<=.06
        state='not_seeded' if video_s<mt[0] else ('no_scan_match' if not same else r['status'])
        valid=same and state=='measured' and r['fresh_estimate']=='True'
        if same and state=='measured' and not valid:state='warmup'
        row=dict(video_s=round(float(video_s),1),video_time=clock_text(video_s+1e-6),
            scan_video_s=float(scan_video),scan_time_unix=float(stamp),
            observation_state=state,p5_available=valid,
            ego_forward_kph=ego_long*3.6,ego_speed_kph=float(np.linalg.norm(ev[0])*3.6),
            lead_speed_kph=np.nan,lead_forward_kph=np.nan,relative_forward_mps=np.nan,
            range_m=np.nan,longitudinal_distance_m=np.nan,lateral_position_m=np.nan,
            lead_accel_2s_mps2=np.nan,p5_raw_mps2=np.nan,acc_cmd_assuming_gate1_mps2=np.nan,
            acc_cmd_longitudinal_gap_sensitivity_mps2=np.nan,min_gap_eff_m=np.nan,
            measurement_age_s=float(r['measurement_age_s']) if same else np.nan)
        if valid:
            # Use the source observation's aligned INS velocity for consistency.
            ego=max(0.,float(r['ego_long_mps']))
            lead=max(0.,float(r['lead_long_mps']))
            distance=float(r['range_m']);gap_x=float(r['x_surface_m'])
            p=p5_accel(ego_speed_mps=ego,gap_m=distance,lead_speed_mps=lead,config=cfg.p5)
            px=p5_accel(ego_speed_mps=ego,gap_m=gap_x,lead_speed_mps=lead,config=cfg.p5)
            def cmd(a):return float(np.clip(cfg.a_f1_mps2+min(0.,a-cfg.a_f1_mps2),cfg.acc_cmd_min_mps2,cfg.acc_cmd_max_mps2))
            row.update(ego_forward_kph=float(r['ego_long_mps'])*3.6,
                lead_speed_kph=float(r['lead_speed_mps'])*3.6,
                lead_forward_kph=float(r['lead_long_mps'])*3.6,
                relative_forward_mps=float(r['relative_long_mps']),range_m=distance,
                longitudinal_distance_m=gap_x,lateral_position_m=float(r['y_surface_m']),
                lead_accel_2s_mps2=float(r['lead_tangential_accel_2s_mps2']),
                p5_raw_mps2=p.accel_mps2,acc_cmd_assuming_gate1_mps2=cmd(p.accel_mps2),
                acc_cmd_longitudinal_gap_sensitivity_mps2=cmd(px.accel_mps2),
                min_gap_eff_m=p.min_gap_eff_m)
        rows.append(row)
    write_csv(out/'p5_10hz.csv',rows)
    moments=[min(rows,key=lambda r:abs(r['video_s']-t)) for t in [270,300,310,315,320,325,330,333,335,350,355]]
    write_csv(out/'p5_key_moments.csv',moments)
    def avg(a,k):
        vals=np.array([x[k] for x in a],float);vals=vals[np.isfinite(vals)]
        return float(vals.mean()) if len(vals) else np.nan
    def extrem(a,k,fn):
        vals=[x[k] for x in a if np.isfinite(x[k])]
        return float(fn(vals)) if vals else np.nan
    def aggregate(step):
        result=[]
        for start in range(180,720,step):
            a=[r for r in rows if start<=r['video_s']<min(start+step,720)]
            v=[r for r in a if r['p5_available']]
            result.append(dict(interval=f'{clock_text(start)}–{clock_text(min(start+step,720))}',
                total_samples=len(a),valid_samples=len(v),valid_percent=100*len(v)/len(a),
                ego_forward_mean_kph=avg(a,'ego_forward_kph'),
                ego_forward_min_kph=extrem(a,'ego_forward_kph',min),ego_forward_max_kph=extrem(a,'ego_forward_kph',max),
                lead_forward_mean_kph=avg(v,'lead_forward_kph'),lead_speed_mean_kph=avg(v,'lead_speed_kph'),
                range_min_m=extrem(v,'range_m',min),range_max_m=extrem(v,'range_m',max),
                relative_forward_mean_mps=avg(v,'relative_forward_mps'),
                p5_raw_mean_mps2=avg(v,'p5_raw_mps2'),p5_raw_min_mps2=extrem(v,'p5_raw_mps2',min),
                acc_cmd_mean_mps2=avg(v,'acc_cmd_assuming_gate1_mps2'),
                acc_cmd_min_mps2=extrem(v,'acc_cmd_assuming_gate1_mps2',min),
                brake_samples=sum(r['acc_cmd_assuming_gate1_mps2']<-.01 for r in v),
                saturated_samples=sum(r['acc_cmd_assuming_gate1_mps2']<=-2.999 for r in v)))
        return result
    minute=aggregate(60);ten=aggregate(10)
    write_csv(out/'p5_by_minute.csv',minute);write_csv(out/'p5_by_10seconds.csv',ten)
    valid=[r for r in rows if r['p5_available']]
    brake=[r for r in valid if r['acc_cmd_assuming_gate1_mps2']<-.01]
    saturated=[r for r in valid if r['acc_cmd_assuming_gate1_mps2']<=-2.999]
    summary=dict(total_samples=len(rows),valid_samples=len(valid),coverage_percent=100*len(valid)/len(rows),
        brake_samples=len(brake),saturated_samples=len(saturated),
        first_brake_s=brake[0]['video_s'],last_brake_s=brake[-1]['video_s'],
        first_saturation_s=saturated[0]['video_s'],last_saturation_s=saturated[-1]['video_s'],
        raw_min_mps2=min(r['p5_raw_mps2'] for r in valid),
        cmd_min_mps2=min(r['acc_cmd_assuming_gate1_mps2'] for r in valid),
        max_gap_definition_cmd_difference_mps2=max(abs(r['acc_cmd_assuming_gate1_mps2']-r['acc_cmd_longitudinal_gap_sensitivity_mps2']) for r in valid),
        assumptions=dict(plan_speed='measured INS forward speed, floored at zero',gate=1.,a_f1_mps2=cfg.a_f1_mps2,
            sampling='10 Hz offline grid, nearest source scan within 60 ms; not live control latency',
            distance='LiDAR-origin planar range to observed surface; not bumper/path gap',
            lead_speed='tracked ENU velocity projected onto ego forward axis, floored at zero',
            prediction='Existing P5 constant lead velocity; measured acceleration is reference only',
            missing='NaN for unseeded/lost/predicted/warmup; no zero substitution',
            scope='Offline conditional replay, not original logged command, no vehicle output'))
    (out/'summary.json').write_text(json.dumps(summary,indent=2,ensure_ascii=False))
    # Meaningful numerical checks of the generated artifact.
    assert len(rows)==5401
    for r in rows:
        if not r['p5_available']:
            assert np.isnan(r['p5_raw_mps2']) and np.isnan(r['acc_cmd_assuming_gate1_mps2'])
        else:
            assert -3.<=r['acc_cmd_assuming_gate1_mps2']<=0.
            assert abs((r['lead_forward_kph']-r['ego_forward_kph'])/3.6-r['relative_forward_mps'])<1e-7
    # Offline browser table with an expandable ten-second summary.
    def fmt(v,d=2):return '—' if not np.isfinite(v) else f'{v:.{d}f}'
    def table(data):
        body=[]
        for r in data:
            gap='—' if not r['valid_samples'] else f"{r['range_min_m']:.1f}~{r['range_max_m']:.1f}"
            body.append('<tr>'+''.join(f'<td>{v}</td>' for v in [r['interval'],fmt(r['ego_forward_mean_kph'],1),
                fmt(r['lead_forward_mean_kph'],1),gap,fmt(r['relative_forward_mean_mps']),
                fmt(r['p5_raw_min_mps2']),fmt(r['acc_cmd_mean_mps2']),fmt(r['acc_cmd_min_mps2']),
                f"{r['valid_percent']:.1f}% ({r['valid_samples']}/{r['total_samples']})"] )+'</tr>')
        return '<table><thead><tr>'+''.join('<th>'+s+'</th>' for s in ['영상 구간','자차 km/h','선행차 전방 km/h','거리 m','상대속도 m/s','P5 원값 최저','제한 후 평균','제한 후 최저','산출 가능 비율'])+'</tr></thead><tbody>'+''.join(body)+'</tbody></table>'
    page='''<!doctype html><html lang="ko"><meta charset="utf-8"><title>LiDAR P5 03:00–12:00</title><style>
body{font:16px system-ui;margin:32px auto;max-width:1250px;padding:0 20px;color:#172638;background:#f5f7fa}h1{font-size:28px}p{line-height:1.7}table{border-collapse:collapse;width:100%;background:white;margin:20px 0}td,th{padding:11px 8px;border-bottom:1px solid #dde3ea;text-align:right}th{background:#18384d;color:white;position:sticky;top:0}td:first-child,th:first-child{text-align:left}.note{background:#fff2d5;padding:18px;border-left:4px solid #b87800}img{width:100%}a{color:#1161a6}summary{cursor:pointer;padding:15px;background:#dce8f0}button{padding:8px 14px;cursor:pointer}</style>
<h1>원본 라이다 영상 03:00–12:00 · P5 오프라인 재계산</h1>
<p class="note">실제 차량 명령 기록이 아닙니다. 실제 자차속도를 계획속도로 사용하고, 선행차를 신뢰한다는 조건 g=1, F1=0으로 기존 P5를 계산했습니다. 미추적·예측만 유지·초기화 구간은 <b>산출 불가(—)</b>이며 브레이크 0이 아닙니다.</p>
<p>자차속도는 GNSS/INS, 선행차는 04:26에 수동 지정한 라이다 추적 결과입니다. 속도 비교는 자차 전방축 성분 기준이며 곡선에서는 선행차 자체 속력과 다릅니다. 기존 P5는 선행차 등속 예측을 사용하므로 가속도 추세는 CSV 참고 열에만 남겼습니다.</p>
<p>자차 평균은 전체 구간, 선행차·거리·P5 통계는 산출 가능한 표본만 사용합니다. 거리 기준은 라이다 중심→관측 표면 직선거리이며 범퍼 간격·경로상 거리가 아닙니다. 같은 주행 경로의 제어 대상인지도 별도 검증이 필요합니다. 단위는 속도 km/h, 상대속도 m/s, P5 m/s²입니다.</p>
<p><a href="p5_10hz.csv">10Hz 전체 CSV</a> · <a href="p5_by_10seconds.csv">10초 집계 CSV</a> · <a href="p5_key_moments.csv">주요 시점 CSV</a> · <a href="summary.json">조건·요약</a></p>
<img src="p5_timeline.png" alt="전체 P5 시계열">'''
    page+=table(minute)+'<details><summary>10초 단위 표 전체 보기</summary>'+table(ten)+'</details></html>'
    (out/'p5_table.html').write_text(page,encoding='utf-8')
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from matplotlib.font_manager import FontProperties
    plt.rcParams['font.family']=FontProperties(fname='/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc').get_name()
    plt.rcParams['axes.unicode_minus']=False
    x=np.array([r['video_s']/60 for r in rows]);av=np.array([r['p5_available'] for r in rows])
    def values(k):return np.array([r[k] for r in rows])
    fig,axs=plt.subplots(4,1,figsize=(15,11),sharex=True)
    axs[0].plot(x,values('ego_forward_kph'),label='자차 전방 속도: INS')
    axs[0].plot(x,values('lead_forward_kph'),label='선행차 전방 속도: 추적');axs[0].set_ylabel('km/h')
    axs[1].plot(x,values('range_m'),label='직선거리: 기존 P5 입력')
    axs[1].plot(x,values('longitudinal_distance_m'),alpha=.6,label='전방축 거리: 참고');axs[1].set_ylabel('m')
    axs[2].plot(x,values('relative_forward_mps'),label='전방 상대속도');axs[2].set_ylabel('m/s')
    axs[3].plot(x,values('p5_raw_mps2'),label='기존 P5 원값')
    axs[3].plot(x,values('acc_cmd_assuming_gate1_mps2'),label='g=1, −3 제한 후',linewidth=2)
    axs[3].set_ylabel('m/s²');axs[3].set_xlabel('원본 라이다 영상 재생 시각 [분]')
    for ax in axs:
        ax.set_xlim(3,12);ax.set_xticks(np.arange(3,12.1,.5));ax.grid(alpha=.25);ax.legend(loc='upper right')
        ax.fill_between(x,0,1,where=~av,transform=ax.get_xaxis_transform(),color='gray',alpha=.12,step='mid')
    fig.suptitle('3분~12분 P5 재계산 — 회색: 산출 불가, 흰 구간: 관측 기반 조건부 계산\n계획속도=실제 자차속도 · g=1 가정 · 원본 차량 명령 아님')
    fig.tight_layout();fig.savefig(out/'p5_timeline.png',dpi=140);plt.close(fig)
    print(json.dumps(summary,indent=2,ensure_ascii=False))
    print('MINUTE TABLE',json.dumps(minute,ensure_ascii=False,allow_nan=True))


if __name__=='__main__':main()
