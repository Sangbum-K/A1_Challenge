"""jiwon_e2e MultiAgnet pkl (numpy 2.x) → py3.8 호환 npz.

ROS env (py3.8) 의 합성 파이프라인이 jiwon_e2e pkl 들을 직접 못 읽으므로
ml_con env (py3.11) 로 한 번에 변환.

Output: 단일 npz 파일에 6877 frame 의 핵심 필드만 (image 경로는 별도 필요 없음 — pkl 의 meta 에서 재구성 가능).
"""
import pickle, glob, os, numpy as np, json, time

PKL_GLOB = "/home/ailab-12/AILabDataset/03_Shared_Repository/jiwon/jiwon_e2e/e2e_racing/data/processed/frames/sangam_track__MultiAgnet__*.pkl"
OUT_NPZ = "/home/ailab-12/AILabDataset/03_Shared_Repository/jiwon/scc_controller_imitator/data/multiagnet_compact.npz"


def main():
    files = sorted(glob.glob(PKL_GLOB))
    print(f"converting {len(files)} pkl files ...")
    t0 = time.time()
    out = {}
    obj_summary = []   # JSON 으로 별도 저장 (variable-length lists)
    for i, fp in enumerate(files):
        with open(fp, 'rb') as f:
            d = pickle.load(f)
        meta = d["meta"]
        ego = np.asarray(d["ego_pose"], dtype=np.float32)
        wp = np.asarray(d["wp_rel_gt"], dtype=np.float32)
        wp_valid = np.asarray(d["wp_valid"], dtype=bool)
        out[f"s{i}__ego_pose"] = ego
        out[f"s{i}__wp_rel_gt"] = wp
        out[f"s{i}__wp_valid"] = wp_valid
        # objects: list of dict {bbox(4,), velocity(2,), class, track_id, mask}. mask 는 너무 크므로 skip.
        obj_list = []
        for o in d["objects"]:
            obj_list.append({
                "bbox": [float(x) for x in o["bbox"]],
                "velocity": [float(x) for x in o["velocity"]],
                "class": int(o["class"]),
                "track_id": int(o.get("track_id", 0)),
            })
        obj_summary.append({
            "i": i,
            "frame_id": meta["frame_id"],
            "track": meta["track"],
            "policy": meta["policy"],
            "run_id": meta["run_id"],
            "tick": int(meta["tick"]),
            "objects": obj_list,
        })
        if (i + 1) % 1000 == 0:
            print(f"  [{i+1}/{len(files)}] {(i+1)/(time.time()-t0):.0f}/s")
    os.makedirs(os.path.dirname(OUT_NPZ), exist_ok=True)
    np.savez_compressed(OUT_NPZ, **out)
    json_path = OUT_NPZ.replace(".npz", "_meta.json")
    with open(json_path, "w") as f:
        json.dump(obj_summary, f)
    print(f"\nwrote {OUT_NPZ} ({os.path.getsize(OUT_NPZ)/1e6:.1f} MB)")
    print(f"wrote {json_path} ({os.path.getsize(json_path)/1e6:.1f} MB)")


if __name__ == "__main__":
    main()
