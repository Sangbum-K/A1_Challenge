"""jiwon_e2e MultiAgnet pkl → 호환 npz. multi-process 로 NFS I/O 병렬화.

mask 필드 (큰 200×100 array) 무시 → pkl 일부만 로드. (그래도 pickle 은 전체 deserialize 함)
근본 fix: 8 worker process 병렬 → I/O wait 흡수.
"""
import pickle, glob, os, numpy as np, json, time, multiprocessing as mp

PKL_GLOB = "/home/ailab-12/AILabDataset/03_Shared_Repository/jiwon/jiwon_e2e/e2e_racing/data/processed/frames/sangam_track__MultiAgnet__*.pkl"
OUT_NPZ = "/home/ailab-12/AILabDataset/03_Shared_Repository/jiwon/scc_controller_imitator/data/multiagnet_compact.npz"


def load_one(fp):
    with open(fp, 'rb') as f:
        d = pickle.load(f)
    meta = d["meta"]
    ego = np.asarray(d["ego_pose"], dtype=np.float32)
    wp = np.asarray(d["wp_rel_gt"], dtype=np.float32)
    wp_valid = np.asarray(d["wp_valid"], dtype=bool)
    obj_list = []
    for o in d["objects"]:
        obj_list.append({
            "bbox": [float(x) for x in o["bbox"]],
            "velocity": [float(x) for x in o["velocity"]],
            "class": int(o["class"]),
            "track_id": int(o.get("track_id", 0)),
        })
    return {
        "ego": ego, "wp": wp, "wp_valid": wp_valid,
        "frame_id": meta["frame_id"], "track": meta["track"],
        "policy": meta["policy"], "run_id": meta["run_id"],
        "tick": int(meta["tick"]),
        "objects": obj_list,
    }


def main():
    files = sorted(glob.glob(PKL_GLOB))
    n = len(files)
    print(f"converting {n} pkl files with {min(16, mp.cpu_count())} workers ...")
    t0 = time.time()
    results = [None] * n
    nw = min(16, mp.cpu_count())
    with mp.Pool(nw) as pool:
        done = 0
        for i, r in enumerate(pool.imap(load_one, files, chunksize=8)):
            results[i] = r
            done += 1
            if done % 500 == 0:
                rate = done / (time.time() - t0)
                eta = (n - done) / rate
                print(f"  [{done}/{n}] {rate:.0f}/s  ETA {eta/60:.1f}min")
    print(f"\nall loaded in {(time.time()-t0)/60:.1f}min. assembling npz ...")

    out = {}
    meta_list = []
    for i, r in enumerate(results):
        out[f"s{i}__ego_pose"] = r["ego"]
        out[f"s{i}__wp_rel_gt"] = r["wp"]
        out[f"s{i}__wp_valid"] = r["wp_valid"]
        meta_list.append({
            "i": i, "frame_id": r["frame_id"], "track": r["track"],
            "policy": r["policy"], "run_id": r["run_id"], "tick": r["tick"],
            "objects": r["objects"],
        })
    os.makedirs(os.path.dirname(OUT_NPZ), exist_ok=True)
    np.savez_compressed(OUT_NPZ, **out)
    json_path = OUT_NPZ.replace(".npz", "_meta.json")
    with open(json_path, "w") as f:
        json.dump(meta_list, f)
    print(f"wrote {OUT_NPZ} ({os.path.getsize(OUT_NPZ)/1e6:.1f} MB)")
    print(f"wrote {json_path} ({os.path.getsize(json_path)/1e6:.1f} MB)")


if __name__ == "__main__":
    main()
