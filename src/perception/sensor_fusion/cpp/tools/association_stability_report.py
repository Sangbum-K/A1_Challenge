"""Compare raw and filtered scan-to-scan changes within one run (not accuracy)."""
import argparse
import json
from collections import defaultdict
from pathlib import Path
import numpy as np

p = argparse.ArgumentParser()
p.add_argument('frames', type=Path)
args = p.parse_args()
frames = json.loads(args.frames.read_text())
tracks = defaultdict(list)
for f in frames:
    for tr in f['tracks']:
        if tr['confirmed'] and 'raw_center' in tr:
            tracks[tr['id']].append((f['stamp'], tr))
report = []
for tid, rows in tracks.items():
    pairs = [(a[1], b[1]) for a, b in zip(rows, rows[1:])
             if 0 < b[0]-a[0] < .15]
    if len(pairs) < 30:
        continue
    item = dict(id=tid, pairs=len(pairs),
                candidate_frames=sum(tid in f.get('candidate_tracks', [f['track']]) for f in frames))
    for key in ('raw_center', 'center', 'raw_size', 'size'):
        delta = np.array([np.array(b[key])-a[key] for a, b in pairs])
        item[key+'_step_rms_m'] = np.sqrt(np.mean(delta**2, axis=0)).tolist()
    report.append(item)
result = dict(
    interpretation='Consecutive-frame variation, not ground-truth accuracy. Motion and filter lag contribute.',
    tracks=sorted(report, key=lambda r: (-r['candidate_frames'], -r['pairs'])))
out = args.frames.parent/'stability.json'
out.write_text(json.dumps(result, indent=2)+'\n')
print(json.dumps(result, indent=2))
