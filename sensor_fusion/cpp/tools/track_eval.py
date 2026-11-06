#!/usr/bin/env python3
"""Summarise a run's association.csv and tracks CSV. Offline only; reads files, starts nothing.

labels.csv is optional and hand-made: start,end,lead with lead=1 when a lead car is really there.
"""
import argparse
import collections
import csv
import json
import statistics
from pathlib import Path


def read(path):
    with open(path, newline='') as stream:
        return list(csv.DictReader(stream))


def shares(values):
    counts = collections.Counter(values)
    total = sum(counts.values())
    return {key: count / total for key, count in counts.items()} if total else {}


def percentile(values, q):
    if not values:
        return None
    ordered = sorted(values)
    position = (len(ordered) - 1) * q
    low = int(position)
    high = min(low + 1, len(ordered) - 1)
    return ordered[low] + (ordered[high] - ordered[low]) * (position - low)


def primary_summary(rows):
    switches = 0
    segments = []
    start = previous_time = None
    previous_id = -1
    for row in rows:
        stamp = float(row['stamp'])
        primary = int(row['primary_id'])
        if primary != previous_id:
            if previous_id >= 0:
                segments.append(previous_time - start)
                if primary >= 0:
                    switches += 1
            start = stamp if primary >= 0 else None
        previous_id = primary
        previous_time = stamp
    if previous_id >= 0:
        segments.append(previous_time - start)
    return {'id_switches': switches, 'segments': len(segments),
            'longest_s': max(segments, default=0.), 'median_s': statistics.median(segments) if segments else 0.}


def label_summary(rows, labels):
    spans = [(float(r['start']), float(r['end']), r['lead'] == '1') for r in read(labels)]
    hits = {True: [0, 0], False: [0, 0]}
    for row in rows:
        stamp = float(row['stamp'])
        for start, end, lead in spans:
            if start <= stamp < end:
                hits[lead][0] += 1
                hits[lead][1] += int(row['primary_id']) >= 0
                break
    rate = lambda pair: pair[1] / pair[0] if pair[0] else None
    return {'kept_rate': rate(hits[True]), 'wrong_rate': rate(hits[False]),
            'lead_scans': hits[True][0], 'empty_scans': hits[False][0]}


def range_shells(tracks):
    shells = collections.defaultdict(lambda: {'tracks': 0, 'measured': 0, 'points': 0})
    for row in tracks:
        low = int(float(row['range_m']) // 20) * 20
        shell = shells[f'{low}-{low + 20}']
        shell['tracks'] += 1
        shell['measured'] += int(row['measured'])
        shell['points'] += int(row['points'])
    return dict(shells)


def summarize(association_csv, labels=None):
    association_csv = Path(association_csv)
    rows = read(association_csv)
    tracks = read(association_csv.with_suffix('.tracks.csv'))
    times = [float(row['processing_ms']) for row in rows]
    out = {
        'scans': len(rows),
        'primary': primary_summary(rows),
        'status_share': shares(row['primary_status'] for row in rows),
        'map_status_share': shares(row['map_filter_status'] for row in rows),
        'ego_status_share': shares(row['ego_status'] for row in rows),
        'raceline_status_share': shares(row['raceline_status'] for row in rows),
        'camera_state_share': shares(row['camera_state'] for row in tracks),
        'processing_ms': {'p50': percentile(times, .5), 'p95': percentile(times, .95)},
        'dropped_scans': int(rows[-1]['dropped_scans']) if rows else 0,
        'range_shells': range_shells(tracks),
        'far_vehicle_scans': sum(int(row['far_vehicle']) for row in tracks),
    }
    if labels:
        out['labels'] = label_summary(rows, labels)
    return out


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('association_csv', type=Path)
    parser.add_argument('--labels', type=Path)
    args = parser.parse_args()
    print(json.dumps(summarize(args.association_csv, args.labels), indent=2, sort_keys=True))


if __name__ == '__main__':
    main()
