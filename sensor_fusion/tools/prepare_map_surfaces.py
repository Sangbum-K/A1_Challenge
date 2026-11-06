#!/usr/bin/env python3
"""Offline map export/audit; never launches sensors or publishes ROS messages."""
import argparse
import hashlib
import json
from pathlib import Path
import subprocess
import os
import tempfile


def digest(path):
    result = hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            result.update(block)
    return result.hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('map', type=Path)
    parser.add_argument('--binary', required=True, type=Path)
    args = parser.parse_args()
    source = args.map.resolve(strict=True)
    if source.suffix != '.mm':
        parser.error('input must be the exact localization .mm map')
    output = source.with_suffix('.surfaces.pcd')
    manifest = source.with_suffix('.surfaces.json')
    original = digest(source)
    fd, temporary = tempfile.mkstemp(prefix='.surfaces-', suffix='.pcd', dir=source.parent)
    os.close(fd)
    temporary = Path(temporary)
    try:
        subprocess.run([str(args.binary.resolve(strict=True)), str(source), str(temporary)], check=True)
        if digest(source) != original:
            raise RuntimeError('source map changed during export')
        metadata = dict(schema=1, algorithm='planar_surfaces_v1', source=str(source),
                        source_sha256=original, surface_sha256=digest(temporary),
                        voxel_m=0.2, plane_radius_m=1.2,
                        note='Geometric floor/vertical-plane candidates, not semantic wall labels.')
        temporary.replace(output)
        manifest.write_text(json.dumps(metadata, indent=2) + '\n')
        print(manifest)
    finally:
        temporary.unlink(missing_ok=True)


if __name__ == '__main__':
    main()
