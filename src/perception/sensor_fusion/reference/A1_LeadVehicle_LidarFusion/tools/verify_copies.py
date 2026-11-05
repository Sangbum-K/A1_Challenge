from pathlib import Path
import hashlib
import json

root = Path(__file__).resolve().parents[1]
items = json.loads((root / 'COPY_MANIFEST.json').read_text())
for item in items:
    path = root / item['copy']
    assert not path.is_symlink(), path
    assert hashlib.sha256(path.read_bytes()).hexdigest() == item['sha256'], path
print(f'{len(items)} copied files: SHA256 verified (local copies only)')
