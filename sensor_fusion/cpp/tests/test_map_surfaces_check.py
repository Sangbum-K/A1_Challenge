"""prepare_map_surfaces.py --check on hand-made files; says nothing about the surfaces themselves."""
import hashlib,importlib.util,json,tempfile
from pathlib import Path

spec=importlib.util.spec_from_file_location('prepare_map_surfaces',Path(__file__).resolve().parents[2]/'tools'/'prepare_map_surfaces.py')
prepare=importlib.util.module_from_spec(spec);spec.loader.exec_module(prepare)

def sha(data):return hashlib.sha256(data).hexdigest()

def make_map(folder):
    (folder/'map.mm').write_bytes(b'map');(folder/'map.surfaces.pcd').write_bytes(b'surfaces')
    (folder/'map.surfaces.json').write_text(json.dumps(dict(source_sha256=sha(b'map'),surface_sha256=sha(b'surfaces'))))

def test_check():
    with tempfile.TemporaryDirectory() as name:
        folder=Path(name).resolve();make_map(folder)
        for given in (folder/'map',folder/'map.mm',folder/'map.simplemap'):
            assert prepare.check(given)==folder/'map.surfaces.pcd',given
        (folder/'map.mm').write_bytes(b'other map')
        try:prepare.check(folder/'map');raise AssertionError('changed map accepted')
        except RuntimeError:pass
        make_map(folder);(folder/'map.surfaces.pcd').write_bytes(b'edited')
        try:prepare.check(folder/'map');raise AssertionError('edited cache accepted')
        except RuntimeError:pass
        (folder/'map.surfaces.pcd').unlink()
        try:prepare.check(folder/'map');raise AssertionError('missing cache accepted')
        except FileNotFoundError:pass

if __name__=='__main__':
    test_check();print('map surface cache check: PASS')
