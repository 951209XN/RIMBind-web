"""Build an upload directory containing source only; never includes weights."""
import argparse,shutil
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
a=argparse.ArgumentParser();a.add_argument('--output',type=Path,required=True)
a.add_argument('--flat',action='store_true',help='Five root files for file-only web upload dialogs')
args=a.parse_args()
if args.output.exists():raise SystemExit('Use a new output directory; refusing overwrite')
args.output.mkdir(parents=True)
for filename in ['README.md','requirements.txt']:
    shutil.copy2(ROOT/'deploy/hf-space'/filename,args.output/filename)
app_source=(ROOT/'deploy/hf-space/app.py').read_text()
if args.flat:
    original_import='from backend.inputs import parse_pdb'
    if app_source.count(original_import)!=1:raise RuntimeError('Unexpected preflight import layout')
    app_source=app_source.replace(original_import,'from inputs import parse_pdb')
    shutil.copy2(ROOT/'backend/inputs.py',args.output/'inputs.py')
else:
    (args.output/'backend').mkdir()
    for filename in ['__init__.py','inputs.py']:
        shutil.copy2(ROOT/'backend'/filename,args.output/'backend'/filename)
(args.output/'app.py').write_text(app_source)
shutil.copy2(ROOT/'LICENSE',args.output/'LICENSE')
print(args.output)
