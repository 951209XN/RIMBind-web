"""Build an upload directory containing source only; never includes weights."""
import argparse,shutil
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
a=argparse.ArgumentParser();a.add_argument('--output',type=Path,required=True);args=a.parse_args()
if args.output.exists():raise SystemExit('Use a new output directory; refusing overwrite')
args.output.mkdir(parents=True)
for filename in ['README.md','requirements.txt','app.py']:
    shutil.copy2(ROOT/'deploy/hf-space'/filename,args.output/filename)
(args.output/'backend').mkdir()
for filename in ['__init__.py','inputs.py']:
    shutil.copy2(ROOT/'backend'/filename,args.output/'backend'/filename)
shutil.copy2(ROOT/'LICENSE',args.output/'LICENSE')
print(args.output)
