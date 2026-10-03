"""Embed the lab source into a single Kaggle notebook."""
from pathlib import Path
import nbformat as nbf

code_dir = Path(__file__).resolve().parent
submission = code_dir.parent
repo = code_dir.parents[2]
nb = nbf.v4.new_notebook()
nb.metadata.update({"kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
                    "language_info": {"name": "python"}, "accelerator": "GPU"})
cells = [nbf.v4.new_markdown_cell("# DeepWeeds Lab Day 2\n\nSelect **GPU T4 x2** and enable **Internet** in Kaggle Notebook settings, then use **Save & Run All**. Outputs are written to `/kaggle/working/lab_output`. The smoke stage opens train and validation only. The full stage chooses the recipe and inference on validation before evaluating test."),
         nbf.v4.new_code_cell("import os, torch\nassert torch.cuda.is_available(), 'Enable GPU in Kaggle settings'\nassert torch.cuda.device_count() >= 2, 'Select T4 x2 in Kaggle settings'\nprint([torch.cuda.get_device_name(i) for i in range(torch.cuda.device_count())])\n%pip -q install timm==1.0.30 fvcore==0.1.5.post20221221 openpyxl")]
for path in [repo / "eval.py", *[code_dir / name for name in
              ("dataset.py", "model.py", "losses.py", "train.py", "inference.py", "benchmark.py", "run_lab.py", "test_lab.py")]]:
    target = Path("/kaggle/working") / path.relative_to(repo)
    cells.append(nbf.v4.new_code_cell(f"from pathlib import Path\np = Path({str(target)!r})\np.parent.mkdir(parents=True, exist_ok=True)\np.write_text({path.read_text()!r}, encoding='utf-8')\nprint(p)"))
cells.append(nbf.v4.new_code_cell("""from pathlib import Path
import hashlib, urllib.request, zipfile
data = Path('/kaggle/working/data')
(data / 'labels').mkdir(parents=True, exist_ok=True)
base = 'https://raw.githubusercontent.com/AlexOlsen/DeepWeeds/master/labels/'
for name in ('labels.csv', 'train_subset0.csv', 'val_subset0.csv', 'test_subset0.csv'):
    target = data / 'labels' / name
    if not target.exists():
        urllib.request.urlretrieve(base + name, target)
archive = data / 'images.zip'
if not archive.exists():
    urllib.request.urlretrieve('https://zenodo.org/records/7939060/files/images.zip?download=1', archive)
digest = hashlib.md5(archive.read_bytes()).hexdigest()
assert digest == 'b7b30f96d466fba86016aa5a26606e0f', digest
images = data / 'images'
if len(list(images.glob('*.jpg'))) != 17509:
    images.mkdir(exist_ok=True)
    with zipfile.ZipFile(archive) as z:
        z.extractall(images)
assert len(list(images.glob('*.jpg'))) == 17509
print('Verified 17,509 images and original fold-0 labels')"""))
cells.append(nbf.v4.new_code_cell("""import subprocess, sys
script = '/kaggle/working/submissions/2A202602982_HaManhTuan/code/run_lab.py'
os.environ['DEEPWEEDS_DATA'] = '/kaggle/working/data'
subprocess.run([sys.executable, '-m', 'unittest', 'discover', '-s',
                '/kaggle/working/submissions/2A202602982_HaManhTuan/code',
                '-p', 'test_*.py', '-v'], check=True)
subprocess.run([sys.executable, script, 'smoke', '--data', '/kaggle/working/data',
                '--output', '/kaggle/working/lab_smoke'], check=True)"""))
cells.append(nbf.v4.new_code_cell("""import shutil
try:
    subprocess.run([sys.executable, script, 'full', '--data', '/kaggle/working/data',
                    '--output', '/kaggle/working/lab_output', '--epochs', '10',
                    '--final-epochs', '12'], check=True)
finally:
    if Path('/kaggle/working/lab_output').exists():
        print('Evidence archive:', shutil.make_archive('/kaggle/working/deepweeds_evidence',
                                                     'zip', root_dir='/kaggle/working/lab_output'))"""))
cells.append(nbf.v4.new_code_cell("""import shutil
from pathlib import Path
out = Path('/kaggle/working/lab_output')
assert (out / 'evidence' / 'completed.json').is_file()
print('Evidence archive:', '/kaggle/working/deepweeds_evidence.zip')
print('Report:', out / 'report.md')
print('Results:', out / 'results.xlsx')"""))
nb.cells = cells
target = submission / "deepweeds_kaggle_t4x2.ipynb"
nbf.write(nb, target)
print(target)
