# Data preprocessing for expression models

## Orthogroup assignment with eggnog-mapper

### Create proteomes and tx2gene maps

```
pixi shell
pip install ../../src/python/a2ze/
mkdir -p tmp/{orthogroups,gffutils}
find ../../data/genomes -mindepth 2 -maxdepth 2 -type d | python -c 'from pathlib import Path; import sys; import os; sys.stdout.write("\0".join((os.path.sep.join(Path(p.rstrip()).parts[4:6]) for p in sys.stdin)))' | parallel --null --jobs 6 --bar --halt 'soon,fail=1' 'mkdir --parents tmp/orthogroups/genomes/{} && ./create_proteome.py --gffutils-cache tmp/gffutils ../../data/genomes/{}/assembly.fa.gz ../../data/genomes/{}/annotation.gff.gz > tmp/orthogroups/genomes/{}/proteome.faa'
find ../../data/genomes -mindepth 2 -maxdepth 2 -type d | python -c 'from pathlib import Path; import sys; import os; sys.stdout.write("\0".join((os.path.sep.join(Path(p.rstrip()).parts[4:6]) for p in sys.stdin)))' | parallel --null --jobs 6 --bar --halt 'soon,fail=1' 'gzip -cd ../../data/genomes/{}/annotation.gff.gz | ./tx2gene.py > tmp/orthogroups/genomes/{}/tx2gene.tsv'
```

### Run eggnog-mapper

```
mkdir tmp/eggnog_data
download_eggnog_data.py -y --data_dir tmp/eggnog_data
parallel --jobs 4 --bar --halt 'soon,fail=1' 'mkdir --parents {//}/eggnog && emapper.py --cpu 2 --data_dir tmp/eggnog_data --temp_dir /tmp -i {} --output_dir {//}/eggnog -o eggnog' ::: tmp/orthogroups/genomes/*/*/proteome.faa
```
