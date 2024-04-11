# Model Training

## Set up

```
ln -sr ../00_preprocessing/tmp/dataset tmp/dataset
```

## Environment package caching

If home directory is too slow to use for a cache (e.g. is mounted over a network), can temporarily set cache path to a more local mountpoint:

```
XDG_CACHE_HOME=/tmp/$USER-cache pixi install
```

## Train all models in Hydra on exp-any

```
CUDA_VISIBLE_DEVICES=0 python train.py --multirun data/task=exp-any data.preprocessing.num_workers=12 training.experiment_name=full training.compile_model=false training.validation_split_name=valOG,valSp,valSpOG model=DanQ,FNetCompression,HyenaDNA,Miniformer training.mlflow_tracking_uri='http://localhost:6000'
```

## Quickly uncompress dataset genomes

```
find tmp/dataset/genomes -name 'assembly.fa.gz' -print0 | parallel --jobs 2 --bar --null --halt 'soon,fail=1' 'bgzip -d -k -@ 3 {} && samtools faidx {.}'
```

## Start MLFlow tracking server

```
mlflow server --backend-store-uri sqlite:///tmp/mlflow/mlflow.db --default-artifact-root tmp/mlflow --host 127.0.0.1 --port 6000
```

## Export/import an experiment

```
MLFLOW_TRACKING_URI="path/to/mlflow" export-experiment --experiment 'nameORid' --output-dir path/to/out
# rsync over to MLFlow server
MLFLOW_TRACKING_URI="http://localhost:6000" import-experiment --experiment-name 'nameORid' --input-dir mlflow-export
```

## Run on SCINet Atlas

```
sbatch --job-name="train_all" --output='tmp/slurm-%A_%a.out' --error='tmp/slurm-%A_%a.err' --account=YOUR_ACCOUNT --nodes=1 --cpus-per-task=8 --mem-per-cpu=2G --time=24:00:00 --partition=gpu-a100 --gres=gpu:a100_1g.10gb:1 --array=0-47%8 SLURM/train_all_atlas.py
sbatch --job-name="predict_all" --output='tmp/slurm-%A_%a.out' --error='tmp/slurm-%A_%a.err' --account=YOUR_ACCOUNT --nodes=1 --cpus-per-task=8 --mem-per-cpu=2G --time=2:00:00 --partition=gpu-a100 --gres=gpu:a100_1g.10gb:1 --array=0-47%4 SLURM/predict_all_atlas.sh
tail --quiet -n+2 predictions.*.tsv | cat <(head -n1 predictions.0.tsv) - | gzip -c > predictions.tsv.gz
```

Note: Atlas can't seem to handle more than 16 concurrent GPU jobs.
Also, spacing the start times of groups of four jobs by 15 minutes seems to stop filesystem errors.

Further note: Batches of 8 jobs seems to avoid filesystem errors.

## Run predictions locally

```
parallel --jobs 1 --bar --halt 'soon,fail=1' 'python run_predictions.py --num_workers 10 "http://localhost:6000" full tmp/dataset tmp/predictions.{}.tsv {}' ::: $(seq 0 47)
tail --quiet -n+2 predictions.*.tsv | cat <(head -n1 predictions.0.tsv) - | gzip -c > predictions.tsv.gz
```

## Run ablation

```
python train.py --multirun data/task=exp-max,exp-any data.preprocessing.num_workers=12 training.experiment_name=ablation data.path=/tmp/dataset +experiment=ablation/n1,ablation/n2,ablation/n3,ablation/n4,ablation/n8,ablation/p1,ablation/p2,ablation/p3,ablation/p4,ablation/p8 training.mlflow_tracking_uri='http://localhost:6000'
```
