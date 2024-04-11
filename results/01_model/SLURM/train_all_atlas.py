#!/usr/bin/env python3

import itertools
import os
from pathlib import Path
import subprocess

# create array of all combinations
architectures = ['DanQ', 'FNetCompression', 'HyenaDNA', 'Miniformer']
tasks = ['exp-max', 'exp-any', 'exp-leaf-abs', 'exp-leaf-bin']
replicates = range(3)
configurations = list(itertools.product(architectures, tasks, replicates))

## sanity check to make sure we set --array correctly
assert int(os.environ['SLURM_ARRAY_TASK_MAX']) < len(configurations)
configuration = configurations[int(os.environ['SLURM_ARRAY_TASK_ID'])]

# rsync dataset to $TMPDIR
dataset_source = Path('tmp/dataset')
subprocess.run(
    args = ['rsync', '--archive', dataset_source, os.environ['TMPDIR']],
    check = True
)

# run training
subenv = os.environ.copy()
subenv['A2ZE_TRAIN_MODEL'] = configuration[0]
subenv['A2ZE_TRAIN_TASK'] = configuration[1]
subprocess.run(
    args = ['pixi', 'run', 'train-atlas'],
    check = True,
    env = subenv
)

# clean up
subprocess.run(
    args = ['rm', '--recursive', '--force', Path(os.environ['TMPDIR']) / 'dataset'],
    check = True
)
