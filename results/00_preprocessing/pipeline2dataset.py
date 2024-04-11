# ---
# jupyter:
#   jupytext:
#     formats: ipynb,py:percent
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.16.1
#   kernelspec:
#     display_name: Python 3 (ipykernel)
#     language: python
#     name: python3
# ---

# %% [markdown]
# # Overview
#
# This notebook will convert the output of `quantify-RNA-pipeline` into a convenient format for data loading.
#
# ## Dataset format
#
# ```
# dataset/
#   genomes/
#   tasks/
#     task_name/
#       train.tsv
#       validate.tsv
#       test.tsv
# ```
#
# ## Imports / Globals

# %%
from collections import defaultdict
import functools
import gzip
import hashlib
import itertools
import json
import logging
import math
import multiprocessing.pool
import operator
from pathlib import Path
from pprint import pprint
import subprocess

import gffutils
import IPython.display
import matplotlib.figure
import numpy as np
import polars as pl
import sklearn.linear_model
from tqdm.auto import tqdm

from a2ze.utils import create_unique_feature_id, sha256sum

# %matplotlib inline

# %%
logger = logging.getLogger(__name__)

# %%
MINIMUM_READS_MAPPED_TO_KEEP = 5_000_000

OUTPUT_DIRECTORY = Path("tmp/dataset")

if not OUTPUT_DIRECTORY.exists():
    OUTPUT_DIRECTORY.mkdir()

gffutils_cache_path = Path("tmp/gffutils")
if not gffutils_cache_path.exists():
    gffutils_cache_path.mkdir()

# %% [markdown]
# # Filter samples

# %%
pipeline_directory = Path("tmp/quantify-RNA-pipeline-results")
orthogroups_root_directory = Path("tmp/orthogroups/genomes")

# %%
reads_mapped = {
    'sample_id': [],
    'reads_mapped': []
}

for meta_info_path in pipeline_directory.glob('quants/*/aux_info/meta_info.json'):
    with open(meta_info_path) as meta_info_file:
        num_reads_mapped = json.load(meta_info_file)['num_mapped']
        reads_mapped['sample_id'].append(meta_info_path.parts[-3])
        reads_mapped['reads_mapped'].append(num_reads_mapped)

reads_mapped = pl.DataFrame(reads_mapped)

# %%
tissues_to_keep = {
    'Leaf, Early senescence',
    'Leaf base, Adult',
    'Leaf tip, Adult',
    'Young tiller, Growing point differentiation',
    'Flower, Flowering',
    'leaf, LP.11 eleven leaves visible stage',  # Leaf, Early senescence (NAM)
    'leaf base, LP.11 eleven leaves visible stage',  # Leaf base, Adult (NAM)
    'leaf tip, LP.11 eleven leaves visible stage',  # Leaf tip, Adult (NAM)
    'shoot, seedling',  # Young tiller, Growing point differentiation (NAM)
    'tassel, LP.18 eighteen leaves visible stage',  # Flower, Flowering (NAM)
    'ear inflorescence, LP.18 eighteen leaves visible stage',  # Flower, Flowering (NAM)
}

is_tissue_to_keep = (pl.col.organ + ', ' + pl.col.age).is_in(tissues_to_keep)
has_enough_reads = pl.col.reads_mapped >= MINIMUM_READS_MAPPED_TO_KEEP

samples = pl.read_csv(
    pipeline_directory / 'samples.tsv',
    separator = "\t"
).join(
    other = reads_mapped,
    on = 'sample_id',
    how = 'inner'
).drop(
    'replicate'
).filter(
    is_tissue_to_keep & has_enough_reads
)

# %%
with pl.Config(tbl_rows = 2):
    IPython.display.display(samples)

# %%
samples.group_by('organ', 'age').len()

# %%
genomes_with_more_than_three_tissues = samples.group_by(
    'species', 'genotype', 'organ', 'age'
).len().group_by(
    'species', 'genotype'
).len().filter(
    pl.col.len >= 3
).select(
    pl.col.species,
    pl.col.genotype
)

# %% [markdown]
# # Link genome files

# %%
for species, genotype in samples.select(pl.col.species, pl.col.genotype).unique().rows():
    for name in ('assembly.fa.gz', 'assembly.fa.gz.fai', 'assembly.fa.gz.gzi'):
        src = pipeline_directory / 'data' / 'genomes' / species.replace(' ', '_') / genotype.replace(' ', '_') / name
        dst = OUTPUT_DIRECTORY / 'genomes' / species.replace(' ', '_') / genotype.replace(' ', '_') / name

        if not dst.parent.exists():
            dst.parent.mkdir(parents = True)

        if not dst.exists():
            if dst.is_symlink():
                # remove broken symlink before replacing
                dst.unlink()
            dst.symlink_to(target = src.resolve())

# %% [markdown]
# # Create gene intervals

# %%
tx2ogGene = []

for species, genotype in samples.select(pl.col.species, pl.col.genotype).unique().rows():
    tx2gene = pl.scan_csv(
        source = orthogroups_root_directory / f"{species.replace(' ', '_')}/{genotype.replace(' ', '_')}/tx2gene.tsv",
        separator = "\t",
        has_header = False,
    ).select(
        pl.col.column_1.alias('transcript'),
        pl.col.column_2.alias('gene')
    )

    tx2ogGene_genome = pl.scan_csv(
        source = orthogroups_root_directory / f"{species.replace(' ', '_')}/{genotype.replace(' ', '_')}/eggnog/eggnog.emapper.annotations",
        separator = "\t",
        comment_prefix = '#',
        has_header = False
    ).select(
        pl.col.column_1.alias('transcript'),
        pl.col.column_5.alias('orthogroups')
    ).with_columns(
        pl.col.orthogroups.str.split(',')
    ).explode(
        'orthogroups'
    ).rename(
        {'orthogroups': 'orthogroup'}
    ).select(
        pl.col.transcript,
        pl.col.orthogroup.str.split('@').list.first(),
        pl.col.orthogroup.str.split('|').list.last().alias('taxa')
    ).filter(
        pl.col.taxa == 'Poales'
    ).select(
        pl.col.transcript,
        pl.col.orthogroup
    ).join(
        other = tx2gene,
        on = ['transcript']
    ).with_columns(
        pl.lit(species).alias('species'),
        pl.lit(genotype).alias('genotype')
    )

    tx2ogGene.append(tx2ogGene_genome)

tx2ogGene = pl.concat(tx2ogGene)

# %%
# %%time
features_cache_path = Path("tmp/pipeline2dataset.features.tsv")

if features_cache_path.exists():
    features = pl.read_csv(
        features_cache_path,
        separator = "\t"
    )
else:
    features = defaultdict(list)
    
    for species, genotype in tqdm(samples.select(pl.col.species, pl.col.genotype).unique().rows(), desc = 'processing features'):
        annotation_path = pipeline_directory / f"data/genomes/{species.replace(' ', '_')}/{genotype.replace(' ', '_')}/annotation.gff.gz"
    
        # load cached gffutils database if previously generated
        gffutils_cache_db_path = gffutils_cache_path / str(sha256sum(annotation_path))
        if gffutils_cache_db_path.exists():
            annotation = gffutils.FeatureDB(str(gffutils_cache_db_path))
        else:
            annotation = gffutils.create_db(str(annotation_path), str(gffutils_cache_db_path), id_spec = create_unique_feature_id, merge_strategy = 'create_unique')
    
        # iterate over transcripts
        for feature in annotation.features_of_type('mRNA'):
            most_upstream_CDS = next(annotation.children(feature.id, featuretype = 'CDS', order_by = 'start', reverse = feature.strand == '-'))
            start = most_upstream_CDS.end if feature.strand == '-' else most_upstream_CDS.start
            TSS = feature.end if feature.strand == '-' else feature.start
    
            features['transcript'].append(feature.id)
            features['species'].append(species)
            features['genotype'].append(genotype)
            features['seqid'].append(feature.seqid)
            # translate GFF 1-based coordinates to 0-based Python coordinates
            features['start'].append(start - 1)
            features['end'].append(start)
            features['TSS'].append(TSS - 1)
            features['strand'].append(feature.strand)
    
    features = pl.DataFrame(
        data = features
    ).lazy().join(
        other = tx2ogGene,
        on = ['species', 'genotype', 'transcript'],
        how = 'inner'
    ).join(  # only keep genomes with more than three tissues to have comparable max expression across tissues
        other = genomes_with_more_than_three_tissues.lazy(),
        on = ['species', 'genotype'],
        how = 'inner'
    ).collect()

    features.write_csv(
        features_cache_path,
        separator = "\t"
    )

# %%
with pl.Config(tbl_rows = 2):
    IPython.display.display(features)

# %% [markdown]
# # Split into training and validation
#
# - orthogroups get split into training or validation
# - species get training, validation, or test
#     - test = _Zea mays_ (see NAM section below)
#     - validation = _Zea *_
#     - training = the rest

# %%
orthogroups = features['orthogroup'].unique().sort().shuffle(seed = 42)

# %%
prop_train = 0.9
n_train = math.ceil(len(orthogroups) * prop_train)

# %%
orthogroups_train = set(orthogroups[:n_train])
orthogroups_val = set(orthogroups[n_train:])

# %%
species = set(features['species'].unique())

# %%
species_test = {'Zea mays'}
species_ignore = {'Zea mays mexicana'}
species_validation = set(filter(lambda s: s.startswith('Zea ') or (s == 'Tripsacum zopilotense'), species)) - species_test - species_ignore

# %%
species_validation

# %%
species_training = species - species_validation - species_test - species_ignore

# %%
pprint(species_training, compact = True)

# %% [markdown]
# # Quantification processing

# %%
quantifications_raw = []

for row in samples.rows(named = True):
    quantification_raw = pl.scan_csv(
        source = pipeline_directory / 'quants' / row['sample_id'] / 'quant.sf',
        separator = "\t"
    ).select(
        pl.lit(row['sample_id']).alias('sample_id'),
        pl.col.Name.alias('transcript'),
        pl.col.TPM
    ).join(
        other = samples.lazy(),
        on = 'sample_id',
        how = 'inner'
    ).select(
        pl.col.sample_id,
        pl.col.species,
        pl.col.genotype,
        pl.col.organ,
        pl.col.age,
        pl.col.transcript,
        pl.col.TPM
    )
    quantifications_raw.append(quantification_raw)

quantifications_raw = pl.concat(quantifications_raw)

# %%
quantifications_repAvged = quantifications_raw.join(
    other = features.lazy(),
    on = ['species', 'genotype', 'transcript'],
    how = 'inner'
).group_by(
    'species', 'genotype', 'organ', 'age', 'transcript'
).agg(
    pl.col.TPM.mean()  # average across tissue replicates
).select(
    pl.col.species,
    pl.col.genotype,
    pl.col.transcript,
    (pl.col.organ + ', ' + pl.col.age).alias('tissue'),
    pl.col.TPM
).join(
    other = features.lazy(),
    on = ['species', 'genotype', 'transcript'],
    how = 'inner'
).cache()

# %%
features_highestExp = quantifications_repAvged.group_by(
    'species', 'genotype', 'gene'
).agg(
    pl.col.transcript.sort_by(pl.col.TPM).last(),  # get highest-expressed transcript (in any one tissue) per gene
    pl.col.TSS.sort_by(pl.col.TPM).last()
)

# %%
quantifications_repAvged_geneTSSsummed = quantifications_repAvged.group_by(
    'species', 'genotype', 'tissue', 'gene', 'TSS',
).agg(
    pl.col.TPM.sum()  # sum TPM for transcripts of same gene sharing same TSS
).join(
    other = features_highestExp,
    on = ['species', 'genotype', 'gene', 'TSS'],
    how = 'inner'
).drop(
    'TSS'
)

# %%
# %%time
data = quantifications_repAvged_geneTSSsummed.join(
    other = features.lazy(),
    on = ['species', 'genotype', 'transcript'],
    how = 'inner'
).select(
    pl.col.species,
    (pl.col.species.str.replace_all(' ', '_') + '/' + pl.col.genotype.str.replace_all(' ', '_')).alias('genome'),
    pl.col.transcript.alias('id'),
    pl.col.seqid,
    pl.col.start,
    pl.col.end,
    pl.col.strand,
    pl.col.orthogroup,
    pl.col.TPM,
    pl.col.tissue
).collect()

# %%
with pl.Config(tbl_rows = 2):
    IPython.display.display(data)


# %% [markdown]
# # Create task data
#
# - Max expression (per genotype, across tissues)
# - On/off expression (per genotype, across tissues)

# %%
def dump_task_data_splits(quantifications: pl.DataFrame, task_dir: Path):
    if not task_dir.exists():
        task_dir.mkdir(parents = True)

    # train
    quantifications.filter(
        pl.col('species').is_in(species_training) & pl.col('orthogroup').is_in(orthogroups_train)
    ).drop(
        'species'
    ).sort(
        by = ['genome', 'id']
    ).write_csv(
        file = task_dir / 'train.tsv',
        separator = "\t"
    )

    # valOG
    quantifications.filter(
        pl.col('species').is_in(species_training) & pl.col('orthogroup').is_in(orthogroups_val)
    ).drop(
        'species'
    ).sort(
        by = ['genome', 'id']
    ).write_csv(
        file = task_dir / 'valOG.tsv',
        separator = "\t"
    )

    # valSp
    quantifications.filter(
        pl.col('species').is_in(species_validation) & pl.col('orthogroup').is_in(orthogroups_train)
    ).drop(
        'species'
    ).sort(
        by = ['genome', 'id']
    ).write_csv(
        file = task_dir / 'valSp.tsv',
        separator = "\t"
    )

    # valSpOG
    quantifications.filter(
        pl.col('species').is_in(species_validation) & pl.col('orthogroup').is_in(orthogroups_val)
    ).drop(
        'species'
    ).sort(
        by = ['genome', 'id']
    ).write_csv(
        file = task_dir / 'valSpOG.tsv',
        separator = "\t"
    )

    # test
    quantifications.filter(
        pl.col('species').is_in(species_test)
    ).drop(
        'species'
    ).sort(
        by = ['genome', 'id']
    ).write_csv(
        file = task_dir / 'test.tsv',
        separator = "\t"
    )


# %% [markdown]
# ## Leaf base absolute

# %%
quantifications_leaf_abs = data.filter(
    (pl.col('tissue') == 'Leaf base, Adult') | (pl.col('tissue') == 'leaf base, LP.11 eleven leaves visible stage')
).drop(
    'tissue'
).rename({
    'TPM': 'targets'
})

# %%
dump_task_data_splits(quantifications_leaf_abs, OUTPUT_DIRECTORY / 'tasks' / 'exp-leaf-abs')

# %% [markdown]
# ## Leaf base on/off

# %%
quantifications_leaf_bin = quantifications_leaf_abs.with_columns(
    (pl.col('targets') > 0).cast(int)
)

# %%
dump_task_data_splits(quantifications_leaf_bin, OUTPUT_DIRECTORY / 'tasks' / 'exp-leaf-bin')

# %% [markdown]
# ## Max across tissues

# %%
quantifications_max = data.drop(
    'tissue'
).rename({
    'TPM': 'targets'
}).group_by(
    'species', 'genome', 'id'
).agg(
    pl.col('seqid').first(),  # take the first because across tissues this should all be the same
    pl.col('start').first(),
    pl.col('end').first(),
    pl.col('strand').first(),
    pl.col('orthogroup').first(),
    pl.col('targets').max()
).filter(
    pl.col.genome.is_in(set(quantifications_leaf_abs['genome'].unique()))  # only use genomes also in leaf dataset
)

# %%
dump_task_data_splits(quantifications_max, OUTPUT_DIRECTORY / 'tasks' / 'exp-max')

# %% [markdown]
# ## Off/on (pseudogene) across tissues

# %%
quantifications_any = quantifications_max.with_columns(
    (pl.col('targets') > 0).cast(int)
)

# %%
dump_task_data_splits(quantifications_any, OUTPUT_DIRECTORY / 'tasks' / 'exp-any')

# %%
