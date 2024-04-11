#!/usr/bin/env python

import argparse
from collections import defaultdict
import hashlib
import logging
from pathlib import Path
import sys
from typing import Optional

import Bio.Seq
import Bio.SeqRecord
import Bio.SeqIO
import gffutils
from pysam import FastaFile

from a2ze.utils import create_unique_feature_id, sha256sum


def main(reference_path: Path, annotation_path: Path, gffutils_cache_path: Path):
    reference = FastaFile(str(reference_path))

    # load cached gffutils database if previously generated
    gffutils_cache_db_path = gffutils_cache_path / str(sha256sum(annotation_path))
    logging.debug("using '%s' as gffutils cache path", gffutils_cache_db_path)
    if gffutils_cache_db_path.exists():
        annotation = gffutils.FeatureDB(str(gffutils_cache_db_path))
    else:
        annotation = gffutils.create_db(str(annotation_path), str(gffutils_cache_db_path), id_spec = create_unique_feature_id, merge_strategy = 'create_unique')

    proteins = []
    for gene in annotation.features_of_type('gene'):
        for transcript in annotation.children(gene, featuretype='mRNA'):
            coding_sequence = []

            # extract CDS sequence
            for CDS in annotation.children(transcript, featuretype='CDS', order_by='start', reverse=(transcript.strand == '-')):
                sequence = Bio.Seq.Seq(reference.fetch(reference = CDS.seqid, start = CDS.start - 1, end = CDS.end))
                coding_sequence.append(sequence.reverse_complement() if transcript.strand == '-' else sequence)

            if coding_sequence and transcript.id:
                coding_sequence = Bio.Seq.Seq('').join(coding_sequence)
                if (len(coding_sequence) % 3) == 0:
                    protein = Bio.SeqRecord.SeqRecord(
                        seq = coding_sequence.translate(),
                        id = transcript.id,
                        name = '',
                        description = ''
                    )
                    if (len(protein) > 0) and (protein[-1] == '*') and ('*' not in protein[:-1]):
                        proteins.append(protein[:-1])
    if proteins:
        Bio.SeqIO.write(proteins, sys.stdout, 'fasta')


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Uses a reference genome and annotation to prepare one set of inputs to OrthoFinder")

    parser.add_argument(
        "-v",
        "--verbose",
        help="output progress and other informative messages",
        action="count",
        dest="verbosity",
        default=0,
    )

    parser.add_argument("reference_path", type=Path, metavar="path/to/reference.fa")
    parser.add_argument("annotation_path", type=Path, metavar="path/to/annotation.gff3")

    parser.add_argument(
        "--gffutils-cache",
        type = Path,
        help = "where to cache gffutil databases",
        default = '.gffutils'
    )

    args = parser.parse_args()

    logging.basicConfig(
        format="{asctime} [{module}:{levelname}] {message}",
        style="{",
        level=max(logging.DEBUG, logging.WARNING - (args.verbosity * 10)),
    )

    if not args.gffutils_cache.exists():
        args.gffutils_cache.mkdir()

    main(
        reference_path = args.reference_path,
        annotation_path = args.annotation_path,
        gffutils_cache_path = args.gffutils_cache
    )
