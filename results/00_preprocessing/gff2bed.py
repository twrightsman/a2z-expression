#!/usr/bin/env python

import sys


for line in sys.stdin:
    if line.startswith('#'):
        continue

    seqid, source, feature_type, start, end, score, strand, phase, attributes = line.rstrip().split("\t")
    start = int(start)
    end = int(end)

    attributes = {k: v for k, v in (kv.split('=') for kv in attributes.rstrip(';').split(';'))}

    interval_name = None
    if 'ID' in attributes:
        interval_name = attributes['ID']

    strand = strand if strand != '?' else '.'

    if start <= end:
        print(f"{seqid}\t{start - 1}\t{end}\t{interval_name}\t.\t{strand}")
