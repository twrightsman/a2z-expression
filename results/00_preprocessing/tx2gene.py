#!/usr/bin/env python

import sys


for line in sys.stdin:
  line = line.rstrip()
  if line and not line.startswith('#'):
    line_split = line.split("\t")
    if line_split[2] == "mRNA":
      attrs = dict((pair.split('=') for pair in line_split[-1].split(";")))
      transcript_id = attrs['ID']
      gene_id = attrs['Parent']
      print(f"{transcript_id}\t{gene_id}")
