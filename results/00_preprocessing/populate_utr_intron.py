#!/usr/bin/env python
import argparse
from collections import defaultdict
import logging
from pathlib import Path
from typing import Generator

import gffutils


def get_left_UTR_boundaries(
    exons: tuple[gffutils.Feature, ...], cds: tuple[gffutils.Feature, ...]
) -> Generator[tuple[gffutils.Feature, int, int], None, None]:
    last_exon_idx = 0
    while cds[0].start > exons[last_exon_idx].end:
        last_exon_idx += 1
    for i in range(last_exon_idx):
        yield (exons[i], exons[i].start, exons[i].end)
    if cds[0].start > exons[last_exon_idx].start:
        yield (exons[last_exon_idx], exons[last_exon_idx].start, cds[0].start - 1)


def get_right_UTR_boundaries(
    exons: tuple[gffutils.Feature, ...], cds: tuple[gffutils.Feature, ...]
) -> Generator[tuple[gffutils.Feature, int, int], None, None]:
    first_exon_idx = len(exons) - 1
    while cds[-1].end < exons[first_exon_idx].start:
        first_exon_idx -= 1
    if cds[-1].end < exons[first_exon_idx].end:
        yield (exons[first_exon_idx], cds[-1].end + 1, exons[first_exon_idx].end)
    for i in range(first_exon_idx + 1, len(exons)):
        yield (exons[i], exons[i].start, exons[i].end)


def create_UTRs(
    db: gffutils.FeatureDB,
    exon_featuretype="exon",
    cds_featuretype="CDS",
    transcript_featuretype="mRNA",
    new_utr5_featuretype="five_prime_UTR",
    new_utr3_featuretype="three_prime_UTR",
    merge_attributes: bool = True,
) -> Generator[gffutils.Feature, None, None]:
    for transcript in db.features_of_type(featuretype=transcript_featuretype):
        exons = tuple(
            db.children(transcript, featuretype=exon_featuretype, order_by="start")
        )
        cds = tuple(
            db.children(transcript, featuretype=cds_featuretype, order_by="start")
        )

        if exons and cds:
            # yield left UTR chunks, assign 5'/3' based on transcript strand
            i = 0
            for exon, start, end in get_left_UTR_boundaries(exons, cds):
                attr = {"ID": f"utr-{transcript.id}-{i+1}"}
                if merge_attributes:
                    attr = gffutils.helpers.merge_attributes(
                        attr, {k: v for k, v in exon.attributes.items() if k != "ID"}
                    )

                yield gffutils.Feature(
                    seqid=exon.seqid,
                    source='gffutils_derived',
                    featuretype=new_utr3_featuretype
                    if (exon.strand == "-")
                    else new_utr5_featuretype,
                    start=start,
                    end=end,
                    strand=exon.strand,
                    attributes=attr,
                    dialect=db.dialect,
                    keep_order=db.keep_order,
                    sort_attribute_values=db.sort_attribute_values,
                )
                i += 1

            # yield right UTR chunks, assign based on strand
            for exon, start, end in get_right_UTR_boundaries(exons, cds):
                attr = {"ID": f"utr-{transcript.id}-{i+1}"}
                if merge_attributes:
                    attr = gffutils.helpers.merge_attributes(
                        attr, {k: v for k, v in exon.attributes.items() if k != "ID"}
                    )

                yield gffutils.Feature(
                    seqid=exon.seqid,
                    source='gffutils_derived',
                    featuretype=new_utr5_featuretype
                    if (exon.strand == "-")
                    else new_utr3_featuretype,
                    start=start,
                    end=end,
                    strand=exon.strand,
                    attributes=attr,
                    dialect=db.dialect,
                    keep_order=db.keep_order,
                    sort_attribute_values=db.sort_attribute_values,
                )
                i += 1


def main(annotation_path: Path):
    annotation = gffutils.create_db(
        str(annotation_path), ":memory:", merge_strategy="create_unique"
    )

    for utr in create_UTRs(annotation):
        print(utr)

    transcript_intron_count = defaultdict(int)
    for intron in annotation.create_introns():
        # fix the intron ID, since gffutils gives two IDs to the intron for the flanking exons...
        parent_transcript_id = intron.attributes['Parent'][0]
        transcript_intron_count[parent_transcript_id] += 1
        intron.attributes['ID'] = [parent_transcript_id + f".intron.{transcript_intron_count[parent_transcript_id]}"]
        print(intron)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Computes UTRs and introns for a given annotation file"
    )

    parser.add_argument(
        "-v",
        "--verbose",
        help="output progress and other informative messages",
        action="count",
        dest="verbosity",
        default=0,
    )

    parser.add_argument("annotation_path", type=Path, metavar="path/to/annotation.gff3")

    args = parser.parse_args()

    logging.basicConfig(
        format="{asctime} [{module}:{levelname}] {message}",
        style="{",
        level=max(logging.DEBUG, logging.WARNING - (args.verbosity * 10)),
    )

    main(annotation_path=args.annotation_path)
