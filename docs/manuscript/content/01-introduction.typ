= Introduction

Non-coding regions of the genome are well-known to be as important as coding regions for understanding how genotype determines phenotype @bib-human-annotation-heritability @bib-maize-chromatin-heritability.
Though tools like AlphaFold2 @bib-AlphaFold2 have dramatically improved our ability to study coding sequence, similarly performing tools do not yet exist for non-coding regions.
Nevertheless, over the last decade deep learning models have rapidly improved performance in predicting non-coding genomic features such as chromatin accessibility @bib-Basenji @bib-a2z, transcription factor binding @bib-BPNet @bib-maize-kmer-grammar, and RNA abundance @bib-Borzoi @bib-Enformer directly from DNA sequence.
These models can then be queried to highlight functional non-coding sites, which can be useful for filtering large sets of variants down to promising genome editing targets.
Further, since most of the modeling work has been done on human and mouse data, there is a need to benchmark their performance in plants.

Models that predict RNA abundance from sequence are particularly attractive due to the relatively cheap cost and standardized protocols of RNA-seq.
However, there is room for improvement in these models across a number of areas.
While RNA abundance models have shown high performance across genes, recent work in humans @bib-personal-transcriptome-variation has highlighted their lack of sensitivity across individuals.
Some expression model architectures @bib-Enformer @bib-Borzoi include coding sequence in the input, which is known to lead to overfitting on gene family instead of true regulatory sequence differences @bib-hai-jacob-cnn.
There is also a tendency to maximize data when training these models, without actually measuring the rate of diminishing returns for each additional observation.
Finally, while multiple species have been included in some training sets, it is common to test on a set of held-out chromosomes within the training species, rather than testing on a completely held-out species.

Deep learning models benefit from large and diverse training sets of different tissues and genotypes, which are rarely available outside model species.
To train RNA expression models on larger sample sizes, we leveraged new long-read genomes and RNA-seq data from 15 wild species of the Andropogoneae tribe.
Diverging around 17.5 million years ago @bib-andropogoneae-divergence, the Andropogoneae includes globally staple crop plants such as maize, sorghum, and sugarcane.
Millions of years of evolution within the tribe has provided a large, diverse pool of training alleles.
Sorghum and maize diverged around 12 million years ago (Mya), on the order of the human-chimpanzee split (6--10 Mya), but have a 10-fold higher rate of nucleotide divergence @bib-maize-sorghum-divergence @bib-human-chimp-divergence.

We tested four sequenced-based genomic deep learning architectures, DanQ @bib-DanQ, HyenaDNA @bib-HyenaDNA, FNetCompression @bib-FNetCompression, and a smaller version of Enformer @bib-Enformer, on their ability to predict across species and alleles.
DanQ is one of the earliest genomic deep learning architectures, leveraging a long short-term memory recurrent layer to learn the syntax and grammar of motifs detected by a convolutional layer.
Enformer is a massive transformer architecture with a context size near 100 kilobases that is among the best performing models for human expression prediction.
HyenaDNA is a novel architecture capable of handling long context windows of up to a million base pairs.
FNetCompression combines a fast Fourier transform with multi-head attention to efficiently learn from sequences of up to 10 kilobases with a few orders of magnitude less parameters than the other architectures.

We aimed to investigate, from a plant perspective, two major open questions in expression modeling from sequence:
1) How well do current sequence-based deep learning architectures generalize across species?
and 2) How sensitive are these models across individuals?
