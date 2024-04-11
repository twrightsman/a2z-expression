Non-coding regions of the genome are just as important as coding regions for understanding the mapping from genotype to phenotype.
Interpreting deep learning models trained on RNA-seq is an emerging method to highlight functional sites within non-coding regions.
Most of the work on RNA abundance models has been done within humans and mice, with little attention paid to plants.
Here, we benchmark four genomic deep learning model architectures with genomes and RNA-seq data from 18 species closely related to maize and sorghum within the Andropogoneae.
The Andropogoneae are a tribe of C4 grasses that have adapted to a wide range of environments worldwide since diverging 18 million years ago.
Hundreds of millions of years of evolution across these species has produced a large, diverse pool of training alleles across species sharing a common physiology.
As model input, we extracted 1,026 base pairs upstream of each gene’s translation start site.
We held out maize as our test set and two closely related species as our validation set, training each architecture on the remaining Andropogoneae genomes.
Within a panel of 26 maize lines, all architectures predict expression across genes moderately well but poorly across alleles.
DanQ consistently ranked highest or second highest among all architectures yet performance was generally very similar across architectures despite orders of magnitude differences in size.
This suggests that state-of-the-art supervised genomic deep learning models are able to generalize moderately well across related species but not sensitively separate alleles within species, the latter of which agrees with recent work within humans.
We are releasing the preprocessed data and code for this work as a community benchmark to evaluate new architectures on our across-species and across-allele tasks.
