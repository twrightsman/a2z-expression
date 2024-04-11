= Discussion

Here we have shown that four genomic deep learning architectures are capable of generalizing across species, though they also show the same lack of allelic sensitivity seen in humans @bib-personal-transcriptome-variation.
FNetCompression's performance is particularly remarkable because it has several orders of magnitude fewer parameters than DanQ (57k versus 1.6m, respectively).
Large foundation models such as AgroNT @bib-AgroNT show promising results within the training species, but FNetCompression suggests smaller, more efficient models, perhaps also utilizing a fast Fourier transform, are worth further exploration.
Since the Pearson correlations we observed are still far from perfect, it is worthwhile to note that we do not expect _cis_ sequence-based models to ever reach perfect correlation, as _cis_ factors explain only a third of the genetic variation in expression in maize @bib-HARE.
The fact that our models show across individual performance in maize ($attach(r, br: s) = 0.092$) double that observed in humans @bib-personal-transcriptome-variation is puzzling.
Population genetics has shown that maize has an order of magnitude more genetic variance than humans @bib-maize-HapMap2, yet our models are generalizing across maize individuals better than what was observed across human individuals.
Unlike our validation set, our maize test set includes orthologs of sequences in our training set, which may result in slightly inflated performance estimates.
However, this inflation is expected to be less than when coding sequences are included in the model @bib-hai-jacob-cnn, as was the case for the human benchmark.
More work will be needed to investigate this and, more generally, where the remaining errors are being made in these models.

This stringent benchmark, both across species and across individuals in a held-out species, is something that all expression models should be continually evaluated against to get a better sense of generalizability than within species testing.
Our ablation results show that we are not yet saturated in terms of training data, meaning there is a need for further benchmarks on larger sets of species.
Training across species presents the opportunity to not only leverage larger data sets but to learn the general rules of eukaryotic gene expression.
Future work should consider training across many distantly-related species to learn general rules, then successively fine-tuning within clades to learn lineage-specific patterns.
Consideration of data balance may be necessary, as prevalent polyploidy within plants @bib-plant_alignment leads to vastly different gene counts across species and complicates transcript quantification.
However, scaling to bigger data will come with metadata challenges, exacerbated by the plethora of standards across databases @bib-BioSharing.
While considering better architectures with higher data needs, it will be increasingly important to better organize expression databases.
Lastly, with the rising utilization of foundation models trained across massive numbers of genomes, it will also be critical to maintain true hold-out species for fair model evaluation.
A CASP-like competition for RNA abundance modeling may be useful for this, as new sequence-based models of non-coding biology are developed.
