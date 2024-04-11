# Interpretation

```
$ mkdir -p tmp/dataset/tasks/{exp-any,exp-max,exp-leaf-abs,exp-leaf-bin}
$ ln -sr ../01_model/tmp/dataset/genomes tmp/dataset/genomes
$ ln -sr ../01_model/tmp/dataset/tasks/exp-any/* tmp/dataset/tasks/exp-any/
$ ln -sr ../01_model/tmp/dataset/tasks/exp-max/* tmp/dataset/tasks/exp-max/
$ ln -sr ../01_model/tmp/dataset/tasks/exp-leaf-abs/* tmp/dataset/tasks/exp-leaf-abs/
$ ln -sr ../01_model/tmp/dataset/tasks/exp-leaf-bin/* tmp/dataset/tasks/exp-leaf-bin/
$ rm tmp/dataset/tasks/*/test.tsv
$ awk '(NR == 1) || ($1 == "Zea_mays/B73")' < ../01_model/tmp/dataset/tasks/exp-any/test.tsv > tmp/dataset/tasks/exp-any/test.tsv
$ awk '(NR == 1) || ($1 == "Zea_mays/B73")' < ../01_model/tmp/dataset/tasks/exp-max/test.tsv > tmp/dataset/tasks/exp-max/test.tsv
$ awk '(NR == 1) || ($1 == "Zea_mays/B73")' < ../01_model/tmp/dataset/tasks/exp-leaf-abs/test.tsv > tmp/dataset/tasks/exp-leaf-abs/test.tsv
$ awk '(NR == 1) || ($1 == "Zea_mays/B73")' < ../01_model/tmp/dataset/tasks/exp-leaf-bin/test.tsv > tmp/dataset/tasks/exp-leaf-bin/test.tsv
```
