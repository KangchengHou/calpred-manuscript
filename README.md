# CalPred (Calibrated Prediction Intervals for Polygenic Scores Across Diverse Contexts)

This repository contains the source code an initial version of CalPred and analysis script used in the manuscript. For a more easy-to-use software package, refer to [CalPred github repo](https://github.com/KangchengHou/calpred).

The following are useful for those interested in exploring this repository.

## Installation
To install the initial version package contained in this manuscript,
```bash
git clone git@github.com:KangchengHou/calpgs.git && cd calpgs
pip install -e .
```

## CLI
Calculate R2 differences between PGS and phenotype across covariate groups.
```bash
toy=tests/test-data/toy.tsv
calpgs group-stats \
    --df ${toy} \
    --y y_cov \
    --pred pgs \
    --group age \
    --out out
```

Modeling the phenotype as a function of covariates.
```bash
calpgs model \
    --df <df_path> \
    --out <model_path>
```

Apply the estimated model to new data
```bash
calpgs predict \
    --df <df_path> \
    --model <model_path> \
    --out <pred_path> \
    --ci 0.9
```

## File structures

- `calpgs/`: source code
- `notebooks/`: example analysis.
- `experiments/`: code to replicate all main figures.
