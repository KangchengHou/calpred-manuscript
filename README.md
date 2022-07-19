# CalPGS (Calibrated PGS)

To install the package,
```bash
git clone git@github.com:KangchengHou/calpgs.git && cd calpgs
pip install -e .
```

## CLI for `CalPGS`
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
