# CalPGS (Calibrated PGS)

To install the package,
```bash
git clone git@github.com:KangchengHou/calpgs.git && cd calpgs
pip install -e .
```

## CLI for using the package
Comparing difference R2 between PGS and phenotype across individual groups.
```bash
toy=tests/test-data/toy.tsv
calpgs r2diff \
    --df ${toy} \
    --y y_cov \
    --pred pgs \
    --group age \
    --out out
```

```bash
calpgs model \
    --df <df_path> \
    --y <response variable> \
    --pred
    --out <model_path>
```

We take input of PGS point estimate (mean), as well as the \alpha-level credible interval (lower-ci, upper-ci), in addition to a set of individuals used to perform the calibration. We aim to produce a tight and calibrated prediction interval.

The pipeline goes as follows:
1. Obtaining the PGS prediction (mean + std) of genetic risk for every individual in the data set (since PRS is usually consisting of hundreds of SNPs, the error terms are likely well-approximated by a Gaussian, which can be described with only mean + std)
2. Since step 1 only contains only genetic risk (the std only capture the genetic risk uncertainty). We perform two steps:
    (a)
## File structures

- `calpgs/`: source code
- `notebooks/`: example analysis.
- `experiments/`: code to replicate all main figures.
