# calprs (Calibrated PRS)

To install the package,
```bash
pip install -e .
```

## CLI for using the package
```bash
toy=tests/test-data/toy.tsv
calprs r2diff \
    --df ${toy} \
    --y y_cov \
    --pred prs \
    --group age \
    --out out
```

```bash
calprs model \
    --df <df_path> \
    --y <response variable> \
    --pred
    --out <model_path>
```

We take input of PRS point estimate (mean), as well as the \alpha-level credible interval (lower-ci, upper-ci), in addition to a set of individuals used to perform the calibration. We aim to produce a tight and calibrated prediction interval.

The pipeline goes as follows:
1. Obtaining the PRS prediction (mean + std) of genetic risk for every individual in the data set (since PRS is usually consisting of hundreds of SNPs, the error terms are likely well-approximated by a Gaussian, which can be described with only mean + std)
2. Since step 1 only contains only genetic risk (the std only capture the genetic risk uncertainty). We perform two steps:
    (a)
## File structures

- `admix_prs/`: source code
- `notebooks/`: example analysis.
- `experiments/`: code to replicate all main figures.
