#!/bin/sh
#$ -cwd
#$ -j y
#$ -l h_data=10G,h_rt=12:00:00 -pe shared 6
#$ -o ./job_out

export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1

Rscript=/u/project/pasaniuc/kangchen/software/miniconda3/envs/r/bin/Rscript

trait=$1

DATA_DIR=../../data/
mkdir -p ${DATA_DIR}/prs/weights/

echo "val_pheno"
cat ${DATA_DIR}/pheno/eur_val.${trait}.residual_pheno | head

${Rscript} ~/project-pasaniuc/software/prs-uncertainty/weight.R \
    --train_bfile=${DATA_DIR}/PLINK/eur_train/merged \
    --train_sumstats=${DATA_DIR}/train_gwas/${trait}/assoc.ldpred2.tsv \
    --val_bfile=${DATA_DIR}/PLINK/eur_val/merged \
    --val_pheno=${DATA_DIR}/pheno/eur_val.${trait}.residual_pheno \
    --test_bfile=${DATA_DIR}/PLINK/eur_test/merged \
    --out_prefix=${DATA_DIR}/prs/weights/${trait} \
    --ld_dir=${DATA_DIR}/LD/eur_train \
    --n_core=6 \
    --n_burn_in=100 \
    --n_iter=500
