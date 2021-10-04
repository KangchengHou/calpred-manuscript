#!/bin/sh
#$ -cwd
#$ -j y
#$ -l rh7,h_data=10G,h_rt=10:00:00 -pe shared 6
#$ -o ./job_out
#$ -t 1-22

export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1

Rscript=/u/project/pasaniuc/kangchen/software/miniconda3/envs/r/bin/Rscript

trait=$1
chr_i=$SGE_TASK_ID

DATA_DIR=../data/
mkdir -p ${DATA_DIR}/prs/weights/cache
mkdir -p ${DATA_DIR}/prs/weights/${trait}

${Rscript} ~/project-pasaniuc/software/prs-uncertainty/prs_uncertainty.R \
    --chr_i=$chr_i \
    --train_bfile=${DATA_DIR}/PLINK/eur_train/merged \
    --train_sumstats=${DATA_DIR}/train_gwas/${trait}/assoc.all.assoc.linear \
    --val_bfile=${DATA_DIR}/PLINK/eur_val/merged \
    --val_pheno=${DATA_DIR}/pheno/eur_val.${trait}.residual_pheno \
    --test_bfile=${DATA_DIR}/PLINK/eur_test/merged \
    --out_dir=${DATA_DIR}/prs/weights/${trait} \
    --cache_dir=${DATA_DIR}/prs/weights/cache \
    --n_cores=6 \
    --num_burn_in=100 \
    --num_iter=500
