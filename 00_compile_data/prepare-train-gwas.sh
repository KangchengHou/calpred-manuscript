#!/bin/bash -l
#$ -cwd
#$ -l h_data=32G,h_rt=16:00:00
#$ -j y
#$ -o ./job_out

trait=$1

DATA_DIR=../data/
out_dir=${DATA_DIR}/train_gwas/${trait}
pheno_file=${DATA_DIR}/pheno/eur_train.${trait}.pheno
covar_file=${DATA_DIR}/pheno/eur_train.covar

mkdir -p ${out_dir}
plink2 --bfile ${DATA_DIR}/PLINK/eur_train/merged \
    --linear omit-ref hide-covar --ci 0.95 \
    --pheno ${pheno_file} \
    --covar ${covar_file} \
    --quantile-normalize \
    --out ${out_dir}/assoc
