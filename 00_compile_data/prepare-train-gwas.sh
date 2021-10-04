#!/bin/bash -l
#$ -cwd
#$ -l rh7,h_data=16G,h_rt=2:00:00
#$ -j y
#$ -o ./job_out

trait=$1
chr_i=${SGE_TASK_ID}

DATA_DIR=../data/
out_dir=${DATA_DIR}/train_gwas/${trait}
pheno_file=${DATA_DIR}/pheno/eur_train.${trait}.pheno
covar_file=${DATA_DIR}/pheno/eur_train.covar

n_cols=$(head -1 $covar_file | awk '{print NF}')
n_cols=$((n_cols - 2))

mkdir -p ${out_dir}
plink2 --allow-no-sex --bfile ${DATA_DIR}/PLINK/eur_train/merged \
    --linear omit-ref hide-covar --ci 0.95 \
    --pheno ${pheno_file} \
    --covar ${covar_file} \
    --quantile-normalize \
    --out ${out_dir}/assoc
