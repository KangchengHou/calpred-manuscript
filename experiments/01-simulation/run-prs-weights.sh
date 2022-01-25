#!/bin/sh
#$ -cwd
#$ -j y
#$ -l h_data=6G,h_rt=15:30:00 -pe shared 6
#$ -o ./job_out
#$ -t 1-10

export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1

Rscript=/u/project/pasaniuc/kangchen/software/miniconda3/envs/r/bin/Rscript

prefix=$1
sim_i=${SGE_TASK_ID}
sim_i=$((sim_i - 1))
BFILE_PREFIX="merged"
DATA_DIR=/u/project/pasaniuc/pasaniucdata/admixture/projects/admix-prs-uncertainty/data
PHENO_DIR=out/pheno/${prefix}
OUT_DIR=out/ldpred2/${prefix}

mkdir -p ${OUT_DIR}

train_sumstats=${PHENO_DIR}/sim_${sim_i}.assoc.ldpred2.tsv
val_pheno=${PHENO_DIR}/sim_${sim_i}.eur_val.pheno.tsv
echo "train_sumstats:"
cat ${train_sumstats} | head
echo "val_pheno:"
cat ${val_pheno} | head

${Rscript} /u/project/pasaniuc/kangchen/software/prs-uncertainty/weight.R \
    --train_bfile=${DATA_DIR}/PLINK/eur_train/${BFILE_PREFIX} \
    --train_sumstats=${PHENO_DIR}/sim_${sim_i}.assoc.ldpred2.tsv \
    --val_bfile=${DATA_DIR}/PLINK/eur_val/${BFILE_PREFIX} \
    --val_pheno=${PHENO_DIR}/sim_${sim_i}.eur_val.pheno.tsv \
    --test_bfile=${DATA_DIR}/PLINK/eur_test/${BFILE_PREFIX} \
    --out_prefix=${OUT_DIR}/sim_${sim_i} \
    --ld_dir=${DATA_DIR}/LD/eur_train/ \
    --n_core=6 \
    --n_burn_in=100 \
    --n_iter=500
