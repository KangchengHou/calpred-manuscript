#!/bin/bash -l
#$ -cwd
#$ -l h_data=8G,h_rt=5:00:00,highp
#$ -j y
#$ -o ./job_out
#$ -t 1-22
#$ -pe shared 6

export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1

Rscript=/u/project/pasaniuc/kangchen/software/miniconda3/envs/r/bin/Rscript

SRC_DIR=/u/project/pasaniuc/kangchen/software/prs-uncertainty
DATA_DIR=/u/project/pasaniuc/pasaniucdata/admixture/projects/admix-prs-uncertainty/data

mkdir -p ${DATA_DIR}/LD/eur_train/

CHROM=${SGE_TASK_ID}

${Rscript} ${SRC_DIR}/ld.R \
    --train_bfile ${DATA_DIR}/PLINK/eur_train/chr${CHROM} \
    --chrom ${CHROM} \
    --n_core 6 \
    --ld_dir ${DATA_DIR}/LD/eur_train/
