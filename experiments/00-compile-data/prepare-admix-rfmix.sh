#!/bin/bash -l
#$ -cwd
#$ -l rh7,h_data=32G,h_rt=10:00:00
#$ -j y
#$ -o ./job_out
#$ -t 1-22


i_chr=${SGE_TASK_ID}

ROOT_DIR=../data/admix_lanc/
GENO_DIR=../data/PLINK/admix/topmed/chr${i_chr}

mkdir -p ${ROOT_DIR}/lanc
RFMIX=~/project-pasaniuc/software/rfmix/rfmix
${RFMIX} \
    -f ${GENO_DIR}/chr${i_chr}.sample.typed.nochr.vcf.gz \
    -r ${GENO_DIR}/chr${i_chr}.ref.typed.nochr.vcf.gz \
    --chromosome=${i_chr} \
    -m ${ROOT_DIR}/metadata/sample_map.tsv \
    -g ${ROOT_DIR}/metadata/genetic_map/chr${i_chr}.tsv \
    -e 1 -n 5 \
    -o ${ROOT_DIR}/lanc/chr${i_chr}
