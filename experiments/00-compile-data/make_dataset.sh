#!/bin/bash -l
#$ -cwd
#$ -l rh7,h_data=12G,h_rt=0:50:00
#$ -j y
#$ -o ./job_out
#$ -t 1-22

. /u/local/Modules/default/init/modules.sh
export PATH=~/project-pasaniuc/software/miniconda3/bin:$PATH
export PYTHONNOUSERSITE=True

ROOT_DIR=/u/project/pasaniuc/pasaniucdata/admixture/projects/admix-prs-uncertainty/data/
chrom=${SGE_TASK_ID}

vcf=${ROOT_DIR}/PLINK/admix/topmed/chr${chrom}/chr${chrom}.sample.typed.nochr.vcf.gz
lanc=${ROOT_DIR}/admix-analysis/lanc/chr${chrom}.msp.tsv
out=${ROOT_DIR}/admix-analysis/dataset/chr${chrom}.zarr

admix vcf_to_dataset --vcf $vcf --lanc $lanc --out $out
