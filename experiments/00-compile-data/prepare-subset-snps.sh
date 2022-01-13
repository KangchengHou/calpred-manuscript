#!/bin/bash -l
#$ -cwd
#$ -l rh7,h_data=32G,h_rt=10:00:00
#$ -j y
#$ -o ./job_out
#$ -t 1-22

i_chr=${SGE_TASK_ID}
out_dir=/u/project/pasaniuc/pasaniucdata/admixture/projects/admix-prs-uncertainty/data/PLINK/all

mkdir -p ${out_dir}

plink \
    --bfile /u/project/sgss/UKBB/data/imp/hard_calls.ukbb-showcase/nodup/${i_chr} \
    --extract /u/project/pasaniuc/pasaniucdata/admixture/projects/PAGE-QC/s00_select_snp/match_UKB.hm3.chr${i_chr}.snp \
    --keep-allele-order \
    --make-bed \
    --out ${out_dir}/chr${i_chr}
