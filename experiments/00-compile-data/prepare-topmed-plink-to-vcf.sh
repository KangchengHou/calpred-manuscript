#!/bin/bash -l
#$ -cwd
#$ -l rh7,h_data=8G,h_rt=0:30:00
#$ -j y
#$ -o ./job_out
#$ -t 1-22

out_dir=../data/PLINK/admix/
chrom=${SGE_TASK_ID}

plink2 \
    --bfile ${out_dir}/chr${chrom} \
    --out ${out_dir}/chr${chrom} \
    --recode vcf

bgzip ${out_dir}/chr${chrom}.vcf
