#!/bin/bash -l
#$ -cwd
#$ -l rh7,h_data=32G,h_rt=10:00:00
#$ -j y
#$ -o ./job_out

data_dir=../data/PLINK
prefix=$1

mkdir -p ${data_dir}/${prefix}

plink \
    --bfile ${data_dir}/all/merged \
    --keep-allele-order \
    --keep-fam ${data_dir}/${prefix}.fam \
    --make-bed \
    --out ${data_dir}/${prefix}/merged

for i_chr in $(seq 1 22); do
    plink \
        --bfile ${data_dir}/${prefix}/merged \
        --keep-allele-order \
        --chr ${i_chr} \
        --make-bed \
        --out ${data_dir}/${prefix}/chr${i_chr}
done
