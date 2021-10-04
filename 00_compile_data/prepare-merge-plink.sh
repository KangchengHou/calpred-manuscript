#!/bin/bash -l
#$ -cwd
#$ -l rh7,h_data=32G,h_rt=10:00:00
#$ -j y
#$ -o ./job_out

out_dir=$1
rm -f ${out_dir}/merged.merge_list

for i in $(seq 2 22); do
    echo -e "${out_dir}/chr${i}" >>${out_dir}/merged.merge_list
done

plink \
    --bfile ${out_dir}/chr1 \
    --keep-allele-order \
    --make-bed \
    --merge-list ${out_dir}/merged.merge_list \
    --out ${out_dir}/merged
