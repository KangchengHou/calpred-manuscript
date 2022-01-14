#!/bin/bash -l
#$ -cwd
#$ -l rh7,h_data=32G,h_rt=16:00:00
#$ -j y
#$ -o ./job_out
#$ -t 1-22

. /u/local/Modules/default/init/modules.sh
module load bcftools
module load htslib

i_chr=${SGE_TASK_ID}
password=m8Hw1DYuDtUzWC

ROOT_DIR=../data/PLINK/admix/topmed/
dir_work=${ROOT_DIR}/chr${i_chr}
mkdir -p ${dir_work}
unzip -P ${password} ${ROOT_DIR}/chr_${i_chr}.zip -d ${dir_work}

cd ${dir_work}

mkdir tmp

vcf_imputed=chr${i_chr}.sample.imputed.vcf.gz

# QC imputed
bcftools filter -i 'INFO/R2>0.8 & INFO/MAF > 0.005' chr${i_chr}.dose.vcf.gz -Oz -o ${vcf_imputed}
tabix -p vcf ${vcf_imputed}

# extract typed data & make index
my_vcf=tmp/typed.vcf.gz
bcftools view -i 'TYPED=1|TYPED_ONLY=1' ${vcf_imputed} -Oz -o ${my_vcf}
tabix -p vcf ${my_vcf}

# match 1kg
kg_vcf=/u/project/pasaniuc/pasaniucdata/DATA/1000_Genomes_30x_GRCh38_phased/CCDG_14151_B01_GRM_WGS_2020-08-05_chr${i_chr}.filtered.shapeit2-duohmm-phased.vcf.gz
bcftools isec -n =2 ${my_vcf} ${kg_vcf} -p ./tmp -c none

cat tmp/0000.vcf | bgzip -c > chr${i_chr}.sample.typed.vcf.gz
cat tmp/0001.vcf | bgzip -c > chr${i_chr}.ref.typed.vcf.gz

# remove chr
echo "chr${i_chr} ${i_chr}" > chr_name.txt
bcftools annotate --rename-chrs chr_name.txt chr${i_chr}.sample.typed.vcf.gz | bgzip > chr${i_chr}.sample.typed.nochr.vcf.gz
bcftools annotate --rename-chrs chr_name.txt chr${i_chr}.ref.typed.vcf.gz | bgzip > chr${i_chr}.ref.typed.nochr.vcf.gz

# clean up
rm -rf tmp
