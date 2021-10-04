#!/bin/sh
#$ -cwd
#$ -j y
#$ -l h_data=10G,h_rt=1:00:00,highp
#$ -o ./job_out

Rscript=/u/project/pasaniuc/kangchen/software/miniconda3/envs/r/bin/Rscript

trait=$1
DATA_DIR=../data/

${Rscript} ~/project-pasaniuc/software/prs-uncertainty/summary.R \
    --out_dir=${DATA_DIR}/prs/weights/${trait}
