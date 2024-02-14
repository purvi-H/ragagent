#!/bin/sh
# Grid Engine options (lines prefixed with #$)
#$ -N llama2_7b
#$ -cwd
#$ -l h_rt=24:00:00
#$ -l h_vmem=100G
#$ -o log.log
#$ -e log.err
#$ -q gpu
#$ -pe gpu-a100 1
#  These options are:
#  job name: -N
#  use the current working directory: -cwd
#  runtime limit: -l h_rt
#  memory limit: -l h_vmem

# Initialise the environment modules and load CUDA version 11.0.2
. /etc/profile.d/modules.sh

export XDG_CACHE_HOME="/exports/eddie/scratch/s2024596/.cache"

#Load Python
module load cuda
module load python/3.11.4

source /exports/eddie/scratch/s2024596/env/bin/activate

# pip install transformers
# pip install torch
# pip install huggingface_hub
# pip install accelerate
# pip install xformers
# pip install -U langchain-community
# pip install chromadb
# pip install sentence-transformers

# Run the program
# python3 /exports/eddie/scratch/s2024596/model.py

python3 /exports/eddie/scratch/s2024596/rag.py

deactivate
