#!/bin/bash -l
#SBATCH --job-name=odeint
#SBATCH --output=odeint_%A_%a.out
#SBATCH --error=odeint_%A_%a.err
#SBATCH --partition=defq
#SBATCH -N 1
#SBATCH -c 64
#SBATCH --mem=180G  
#SBATCH --time=24:00:00
#SBATCH --array=1

export HF_HOME="/Users/robsonlab/scratch"
export HF_TRANSFORMERS_CACHE="${HF_HOME}"
export HF_DATASETS_CACHE="${HF_HOME}/datasets"

#codex WITH TEMPS DRIVER
#DON'T USE CONDA ENV, ITS GCC IS INCOMPATIBLE 
module load CMake 
source ~/Teetly/.legion_setup 
source /Users/robsonlab/Teetly/.venv/bin/activate  
source /Users/robsonlab/Teetly/.secrets 
cd /Users/robsonlab/scratch/optimize_hpx_to_legion/analysis

python code_only_\@k.py --results-root /Users/robsonlab/scratch/optimize_hpx_to_legion/ --target-path /Users/robsonlab/Teetly/ParEval-Repo/targets --system-config /Users/robsonlab/Teetly/ParEval-Repo/config/perlmutter-config.json --scratch-dir /tmp/scratch --enable-scaling --apps osc_chain_1d --force-overwrite