#!/bin/bash
#SBATCH --job-name=mimeta_data
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --nodes=1
#SBATCH --time=0-05:00
#SBATCH --partition=cpu-short
#SBATCH --gres=gpu:0
#SBATCH --mem=200G
#SBATCH --output=mimetadata_%j.out
#SBATCH --error=mimetadata_%j.err
#SBATCH --mail-type=ALL
#SBATCH --mail-user=arthur.jaques@student.uni-tuebingen.de

source $HOME/.bashrc

conda activate meta_learning

scontrol show job $SLURM_JOB_ID

declare -ar non_script_py_files=("utils.py")
for f in mimeta_pipelines/datasets/*.py
do
    if ! [[ " ${non_script_py_files[*]} " =~ " $f " ]]
    then
        python -m "mimeta_pipelines.datasets.$(basename "$f" .py)";
    fi
done

conda deactivate
