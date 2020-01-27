#!/bin/bash
#SBATCH -N 1
#SBATCH --job-name=UNet-ResWDCGAN
#SBATCH -n 1
#SBATCH -c 1
#SBATCH --mem=150000
#SBATCH -o out.txt
#SBATCH -e error.txt
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1

module load cuda/10.0.130
module load gnu7
module load openmpi3
module load anaconda/3.6
source activate /opt/ohpc/pub/apps/tensorflow_1.13

#pip install --user pandas

srun -n 1 python trainResWDCGAN.py
