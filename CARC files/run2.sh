#!/bin/bash

#SBATCH --account=jessetho_1016
#SBATCH --mem=16GB
#SBATCH --partition=gpu
#SBATCH --gres=gpu:2

module purge
module load gcc/8.3.0
module load python/3.9.2
module load gcc/11.3.0
module load py-pip/21.3.1
module load openblas/0.3.20
module load conda/4.12.0
module load cuda/11.6.2
module load py-numpy/1.22.4
module load py-pillow/9.0.0


pip install torch torchvision torchaudio
pip install pandas

python3 moco.py
