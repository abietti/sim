#!/bin/bash
#SBATCH --job-name=sim
#SBATCH --open-mode=append
#SBATCH --output=./logs/%x_%A_%a.out
#SBATCH --error=./logs/%x_%A_%a.err
#SBATCH --export=ALL
#SBATCH --time=3:00:00
#SBATCH --mem=3G

#SBATCH --array=1-300

singularity exec --overlay /scratch/ab5637/fl_privacy/overlay-50G-10M.ext3:ro /scratch/work/public/singularity/cuda10.1-cudnn7-devel-ubuntu18.04-20201207.sif /bin/bash -c "
source /ext3/miniconda3/etc/profile.d/conda.sh
conda activate flp_nn
# python ./main_sim.py $SLURM_ARRAY_TASK_ID $SLURM_ARRAY_TASK_COUNT --task_offset 0 --name shifted_d10_it1000_s1 --s 1
python ./main_sim.py $SLURM_ARRAY_TASK_ID $SLURM_ARRAY_TASK_COUNT --task_offset 0 --name d20_it1000_s3 --d 20 --s 3

"

