#!/bin/bash
#SBATCH --job-name=job_Aorta_80_ts12         # Job name
#SBATCH --output=job_Aorta_80_ts12_%j.out    # Output file (%j = job ID)
#SBATCH --nodes=1                         # Request 1 node
#SBATCH --gpus-per-node=1                 # Request 1 GPU
#SBATCH --time=10:00:00                   # Max runtime (30 minutes)

# Load modules
module purge; module load StdEnv/2023 python gcc/12.3 cuda vtk/9.3.0 opencv symengine

# Activate Python environment (if applicable)
source /home/khanmu11/Softwares/PhysicsNemo/bin/activate

# Check GPU allocation
srun nvidia-smi

# Run your workload
srun python Aorta_80_ts12.py
