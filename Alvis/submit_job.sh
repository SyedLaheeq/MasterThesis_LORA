#!/bin/bash
#SBATCH -A NAISS2026-4-376      # Updated to your active 2026 project
#SBATCH -p alvis                 # Partition
#SBATCH --nodes=1
#SBATCH --gpus-per-node=V100:1   # Request 1 A100 (40GB)
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8        # Gives you ~54GB RAM
#SBATCH -t 01:00:00              # 8 hours
#SBATCH -J RoLoRA_Llama3
#SBATCH -o logs/%j.log
#SBATCH -e logs/%j.log

# 1. Load Modules
module purge
module load PyTorch/2.1.2-foss-2023a-CUDA-12.1.1

# 2. Activate your Virtual Environment
source /mimer/NOBACKUP/groups/naiss2025-22-260/laheeq/venv_alvis/bin/activate

# 3. Set Environment Variables
export HF_HOME=/mimer/NOBACKUP/groups/naiss2025-22-260/laheeq/.cache
export WANDB_API_KEY="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"
export HF_TOKEN="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"
export HUGGING_FACE_HUB_TOKEN=$HF_TOKEN

# Ensure logs directory exists
mkdir -p logs


export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
# 4. Run the Training Agent
python run_agents.py --config config/rolora_llama3.yaml