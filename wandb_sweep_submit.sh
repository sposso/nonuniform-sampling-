#!/bin/bash

#NUMBER OF AGENTS TO REGISTER AS WANDB AGENTS
#SHOULD BE -array=1-X, where X is number of estimated runs
#SBATCH --array=1-4    #e.g. 1-4 will create agents labeled 1,2,3,4

#Below is configuration PER AGENT

#SBATCH --gres=gpu:2
#SBATCH --cpus-per-task=8
#SBATCH --ntasks=2
#SBATCH --ntasks-per-node=2
#SBATCH --nodes=1

#SBATCH -t 0-06:00:00			#Time Limit d-hh:mm:ss
#SBATCH --partition=V4V32_SKY32M192_L		#partition/queue CAC48M192_L
#SBATCH --account=gcl_lsa273_uksr	#project allocation accout 

#SBATCH --job-name=sweep		#Name of the job
#SBATCH  --output=sweep.out		#Output file name
#SBATCH  --error=sweep.err		#Error file name

#SBATCH --mail-type ALL                 #Send email on start/end
#SBATCH --mail-user ofsk222@uky.edu     #Where to send email

#Modules needed for the job
module purge
module load ccs/singularity
echo "Job SWEEP $SLURM_JOB_ID running on SLURM NODELIST: $SLURM_NODELIST"

CONTAINER="$PROJECT/lsa273_uksr/containers/python310/python310.sif"

# SET SWEEP_ID HERE. Note sweep must already be created on wandb before submitting job
SWEEP_ID="**************************************"
API_KEY="******************************************"

# LOGIN IN ALL TASKS
srun singularity run --nv $CONTAINER wandb login $api_key

# RUN WANDB AGENT IN ONE TASK
{
    IFS=$'\n' read -r -d '' SWEEP_DETAILS; RUN_ID=$(echo $SWEEP_DETAILS | sed -e "s/.*\[\([^]]*\)\].*/\1/g" -e "s/[\'\']//g")
    IFS=$'\n' read -r -d '' SWEEP_COMMAND;
} < <((printf '\0%s\0' "$(srun --ntasks=1 singularity run --nv $CONTAINER wandb agent --count 1 $SWEEP_ID)" 1>&2) 2>&1)


SWEEP_COMMAND="${SWEEP_COMMAND} --wandb_resume_version ${RUN_ID}"

# WAIT FOR ALL TASKS TO CATCH UP
wait

# RUN SWEEP COMMAND IN ALL TASKS
srun singularity run --nv $CONTAINER $SWEEP_COMMAND