#!/bin/bash

#SBATCH --gres=gpu:4
#SBATCH --cpus-per-task=4
###SBATCH --ntasks=4
#SBATCH --ntasks-per-node=4
###SBATCH --nodes=1

#SBATCH -t 3-00:00:00			#Time Limit d-hh:mm:ss
#SBATCH --partition=P4V16_HAS16M128_L		#partition/queue CAC48M192_L
#SBATCH --account=gcl_lsa273_uksr	#project allocation accout 

#SBATCH --job-name=classifier		#Name of the job
#SBATCH  --output=classifier.out		#Output file name
#SBATCH  --error=classifier.err		#Error file name

#SBATCH --mail-type ALL                 #Send email on start/end
#SBATCH --mail-user ofsk222@uky.edu     #Where to send email


#Modules needed for the job
module purge
module load ccs/singularity
echo "Job $SLURM_JOB_ID running on SLURM NODELIST: $SLURM_NODELIST "

CONTAINER="$PROJECT/lsa273_uksr/containers/pytorch/pytorch-repitl.sif"
SCRIPT="$HOME/projects/nonuniform_sampling/run.sh"
srun singularity run --nv $CONTAINER $SCRIPT
