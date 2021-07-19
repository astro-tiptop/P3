#!/bin/bash
#SBATCH -J PSF_Generate.py     # run's name
#SBATCH -N 1                   # nb of nodes 
#SBATCH -c 1                   # nb of cpus per task
#SBATCH --mem=15GB             # RAM
#SBATCH -t 2:00:00             # walltime
#SBATCH -o resJobG.txt         # output file name
#SBATCH -e errJobG.txt         # error file name
#SBATCH --mail-type=BEGIN,END  
#SBATCH --mail-user=olivier.beltramo-martin@lam.fr

python3 ./deepLoopPerformanceBatch.py --txt='/home/omartin/Projects/APPLY/RESULTS/task1_fetick/NN3/NOISE_NOSTATIC/MAG12/ResTest_NN06291109_DS06291109_Sc1_all.txt' --ini='/home/omartin/Projects/APPLY/CODES/_ANN/dataGen/nirc2.ini' --mag=12 --nPSF=10 --fit=True > 'perfStatus.txt'
