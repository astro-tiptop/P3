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

python3 ./dataGeneratorBatch.py --ini='/home/omartin/Projects/APPLY/CODES/_ANN/dataGen/nirc2.ini' --mag=10 --addStat=0 --nPSFperFolder=10 --ntest=0.1 --savePath='/home/omartin/Projects/APPLY/' > '/home/omartin/Projects/APPLY/genStatus.txt'
