#!/bin/bash
#SBATCH --time=96:00:00
#SBATCH --mem=60gb
#SBATCH -c 5

source /vol/ek/Home/orlyl02/venv/oligopred2/bin/activate

# Run the script
python3 /vol/ek/Home/orlyl02/working_dir/oligopred/embed_transfer/check_clustering.py > check_clust_c0.3.log

deactivate
