#!/bin/bash
#SBATCH --time=96:00:00
#SBATCH --mem=10gb
#SBATCH -c 2

source /vol/ek/Home/orlyl02/venv/oligopred2/bin/activate

# Run the script
python3 /vol/ek/Home/orlyl02/working_dir/oligopred/embed_transfer/qs_from_mat_no_clustering.py > qs_no_clust_loop.log

deactivate
