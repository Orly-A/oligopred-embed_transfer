#!/bin/sh
#SBATCH --time=96:00:00
#SBATCH --mem=10gb
#SBATCH -c 4

source /vol/ek/Home/orlyl02/working_dir/python3_venv/bin/activate.csh
/vol/ek/Home/orlyl02/working_dir/python3_venv/bin/python3 /vol/ek/Home/orlyl02/working_dir/oligopred/embed_transfer/parse_seq_similairty.py > parse_seq.log