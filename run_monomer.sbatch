#!/bin/bash
#SBATCH -N 1
#SBATCH --partition=batch
#SBATCH -J AlphaFold2
#SBATCH -o out_ibex/%x.%j.out
#SBATCH --time=02:00:00
#SBATCH --mem=64G
#SBATCH --gres=gpu:4
#SBATCH --cpus-per-task=16
#SBATCH --constraint=[v100]

module load alphafold cuda/11.2.2
export ALPHAFOLD_DATA=/ibex/reference/KSL/alphafold/2.1.1/all_alphafold_data
export CUDA_VISIBLE_DEVICES=0,1,2,3
export TF_FORCE_UNIFIED_MEMORY=1
export XLA_PYTHON_CLIENT_MEM_FRACTION=0.5
export XLA_PYTHON_CLIENT_ALLOCATOR=platform

fasta_file=WRITE_HERE
out_dir=output
recycles=3

time python3 ./run_alphafold.py \
 --data_dir=$ALPHAFOLD_DATA \
 --output_dir=$out_dir \
 --fasta_paths=$fasta_file \
 --db_preset=full_dbs \
 --bfd_database_path=$ALPHAFOLD_DATA/bfd/bfd_metaclust_clu_complete_id30_c90_final_seq.sorted_opt \
 --uniclust30_database_path=$ALPHAFOLD_DATA/uniclust30/uniclust30_2018_08/uniclust30_2018_08 \
 --uniref90_database_path=$ALPHAFOLD_DATA/uniref90/uniref90.fasta \
 --mgnify_database_path=$ALPHAFOLD_DATA/mgnify/mgy_clusters_2018_12.fa \
 --template_mmcif_dir=$ALPHAFOLD_DATA/pdb_mmcif/mmcif_files \
 --model_preset=monomer_ptm \
 --obsolete_pdbs_path=$ALPHAFOLD_DATA/pdb_mmcif/obsolete.dat \
 --pdb70_database_path=$ALPHAFOLD_DATA/pdb70/pdb70 \
 --max_template_date=2021-12-01 \
 --recycles=$recycles
