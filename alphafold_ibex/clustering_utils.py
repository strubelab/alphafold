"""
Clustering functions
"""

import argparse
from pathlib import Path
import shutil
import subprocess
import re
import os
import multiprocessing
from itertools import combinations
from typing import Dict, Set, Union, Tuple
from subprocess import CalledProcessError

import pandas as pd
import numpy as np
import biskit as b

import logging
logging.getLogger().setLevel(logging.INFO)


def run_sequence_clustering(destination:Path, candidates_file:Path,
                        results_prefix:str="clusterRes", temp_dir:str="tmp",
                        min_seq_id:float=0.3, coverage:float=0.8, cov_mode:int=2,
                        sensitivity:int=7):
    """
    Run mmseqs easy-cluster to cluster the candidates by sequence identity

    Args:
        destination (Path): Parent destination path
        candidates_file (Path): Path to the FASTA file with the candidates
        results_prefix (str, optional): Prefix for the result files. Defaults to "clusterRes".
        temp_dir (str, optional): Defaults to "tmp".
        min_seq_id (float, optional): Defaults to 0.3.
        coverage (float, optional): Defaults to 0.8.
        cov_mode (int, optional): Defaults to 2.
        sensitivity (int, optional): Defaults to 7.
    """
    
    out_seqcluster = destination / "seqclusters"
    out_seqcluster.mkdir(exist_ok=True)

    seqclusters_tsv = out_seqcluster / "clusterRes_cluster.tsv"

    if not seqclusters_tsv.exists():

        logging.info("Running sequence clustering...")
        command = (f"mmseqs easy-cluster {candidates_file} {results_prefix} {temp_dir} "
                f"--min-seq-id {min_seq_id} -c {coverage} --cov-mode {cov_mode} "
                f"-s {sensitivity}").split()
        try:
            p = subprocess.run(command, cwd=out_seqcluster, capture_output=True)
            p.check_returncode()
        except CalledProcessError as e:
            fail(p, "mmseqs", command, e)
        
    else:
        logging.info("Sequence clustering output already exists.")
        
    logging.info("Processing output...")
    seqclusters = pd.read_table(seqclusters_tsv, header=None, names=["rep", "member"])

    return seqclusters


def get_top_pdbs(models_dir:Path, destination:Path):
    """
    Copy the top ranked models for each complex, along with their 2D structure
    plots, to a new directory.

    Args:
        models_dir (Path): Path to the directory with the models
        destination (Path): Path to the directory where the pdbs will be copied
    """
    
    pdbs_dir = destination / "all_pdbs"
    plots_dir = destination / "all_plots"
    pdbs_dir.mkdir()
    plots_dir.mkdir()
    
    count=0
    for d in models_dir.iterdir():
        if d.is_dir():
            top_pdb = d / "ranked_0.pdb"
            if top_pdb.exists():
                count += 1
                # Copy top pdb to new destination
                uid = re.search(r'-1_(\w+)-1', d.name).group(1)
                new_name = pdbs_dir / (uid + ".pdb")
                shutil.copy(top_pdb, new_name)
                
                # Copy plots for top pdbs to new destination
                top_plot = list((d / "plots").glob("rank_0*.png"))[0]
                new_name = plots_dir / (uid + ".png")
                shutil.copy(top_plot, new_name)
    
    logging.info(f"Processed {count} models.")
    
    return pdbs_dir


def run_structure_clustering(destination:Path, top_models_dir:Union[Path, None],
                    models_dir:Path=None, results_prefix:str="clusterRes",
                    temp_dir:str="tmp", coverage:float=0.8, cov_mode:int=2,
                    evalue:float=0.01) -> Tuple[pd.DataFrame, Path]:
    """
    Run foldseek easy-cluster to cluster the models by structure

    Args:
        destination (Path): Parent destination path
        top_models_dir (Path): Path to the directory with the PDBs to cluster
        models_dir (Path): Path to the directory with the models from AlphaFold
        results_prefix (str, optional): Defaults to "clusterRes".
        temp_dir (str, optional): Defaults to "tmp".
        coverage (float, optional): Defaults to 0.8.
        cov_mode (int, optional): Defaults to 2.
        evalue (float, optional): Defaults to 0.01.
    """
    
    out_strcluster = destination / "strclusters"
    out_strcluster.mkdir(exist_ok=True)

    if top_models_dir is None:
        logging.info("Copying models to new directory...")
        pdbs_dir = get_top_pdbs(models_dir, out_strcluster)
    else:
        pdbs_dir = top_models_dir
    
    strclusters_tsv = out_strcluster / "clusterRes_cluster.tsv"

    if not strclusters_tsv.exists():

        logging.info("Running structural clustering...")
        command = (f"foldseek easy-cluster {pdbs_dir} {results_prefix} {temp_dir} "
                f"-c {coverage} --cov-mode {cov_mode} -e {evalue}").split()
        
        try:
            p = subprocess.run(command, cwd=out_strcluster, capture_output=True)
            p.check_returncode()
        except CalledProcessError as e:
            fail(p, "foldseek", command, e)
    
    else:
        logging.info("Structure clustering output already exists.")
    
    logging.info("Processing output...")
    strclusters = pd.read_table(strclusters_tsv, header=None, names=["rep", "member"])

    return strclusters, pdbs_dir


def fail(process:subprocess.CompletedProcess, program:str, args:list,
         error:CalledProcessError):
    """
    Generates the error message and raises the corresponding error if the
    program fails.
    """
    error_string = \
        f"\n{program} EXECUTION FAILED.\n"+\
        f"Command: {' '.join(args)}\n"

    error_string += \
        f"Returncode: {process.returncode}\n"+\
        f"STDERR: \n"+\
        process.stderr.decode("utf-8")

    logging.error(error_string)
    
    raise error
