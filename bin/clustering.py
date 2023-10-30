#!/usr/bin/env python3

"""
Script to extract the quality scores the models made by AlphaFold
"""

import argparse
from pathlib import Path
import shutil
import subprocess
import re
from pymol import cmd

import pandas as pd

import logging
logging.getLogger().setLevel(logging.INFO)

def parsing(args: list=None) -> argparse.Namespace:
    """
    Creates the argument parser instance and applies it to the command line
    input

    Args:
        args (list, optional): List of the arguments to be parsed (only to be
            used for testing). If none is provided, it is taken from sys.argv.
            Defaults to None.

    Returns:
        argparse.Namespace
    """
    
    def validate_dir(d:str) -> Path:
        """
        Validate that the directory with the features exists
        """
        d = Path(d)
        if not d.exists():
            raise ValueError("The specified directory doesn't exist.")
            
        return d
    
    def validate_out(d:str) -> Path:
        """
        Create the directory if it doesn't exist.
        """
        d = Path(d)
        if not d.exists():
            d.mkdir()
            
        return d
    
    # def validate_input(d:str) -> Path:
    #     """
    #     Validate that the input file exists
    #     """
    #     d = Path(d)
    #     if not d.exists():
    #         raise ValueError("The specified file doesn't exist.")
            
    #     return d

    parser = argparse.ArgumentParser(description=('Copy the top structures and '
        'plots to other directories.'))
    
    parser.add_argument("--kind",
        help=("Kind of clustering to perform. (Default=sequence)"), default="sequence",
        choices=["sequence", "structure"])
    
    parser.add_argument("--models_dir", 
        help=('Path with the resulting AlphaFold models. Required if --kind=sequence.'),
        type=validate_dir)
    
    parser.add_argument("--candidates", 
        help=('If --kind=sequence, this should be a FASTA file with the candidates'
              ' to be clustered.'),
        type=Path)
    
    parser.add_argument("--destination",
        help=("Path to save the results from clustering, as well as the "
              "structures of the top clusters."), required=True,
        type=validate_out)
    
    parser.add_argument("--min_seq_id",
        help=('Minimum sequence identity for clustering. (Default=0.3)'), default=0.3, 
        type=float)
    
    parser.add_argument("--coverage", 
        help=('Minimum coverage. (Default=0.8)'), default=0.8, type=float)
    
    parser.add_argument("--cov_mode",
        help=('Coverage mode. (Default=2)'), default=2, type=int)
    
    parser.add_argument("--sensitivity",
        help=('Sensitivity. (Default=7)'), default=7, type=int)
    
    parser.add_argument("--evalue",
        help=('E-value for structural clustering. (Default=0.01)'), default=0.01,
        type=float)
    
    return parser.parse_args(args)


def run_sequence_clustering(candidates_file:Path, results_prefix:str="clusterRes",
                   temp_dir:str="tmp", min_seq_id:float=0.3, coverage:float=0.8,
                   cov_mode:int=2, sensitivity:int=7, destination:Path=Path(".")):
    """
    Run mmseqs easy-cluster to cluster the candidates by sequence identity

    Args:
        candidates_file (Path): Path to the FASTA file with the candidates
        results_prefix (str, optional): Prefix for the result files. Defaults to "clusterRes".
        temp_dir (str, optional): Defaults to "tmp".
        min_seq_id (float, optional): Defaults to 0.3.
        coverage (float, optional): Defaults to 0.8.
        cov_mode (int, optional): Defaults to 2.
        sensitivity (int, optional): Defaults to 7.
        destination (Path, optional): Path to run the process from. Defaults to Path(".").
    """
    
    command = (f"mmseqs easy-cluster {candidates_file} {results_prefix} {temp_dir} "
               f"--min-seq-id {min_seq_id} -c {coverage} --cov-mode {cov_mode} "
               f"--sensitivity {sensitivity}").split()
    
    p = subprocess.run(command, cwd=destination)


def run_structure_clustering(models_dir:Path, results_prefix:str="clusterRes",
                    temp_dir:str="tmp", coverage:float=0.8, cov_mode:int=2,
                    evalue:float=0.01, destination:Path=Path(".")):
    """
    Run foldseek easy-cluster to cluster the models by structure

    Args:
        models_dir (Path): Path to the directory with the models to cluster
        results_prefix (str, optional): Defaults to "clusterRes".
        temp_dir (str, optional): Defaults to "tmp".
        coverage (float, optional): Defaults to 0.8.
        cov_mode (int, optional): Defaults to 2.
        evalue (float, optional): Defaults to 0.01.
        destination (Path, optional): Path to run the process from. Defaults to Path(".").
    """
    
    command = (f"foldseek easy-cluster {models_dir} {results_prefix} {temp_dir} "
               f"-c {coverage} --cov-mode {cov_mode} -e {evalue}").split()
    
    p = subprocess.run(command, cwd=destination)


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
                uid = re.search(r'-1_(\w+)-1.pdb', d.name).group(1)
                new_name = pdbs_dir / (uid + ".pdb")
                shutil.copy(top_pdb, new_name)
                
                # Copy plots for top pdbs to new destination
                top_plot = list((d / "plots").glob("rank_0*.png"))[0]
                new_name = plots_dir / (uid + ".png")
                shutil.copy(top_plot, new_name)
    
    logging.info(f"Processed {count} models.")
    
    return pdbs_dir


def copy_pdbs(clusters:pd.DataFrame, models_dir:Path, destination:Path) -> None:
    """
    Copy the top pdbs from the top clusters to a new directory
    
    Args:
        clusters (pd.DataFrame): DataFrame with the cluster representatives and
            members
        models_dir (Path): Path to the directory with the models
        destination (Path): Path to the directory where the pdbs will be copied
    """
    
    top10 = clusters.rep.value_counts().sort_values(ascending=False).head(10)
    
    for i, cluster in enumerate(top10.index):
        # Get cluster members
        members = clusters[clusters.rep == cluster].member.values
        cluster_dir = destination / f"cluster{i}_{cluster}"
        cluster_dir.mkdir()
        for member in members:
            # Traverse the models directory to find the member
            for d in models_dir.iterdir():
                if d.is_dir():
                    if member in d.name:
                        # Copy top pdb to new destination
                        new_name = cluster_dir / (member + ".pdb")
                        shutil.copy(d / "ranked_0.pdb", new_name)
                        break


def make_pymol_sessions(clusters:pd.DataFrame, destination:Path):
    
    top10 = clusters.rep.value_counts().sort_values(ascending=False).head(10)
    
    for i, cluster in enumerate(top10.index):
        # Get cluster members
        members = clusters[clusters.rep == cluster].member.values
        cluster_dir = destination / f"cluster{i}_{cluster}"
        
        # Load the cluster representative first
        if re.search(r'.pdb_[AB]$', cluster):
            chain = cluster[-1]
            cname = cluster.split('.pdb')[0]
            fname = cluster_dir / (cname + ".pdb")
            oname = f"{cname}_rep{chain}"
            cmd.load(fname, oname)
            cmd.do(f"select chain {chain} AND model {oname}")
        else:
            fname = cluster_dir / (cluster + ".pdb")
            cmd.load(fname, f"{cluster}_rep")
            cmd.do(f"select chain B AND model {cluster}_rep")
        
        # Load and align the members one by one
        for member in members:
            if not member in cluster:
                fname = cluster_dir / (member + ".pdb")
                cmd.load(fname)
                cmd.align(member, "sele")
        
        cmd.do('bg white')
        cmd.do('set ray_shadow, 0')
        cmd.do('color grey80')
        cmd.do('select chain A')
        cmd.do('color slate, sele')
        
        cmd.save(cluster_dir / "session.pse")
        cmd.do('delete all')


if __name__ == '__main__':

    args = parsing()
    
    if args.kind=='sequence':
    
        logging.info("Running clustering...")
        run_sequence_clustering(args.candidates, "clusterRes", "tmp", args.min_seq_id,
                    args.coverage, args.cov_mode, args.sensitivity, args.destination)
        
        logging.info("Processing output...")
        clusters_tsv = args.destination / "clusterRes_cluster.tsv"
        clusters = pd.read_table(clusters_tsv, header=None, names=["rep", "member"])
    
    else:
        
        logging.info("Copying models to new directory...")
        pdbs_dir = get_top_pdbs(args.models_dir, args.destination)
        
        logging.info("Running strcutural clustering...")
        run_structure_clustering(pdbs_dir, "clusterRes", "tmp", args.coverage,
                                 args.cov_mode, args.evalue, args.destination)
        
        logging.info("Processing output...")
        clusters_tsv = args.destination / "clusterRes_cluster.tsv"
        clusters = pd.read_table(clusters_tsv, header=None, names=["rep", "member"])
        # Get only the clusters for the structures of the binder
        clusters = clusters[clusters.member.str.endswith('_B')]
        # See if all the representatives also come from chain B
        if not all(clusters.rep.str.endswith('_B')):
            logging.info("NOT ALL REPRESENTATIVES COME FROM CHAIN B")
        # Remove the '.pdb_B' suffix from the members' names
        clusters['member'] = clusters['member'].str.split('.pdb').str[0]
    
    logging.info("Copying pdbs from the top clusters...")
    copy_pdbs(clusters, args.models_dir, args.destination)
    
    logging.info("Making Pymol sessions...")
    make_pymol_sessions(clusters, args.destination)
    
    logging.info("Done!!")
