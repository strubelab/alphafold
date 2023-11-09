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
from typing import Dict, Set

import pandas as pd
import numpy as np

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
    
    parser.add_argument("--models_dir", 
        help=('Path with the resulting AlphaFold models.'), required=True,
        type=validate_dir)
    
    parser.add_argument("--candidates", 
        help=('FASTA file with the sequences to be clustered.'), required=True,
        type=Path)
    
    parser.add_argument("--get_top_models",
        help=("Whether to copy first the top models to a new directory."),
        action='store_true')
    
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
               f"-s {sensitivity}").split()
    
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
                uid = re.search(r'-1_(\w+)-1', d.name).group(1)
                new_name = pdbs_dir / (uid + ".pdb")
                shutil.copy(top_pdb, new_name)
                
                # Copy plots for top pdbs to new destination
                top_plot = list((d / "plots").glob("rank_0*.png"))[0]
                new_name = plots_dir / (uid + ".png")
                shutil.copy(top_plot, new_name)
    
    logging.info(f"Processed {count} models.")
    
    return pdbs_dir


def copy_pdbs(clusters:pd.DataFrame, pdbs_dir:Path, destination:Path) -> None:
    """
    Copy the top pdbs from the top clusters to a new directory
    
    Args:
        clusters (pd.DataFrame): DataFrame with the cluster representatives and
            members
        pdbs_dir (Path): Path to the directory with the models
        destination (Path): Path to the directory where the pdbs will be copied
    """
    
    top10 = (clusters.groupby('merged_rep').agg({'iptms': 'mean', 'binder': 'count'})
             .sort_values(by=['binder', 'iptms'], ascending=False).head(10))
    
    for i, cluster in enumerate(top10.index):
        # Get cluster members
        members = clusters[clusters.merged_rep == cluster].member.values
        cluster_dir = destination / f"cluster{i+1}_{cluster}"
        cluster_dir.mkdir()
        for member in members:
            pdb_name = pdbs_dir / (member + ".pdb")
            assert pdb_name.exists(), f"{pdb_name} doesn't exist"
            new_name = cluster_dir / (member + ".pdb")
            shutil.copy(pdb_name, new_name)


def make_pymol_sessions(clusters:pd.DataFrame, destination:Path):
    
    top10 = (clusters.groupby('merged_rep').agg({'iptms': 'mean', 'binder': 'count'})
             .sort_values(by=['binder', 'iptms'], ascending=False).head(10))
    
    for i, cluster in enumerate(top10.index):
        # Get cluster members
        members = clusters[clusters.merged_rep == cluster].member.values
        cluster_dir = destination / f"cluster{i+1}_{cluster}"

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


def get_topcluster_members(clusters:pd.DataFrame, min_count:int=2) -> Dict[str, Set[str]]:
    """
    Obtain the top clusters in dictionary format, with the cluster representative
    as the key and the members as the values in a set.

    Args:
        clusters (pd.DataFrame): _description_
        min_count (int, optional): _description_. Defaults to 2.

    Returns:
        dict: _description_
    
    E.g.
    {'Q5N7Y5': {'Q5N7Y5', 'Q69XA8', 'Q6K7U3'},
     'Q2QN41': {'Q2QN41', 'Q8W0W4'},
     'A0A0P0XNZ0': {'A0A0P0XNZ0', 'A0A0P0XQ16'}}
    """
    # Get the member count
    member_counts = (clusters.groupby('rep').agg({'member': 'count'})
                 .sort_values(by='member', ascending=False).reset_index())
    # Get the clusters with more than one member
    top_clusters = member_counts[member_counts.member >= min_count]
    # Get the members of the top clusters
    top_clusters_members = {r:set(clusters[clusters.rep==r].member.tolist()) \
                           for r in top_clusters.rep}
    
    return top_clusters_members


def merge_dict_values(unmerged_dict:Dict[str, Set[str]]) -> Dict[str, Set[str]]:
    """
    Take a dictionary with sets as values, and merge the sets that have elements
    in common.

    Args:
        d (Dict[str, Set[str]]): _description_

    Returns:
        Dict[str, Set[str]]: _description_
    """
    
    joint_dict = {}
    while unmerged_dict:
        rep, members = unmerged_dict.popitem()

        if members is None:
            continue

        merged_members = set()
        keys_to_merge = []
        for rep2, members2 in unmerged_dict.items():
            if members2 is None:
                continue
            if not members.isdisjoint(members2):    # if they have elements in common
                merged_members = merged_members | members | members2
                keys_to_merge.append(rep2)

        # Erase the values of the merged clusters
        for rep2 in keys_to_merge:
            unmerged_dict[rep2] = None

        if merged_members:
            joint_dict[rep] = merged_members
        else:
            joint_dict[rep] = members
    
    return joint_dict


def joint_cluster(seqclusters:pd.DataFrame, strclusters:pd.DataFrame) -> Dict[str, Set[str]]:
    
    top_seqclusters_members = get_topcluster_members(seqclusters)
    top_strclusters_members = get_topcluster_members(strclusters)
    
    # Iterate over the top structure clusters, and see if they have elements in
    # common with the top sequence clusters. If they do, join them.
    joint_clusters = {}
    merged_seqclusters = []
    for rep, members in top_strclusters_members.items():
        merged_members = set()
        for rep2, members2 in top_seqclusters_members.items():
            if not members.isdisjoint(members2):
                merged_members = merged_members | members | members2
                merged_seqclusters.append(rep2)
        
        if merged_members:
            joint_clusters[rep] = merged_members

    # Merge joint clusters further in cases where they have elements in common
    joint_clusters2 = merge_dict_values(joint_clusters)

    # Merge the rest of the structure clusters
    joint_clusters2 = top_strclusters_members | joint_clusters2

    # Merge the missing sequence clusters
    missing_seqclusters = set(top_seqclusters_members.keys()) - set(merged_seqclusters)
    if missing_seqclusters:
        for rep in missing_seqclusters:
            joint_clusters2[rep] = top_seqclusters_members[rep]
    
    return joint_clusters2
    


if __name__ == '__main__':

    args = parsing()
    
    # Sequence clustering
    out_seqcluster = args.destination / "seqclusters"
    out_seqcluster.mkdir()
    logging.info("Running clustering...")
    run_sequence_clustering(args.candidates, "clusterRes", "tmp", args.min_seq_id,
                args.coverage, args.cov_mode, args.sensitivity, out_seqcluster)

    logging.info("Processing output...")
    seqclusters_tsv = out_seqcluster / "clusterRes_cluster.tsv"
    seqclusters = pd.read_table(seqclusters_tsv, header=None, names=["rep", "member"])
    
    # Structure clustering
    out_strcluster = args.destination / "strclusters"
    out_strcluster.mkdir()

    if args.get_top_models:
        logging.info("Copying models to new directory...")
        pdbs_dir = get_top_pdbs(args.models_dir, out_strcluster)
    else:
        pdbs_dir = args.models_dir

    logging.info("Running strcutural clustering...")
    run_structure_clustering(pdbs_dir, "clusterRes", "tmp", args.coverage,
                                args.cov_mode, args.evalue, out_strcluster)
    
    logging.info("Processing output...")
    strclusters_tsv = out_strcluster / "clusterRes_cluster.tsv"
    strclusters = pd.read_table(strclusters_tsv, header=None, names=["rep", "member"])
    # Get only the clusters for the structures of the binder
    strclusters = strclusters[strclusters.member.str.endswith('_B')]
    # See if all the representatives also come from chain B
    if not all(strclusters.rep.str.endswith('_B')):
        logging.info("NOT ALL REPRESENTATIVES COME FROM CHAIN B")
    # Remove the '.pdb_B' suffix from the members' names
    strclusters['member'] = strclusters['member'].str.split('.pdb').str[0]

    # Add quality scores to the clusters
    logging.info("Adding quality scores to the clusters...")
    pdockq = pd.read_csv(args.models_dir / "scores/scores_pdockq.csv")
    pdockq['binder'] = pdockq.complex.str.split('_').str[1].str[:-2]

    strclusters = pdockq.merge(strclusters, how='left', left_on='binder',
                                right_on='member')

    seqclusters = pdockq.merge(seqclusters, how='left', left_on='binder',
                                right_on='member')

    # Joint clustering
    joint_clusters = joint_cluster(seqclusters, strclusters)
    
    # Modify old columns
    strclusters.rename(columns={'rep':'str_rep'}, inplace=True)
    seqclusters.rename(columns={'rep':'seq_rep'}, inplace=True)

    # Make the `complex` column the index
    strclusters.set_index('complex', inplace=True)
    seqclusters.set_index('complex', inplace=True)

    # Add the `seq_rep` column to the `strclusters` dataframe
    strclusters['seq_rep'] = seqclusters.seq_rep
    
    # Initialize a new column `merged_rep`
    strclusters['merged_rep'] = None
    
    # Iterate over joint_clusters and change the values of the `merged_rep` column
    # in the `strclusters` dataframe
    for rep, members in joint_clusters.items():
        strclusters.loc[strclusters.member.isin(members), 'merged_rep'] = rep

    strclusters.fillna(np.nan, inplace=True)
    
    # Copy pdbs from the top clusters
    out_merged = args.destination / "merged_clusters"
    out_merged.mkdir()
    logging.info("Copying pdbs from the top clusters...")
    copy_pdbs(strclusters, pdbs_dir, out_merged)
    
    logging.info("Making Pymol sessions...")
    make_pymol_sessions(strclusters, out_merged)

    strclusters.to_csv(out_merged / "scores_clusters.csv")

    logging.info("Done!!")
