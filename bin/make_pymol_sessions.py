#!/usr/bin/env python3

"""
This script performs the following actions, as a continuation to clusters.py:
1. Obtain the top clusters from the merged clusters
4. Copy the PDBs from each cluster to a new directory
5. Make Pymol sessions for the clusters
6. Do structural clustering on the top clusters to find subclusters
"""

import argparse
from pathlib import Path
import shutil
import subprocess
import re
import os
from pymol import cmd
from typing import Union, Tuple

import pandas as pd
import biskit as b

import logging
logging.getLogger().setLevel(logging.INFO)

from alphafold_ibex.clustering_utils import run_structure_clustering

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

    parser = argparse.ArgumentParser(description=('Copy the top structures and '
        'plots to other directories.'))
    
    parser.add_argument("--clusters_dir",
        help=('Path to the directory with the merged clustering results.'),
        required=True, type=validate_dir)
    
    parser.add_argument("--top_models_dir",
        help=("Path with the top models for each complex. If this is given, "
              "the top models will NOT be copied to a new directory."),
        type=validate_dir, default=None)
    
    parser.add_argument("--min_members",
        help=("Minimum number of members for a cluster to be considered. "
              "Default: 5"),
        type=int, default=5)
    
    parser.add_argument("--min_tmscore",
        help=("Minimum median TM-score for a cluster to be considered. "
              "(Default: 0.2)"),
        type=float, default=0.2)
    
    parser.add_argument("--min_fraction_binder",
        help=("Minimum fraction of the binder for a cluster to be considered. "
              "(Default: 0.2)"),
        type=float, default=0.2)
    
    parser.add_argument("--min_size",
        help=("Minimum number of members for a cluster to be considered. If "
              "this is given, the clusters will be filtered by size after "
              "filtering by other criteria."),
        type=int, default=None)
    
    parser.add_argument("--max_rmsd",
        help=("Maximum RMSD for a cluster to be considered. (Default: 15.0)"),
        type=float, default=15.0)
    
    return parser.parse_args(args)
    


def copy_pdbs(clusters:pd.DataFrame, pdbs_dir:Path, destination:Path,
              topclusters:list) -> None:
    """
    Copy the top pdbs from the top clusters to a new directory
    
    Args:
        clusters (pd.DataFrame): DataFrame with the cluster representatives and
            members
        pdbs_dir (Path): Path to the directory with the models
        destination (Path): Path to the directory where the pdbs will be copied
        topclusters (list): List with the top clusters
    """
    
    for cluster in topclusters:
        # Get cluster members
        members = clusters[clusters.merged_rep == cluster].member.values
        cluster_dir = destination / f"cluster_{cluster}"
        cluster_dir.mkdir()
        for member in members:
            pdb_name = pdbs_dir / (member + ".pdb")
            assert pdb_name.exists(), f"{pdb_name} doesn't exist"
            new_name = cluster_dir / (member + ".pdb")
            shutil.copy(pdb_name, new_name)


def make_pymol_sessions(clusters:pd.DataFrame, destination:Path,
                        topclusters:list) -> None:
    """
    Create the pymol session with the superimposition of each of the clusters

    Args:
        clusters (pd.DataFrame): DataFrame with the clustering results
        destination (Path): Path to the directory with the pdbs for each cluster
        topclusters (list): List with the top clusters
    """
    
    for cluster in topclusters:
        # Get cluster members
        members = clusters[clusters.merged_rep == cluster].member.values
        cluster_dir = destination / f"cluster_{cluster}"

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


def cluster_clusters(destination:Path, topclusters:list):
    """
    For each cluster:
    1. Do structural clustering to identify the "consensus" structure of the cluster
    2. Use us-align to align all the members of the cluster to the consensus structure
    3. Save the alignment scores in a DataFrame
    
    Args:
        clusters (pd.DataFrame): DataFrame with the cluster representatives and
            members
        destination (Path): Path to the directory with the pdbs for each cluster
        topclusters (list): List with the top clusters
    """
    clustered_clusters = []
    for cluster in topclusters:

        logging.info(f"Clustering {cluster}")

        cluster_dir = destination / f"cluster_{cluster}"
        cluster_clusters_dir = destination / f"cluster_{cluster}_clusters"
        if not cluster_clusters_dir.exists():
            cluster_clusters_dir.mkdir()
        
        # Do structural clustering on the merged chains
        strclusters, pdbs_dir = run_structure_clustering(
                                    destination=cluster_clusters_dir,
                                    top_models_dir=cluster_dir)
        
        # Get only the clusters for the structures of the binder
        strclusters = strclusters[strclusters.member.str.endswith('_B')]
        # Remove the '.pdb_B' suffix from the members' names
        strclusters['member'] = strclusters['member'].str.split('.pdb').str[0]
        strclusters['cluster'] = cluster
        
        strclusters.rename(columns={'rep':'subcluster_rep'}, inplace=True)
        
        clustered_clusters.append(strclusters)
    
    return pd.concat(clustered_clusters)


def get_top_clusters(median_scores:pd.DataFrame,
                     min_members:int,
                     min_tmscore:float=0.2,
                     min_fraction_binder:float=0.2,
                     min_size:int=None,
                     max_rmsd:float=15.0) -> list:
    """
    Get the top clusters based on the median alignment score and fraction of
    the binder

    Args:
        median_scores (pd.DataFrame): DataFrame with the median alignment scores
        min_members (int): Minimum number of members for a cluster to be
            considered
        min_tmscore (float): Minimum median TM-score for a cluster to be
            considered
        min_fraction_binder (float): Minimum fraction of the binder for a
            cluster to be considered

    Returns:
        list of str: List with the top clusters
    """

    # Select the clusters based on all the criteria
    select = ((median_scores.cluster_size >= min_members) & \
              (median_scores.tmscore >= min_tmscore) & \
              (median_scores.fraction_binder >= min_fraction_binder) & \
              (median_scores.rmsd <= max_rmsd))
    median_scores_filtered = median_scores[select]

    if min_size:
        select = median_scores_filtered.cluster_size >= min_size
        clusters = list(median_scores_filtered[select].cluster.unique())
    else:
        clusters = list(median_scores_filtered.cluster.unique())
    
    return clusters


if __name__ == '__main__':

    args = parsing()
    
    strclusters = pd.read_csv(args.clusters_dir / "scores_clusters.csv")
    alignment_scores = pd.read_csv(args.clusters_dir / "alignment_scores.csv")
    median_scores = pd.read_csv(args.clusters_dir / "median_scores.csv")
    pdbs_dir = args.top_models_dir
    out_merged = args.clusters_dir
    
    clusters = get_top_clusters(median_scores, args.min_members,
                                args.min_tmscore, args.min_fraction_binder,
                                args.min_size, args.max_rmsd)
    
    logging.info(f"Identified {len(clusters)} top clusters.")
    logging.info(f"Top clusters: {clusters}")
    
    logging.info("Copying pdbs from the top clusters...")
    copy_pdbs(strclusters, pdbs_dir, out_merged, clusters)
    
    logging.info("Clustering clusters...")
    clustered_clusters = cluster_clusters(out_merged, clusters)
    clustered_clusters.to_csv(out_merged / "clustered_clusters.csv", index=False)
    
    logging.info("Making Pymol sessions...")
    make_pymol_sessions(strclusters, out_merged, clusters)

    logging.info("Done!!")
