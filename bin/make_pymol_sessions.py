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
        help=("Minimum number of members for a cluster to be considered."),
        type=int, default=5)
    
    parser.add_argument("--min_tmscore",
        help=("Minimum median TM-score for a cluster to be considered."),
        type=float, default=0.2)
    
    parser.add_argument("--min_fraction_binder",
        help=("Minimum fraction of the binder for a cluster to be considered."),
        type=float, default=0.2)
    
    return parser.parse_args(args)
    

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
    out_strcluster.mkdir()

    if top_models_dir is None:
        logging.info("Copying models to new directory...")
        pdbs_dir = get_top_pdbs(models_dir, out_strcluster)
    else:
        pdbs_dir = top_models_dir

    logging.info("Running structural clustering...")
    command = (f"foldseek easy-cluster {pdbs_dir} {results_prefix} {temp_dir} "
               f"-c {coverage} --cov-mode {cov_mode} -e {evalue}").split()
    
    p = subprocess.run(command, cwd=out_strcluster, capture_output=True)
    
    logging.info("Processing output...")
    strclusters_tsv = out_strcluster / "clusterRes_cluster.tsv"
    strclusters = pd.read_table(strclusters_tsv, header=None, names=["rep", "member"])

    return strclusters, pdbs_dir


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


def merge_chains(pdbs_dir:Path, destination:Path):
    """
    Merge the chains of a PDB into a single chain and write it to the destination

    Args:
        pdbs_dir (Path): Directory with the PDBs to merge
    """
    
    pdbs = pdbs_dir.glob("*.pdb")
    
    count=0
    for pdb in pdbs:
        
        m = b.PDBModel(os.fspath(pdb))
        try:
            len_a = len(m.takeChains([0]).atom2resProfile('residue_number'))
            chainb = m.takeChains([1])
        except:
            print(pdb)
            raise
        chainb.renumberResidues(start=len_a+31)
        m2 = m.takeChains([0]).concat(chainb)
        m2.mergeChains(0, renumberResidues=False)
        
        new_name = destination / (pdb.name)
        m2.writePdb(os.fspath(new_name))
        
        count += 1

    logging.info(f"Processed {count} models.")


def cluster_clusters(destination:Path, topclusters:list):
    """
    For each cluster:
    1. Merge the chains of each PDB in the cluster
    2. Do structural clustering to identify the "consensus" structure of the cluster
    3. Use us-align to align all the members of the cluster to the consensus structure
    4. Save the alignment scores in a DataFrame
    
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
        cluster_merged = destination / f"cluster_{cluster}_merged"
        if not cluster_merged.exists():
            cluster_merged.mkdir()
        cluster_clusters_dir = destination / f"cluster_{cluster}_clusters"
        if not cluster_clusters_dir.exists():
            cluster_clusters_dir.mkdir()
        
        # Merge the chains
        logging.info(f"Merging chains for cluster {cluster_dir.name}...")
        merge_chains(cluster_dir, cluster_merged)
        
        # Do structural clustering on the merged chains
        strclusters, pdbs_dir = run_structure_clustering(
                                    destination=cluster_clusters_dir,
                                    top_models_dir=cluster_merged)
        strclusters['member'] = strclusters['member'].str.split('.pdb').str[0]
        strclusters['rep'] = strclusters['rep'].str.split('.pdb').str[0]
        strclusters['cluster'] = cluster
        
        strclusters.rename(columns={'rep':'subcluster_rep'}, inplace=True)
        
        clustered_clusters.append(strclusters)
    
    return pd.concat(clustered_clusters)


def get_top_clusters(median_scores:pd.DataFrame,
                     min_members:int,
                     min_tmscore:float=0.2,
                     min_fraction_binder:float=0.2) -> list:
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

    # Select the clusters with at least 5 members
    select = median_scores.cluster_size >= min_members
    median_scores_filtered = median_scores[select]
    
    select = ((median_scores_filtered.tmscore >= min_tmscore) & \
              (median_scores_filtered.fraction_binder >= min_fraction_binder))
    median_scores_filtered = median_scores_filtered[select]

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
                                args.min_tmscore, args.min_fraction_binder)
    
    logging.info(f"Identified {len(clusters)} top clusters.")
    logging.info(f"Top clusters: {clusters}")
    
    logging.info("Copying pdbs from the top clusters...")
    copy_pdbs(strclusters, pdbs_dir, out_merged, clusters)
    
    logging.info("Making Pymol sessions...")
    make_pymol_sessions(strclusters, out_merged, clusters)

    logging.info("Clustering clusters...")
    clustered_clusters = cluster_clusters(out_merged, clusters)
    clustered_clusters.to_csv(out_merged / "clustered_clusters.csv", index=False)

    logging.info("Done!!")
