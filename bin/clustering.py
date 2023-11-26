#!/usr/bin/env python3

"""
This script performs the following actions to cluster structures:
1. Run mmseqs easy-cluster to cluster the candidates by sequence identity
2. Run foldseek easy-cluster to cluster the models by structure
3. Merge the sequence and structure clusters
4. Copy the PDBs from each joint cluster to a new directory
5. Make Pymol sessions for the top clusters
6. Align all vs all members of each cluster, and save the alignment scores
"""

import argparse
from pathlib import Path
import shutil
import subprocess
import re
import os
from pymol import cmd
from typing import Dict, Set, Union, Tuple

import pandas as pd
import numpy as np
import biskit as b

import logging
logging.getLogger().setLevel(logging.INFO)

from alphafold_ibex.run_usalign import calculate_tmscore

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
    
    parser.add_argument("--top_models_dir",
        help=("Path with the top models for each complex. If this is given, "
              "the top models will NOT be copied to a new directory."),
        type=validate_dir, default=None)
    
    parser.add_argument("--candidates", 
        help=('FASTA file with the sequences to be clustered.'), required=True,
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
    
    parser.add_argument("--top_n",
        help=('Number of top clusters to copy. (Default=10)'), default=10,
        type=int)
    
    return parser.parse_args(args)



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
    out_seqcluster.mkdir()

    logging.info("Running sequence clustering...")
    command = (f"mmseqs easy-cluster {candidates_file} {results_prefix} {temp_dir} "
               f"--min-seq-id {min_seq_id} -c {coverage} --cov-mode {cov_mode} "
               f"-s {sensitivity}").split()
    
    p = subprocess.run(command, cwd=out_seqcluster, capture_output=True)

    logging.info("Processing output...")
    seqclusters_tsv = out_seqcluster / "clusterRes_cluster.tsv"
    seqclusters = pd.read_table(seqclusters_tsv, header=None, names=["rep", "member"])

    return seqclusters
    

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

    logging.info("Running strcutural clustering...")
    command = (f"foldseek easy-cluster {pdbs_dir} {results_prefix} {temp_dir} "
               f"-c {coverage} --cov-mode {cov_mode} -e {evalue}").split()
    
    p = subprocess.run(command, cwd=out_strcluster, capture_output=True)
    
    logging.info("Processing output...")
    strclusters_tsv = out_strcluster / "clusterRes_cluster.tsv"
    strclusters = pd.read_table(strclusters_tsv, header=None, names=["rep", "member"])

    return strclusters, pdbs_dir


def add_quality_scores(strclusters:pd.DataFrame, seqclusters:pd.DataFrame,
                       models_dir:Path) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Add the quality scores to the clusters

    Args:
        strclusters (pd.DataFrame): DataFrame with the structural clusters
        seqclusters (pd.DataFrame): DataFrame with the sequence clusters
        models_dir (Path): Path to the directory with the models from AlphaFold
                           and pdockq scores

    """
    
    logging.info("Adding quality scores to the clusters...")
    pdockq = pd.read_csv(models_dir / "scores/scores_pdockq.csv")
    pdockq['binder'] = pdockq.complex.str.split('_').str[1].str[:-2]

    strclusters = pdockq.merge(strclusters, how='left', left_on='binder',
                                right_on='member')

    seqclusters = pdockq.merge(seqclusters, how='left', left_on='binder',
                                right_on='member')
    
    return strclusters, seqclusters


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
              topclusters:pd.DataFrame) -> None:
    """
    Copy the top pdbs from the top clusters to a new directory
    
    Args:
        clusters (pd.DataFrame): DataFrame with the cluster representatives and
            members
        pdbs_dir (Path): Path to the directory with the models
        destination (Path): Path to the directory where the pdbs will be copied
        top_n (int): Number of top clusters to copy
    """
    
    for i, cluster in enumerate(topclusters.index):
        # Get cluster members
        members = clusters[clusters.merged_rep == cluster].member.values
        cluster_dir = destination / f"cluster{i+1}_{cluster}"
        cluster_dir.mkdir()
        for member in members:
            pdb_name = pdbs_dir / (member + ".pdb")
            assert pdb_name.exists(), f"{pdb_name} doesn't exist"
            new_name = cluster_dir / (member + ".pdb")
            shutil.copy(pdb_name, new_name)


def make_pymol_sessions(clusters:pd.DataFrame, destination:Path,
                        topclusters:pd.DataFrame) -> None:
    """
    Create the pymol session with the superimposition of each of the clusters

    Args:
        clusters (pd.DataFrame): DataFrame with the clustering results
        destination (Path): Path to the directory with the pdbs for each cluster
        topclusters (pd.DataFrame): DataFrame with the top clusters
    """
    
    for i, cluster in enumerate(topclusters.index):
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


def cluster_clusters(clusters:pd.DataFrame, destination:Path,
                     topclusters:pd.DataFrame):
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
        topclusters (pd.DataFrame): DataFrame with the top clusters
    """
    aligned_dfs = []
    for i, cluster in enumerate(topclusters.index):

        logging.info(f"Clustering {cluster}")

        members = clusters[clusters.merged_rep == cluster].member.values
        cluster_dir = destination / f"cluster{i+1}_{cluster}"
        cluster_merged = destination / f"cluster{i+1}_{cluster}_merged"
        if not cluster_merged.exists():
            cluster_merged.mkdir()
        cluster_clusters_dir = destination / f"cluster{i+1}_{cluster}_clusters"
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
        # This is the ID of the largest cluster representative
        # It will be used to align all the members of the supercluster
        largest_rep = strclusters.rep.value_counts().index[0]
        
        ## use us-align to align all the members of the supercluster
        ## to the largest cluster representative
        logging.info("Aligning members to cluster representative...")
        aligned_scores = []
        for m in members:
            subcluster_rep = strclusters[strclusters.member == m].rep.values[0]
            m_path = cluster_dir / (m + ".pdb")
            rep_path = cluster_dir / (largest_rep + ".pdb")
            aligned_length, rmsd, tmscore = calculate_tmscore(m_path, rep_path)
            aligned_scores.append((cluster, largest_rep, m, aligned_length,
                                    rmsd, tmscore, subcluster_rep))
        
        # Make dataframe
        columns = ['cluster', 'rep_align', 'member', 'aligned_length', 'rmsd',
                     'tmscore', 'subcluster_rep']
        aligned_df = pd.DataFrame(aligned_scores, columns=columns)
        
        aligned_dfs.append(aligned_df)
    
    return pd.concat(aligned_dfs)


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


def joint_clusters_df(seqclusters:pd.DataFrame, strclusters:pd.DataFrame
                          ) -> pd.DataFrame:
    """
    Get the DataFrame with the joint clusters from the clusters of the structures
    and the clusters of the sequences.
    """
    
    joint_clusters = joint_cluster(seqclusters, strclusters)
    
    # Modify old columns and set `complex` as the index
    strclusters = strclusters.rename(columns={'rep':'str_rep'}).set_index('complex')
    seqclusters = seqclusters.rename(columns={'rep':'seq_rep'}).set_index('complex')

    # Add the `seq_rep` column to the `strclusters` dataframe
    strclusters['seq_rep'] = seqclusters.seq_rep
    
    # Initialize a new column `merged_rep`
    strclusters['merged_rep'] = None
    
    # Iterate over joint_clusters and change the values of the `merged_rep` column
    # in the `strclusters` dataframe
    for rep, members in joint_clusters.items():
        strclusters.loc[strclusters.member.isin(members), 'merged_rep'] = rep

    strclusters.fillna(np.nan, inplace=True)
    
    return strclusters


def align_all(clusters:pd.DataFrame, destination: Path,
              topclusters:pd.DataFrame) -> pd.DataFrame:
    """
    Align all vs all the members of each of the top clusters

    Args:
        clusters (pd.DataFrame): DataFrame with the cluster representatives and
            members
        destination (Path): Path to the directory with the pdbs for each cluster
        topclusters (pd.DataFrame): DataFrame with the top clusters

    Returns:
        pd.DataFrame: DataFrame with the alignment scores
    """
    aligned_dfs = []
    for i, cluster in enumerate(topclusters.index):
        
        logging.info("Aligning {cluster}...")
        
        members = clusters[clusters.merged_rep == cluster].member.values
        cluster_dir = destination / f"cluster{i+1}_{cluster}"
        
        aligned_scores = []
        while members:
            ref = members.pop()
            ref_path = cluster_dir / (ref + ".pdb")
            
            for m in members:
                m_path = cluster_dir / (m + ".pdb")
                aligned_length, rmsd, tmscore_m, tmscore_ref = calculate_tmscore(
                                                                m_path, ref_path)
                aligned_scores.append((cluster, ref, m, tmscore_ref, tmscore_m,
                                       aligned_length, rmsd))

        # Make dataframe
        columns = ['cluster', 'ref', 'member', 'tmscore_ref', 'tmscore_m',
                'aligned_length', 'rmsd']
        aligned_df = pd.DataFrame(aligned_scores, columns=columns)
        aligned_dfs.append(aligned_df)

    return pd.concat(aligned_dfs)


if __name__ == '__main__':

    args = parsing()
    
    seqclusters = run_sequence_clustering(args.destination, args.candidates,
                        results_prefix="clusterRes", temp_dir="tmp",
                        min_seq_id=args.min_seq_id, coverage=args.coverage,
                        cov_mode=args.cov_mode, sensitivity=args.sensitivity)
    
    strclusters, pdbs_dir = run_structure_clustering(args.destination,
                        args.top_models_dir, args.models_dir,
                        results_prefix="clusterRes", temp_dir="tmp",
                        coverage=args.coverage, cov_mode=args.cov_mode,
                        evalue=args.evalue)
    
    # Get only the clusters for the structures of the binder
    strclusters = strclusters[strclusters.member.str.endswith('_B')]
    # See if all the representatives also come from chain B
    if not all(strclusters.rep.str.endswith('_B')):
        logging.info("NOT ALL REPRESENTATIVES COME FROM CHAIN B")
    # Remove the '.pdb_B' suffix from the members' names
    strclusters['member'] = strclusters['member'].str.split('.pdb').str[0]
    
    
    strclusters, seqclusters = add_quality_scores(strclusters, seqclusters,
                                                  args.models_dir)
    
    strclusters = joint_clusters_df(seqclusters, strclusters)

    # Copy the pdbs from the top clsuters to a new directory
    # and make pymol sessions for the top clusters
    out_merged = args.destination / "merged_clusters"
    out_merged.mkdir()
    topclusters = (strclusters.groupby('merged_rep')
                   .agg({'iptms': 'mean', 'binder': 'count'})
                   .sort_values(by=['binder', 'iptms'], ascending=False)
                   .head(args.top_n))
    logging.info("Copying pdbs from the top clusters...")
    copy_pdbs(strclusters, pdbs_dir, out_merged, topclusters)
    
    logging.info("Making Pymol sessions...")
    make_pymol_sessions(strclusters, out_merged, topclusters)
    
    strclusters.to_csv(out_merged / "scores_clusters.csv")

    # logging.info("Clustering clusters...")
    # alignment_scores = cluster_clusters(strclusters, out_merged, topclusters)
    
    logging.info("Aligning all vs all members of each cluster...")
    alignment_scores = align_all(strclusters, out_merged, topclusters)
    
    alignment_scores.to_csv(out_merged / "alignment_scores.csv", index=False)

    logging.info("Done!!")
