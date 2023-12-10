#!/usr/bin/env python3

"""
This script performs the following actions to cluster structures:
1. Run mmseqs easy-cluster to cluster the candidates by sequence identity
2. Run foldseek easy-cluster to cluster the models by structure
3. Merge the sequence and structure clusters, and save the results
6. Align all vs all members of each cluster, and save the alignment scores
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
    
    parser.add_argument("--cpus",
        help=('Number of CPUs to use. (Default=1)'), default=1, type=int)
    
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
    member_counts = clusters.rep.value_counts()
    # Get the clusters with more than one member
    top_clusters = member_counts[member_counts >= min_count].index
    # Get the members of the top clusters
    top_clusters_members = {r:set(clusters[clusters.rep==r].member.tolist()) \
                           for r in top_clusters}
    
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

        merged_members = members
        keys_to_merge = []
        for rep2, members2 in unmerged_dict.items():
            if members2 is None:
                continue
            if not merged_members.isdisjoint(members2):
                merged_members = merged_members | members2
                keys_to_merge.append(rep2)

        # Erase the values of the merged clusters
        for rep2 in keys_to_merge:
            unmerged_dict[rep2] = None
        
        # Check in the joint dictionary if the merged cluster has elements in
        # common with other clusters
        keys_to_merge = []
        for rep2, members2 in joint_dict.items():
            if not merged_members.isdisjoint(members2):
                merged_members = merged_members | members2
                keys_to_merge.append(rep2)
        
        # Erase the old values from the joint dictionary
        for rep2 in keys_to_merge:
            del joint_dict[rep2]

        joint_dict[rep] = merged_members

    return joint_dict


def joint_cluster(seqclusters:pd.DataFrame, strclusters:pd.DataFrame) -> Dict[str, Set[str]]:
    """
    Function to merge the clusters from the structure and sequence clustering.
    The top clusters from the structure clustering are taken as the base, and
    the top clusters from the sequence clustering are merged into them. If a
    sequence cluster has elements in common with more than one structure
    cluster, they are merged into the same cluster.

    Args:
        seqclusters (pd.DataFrame): sequence clusters
        strclusters (pd.DataFrame): structure clusters

    Returns:
        Dict[str, Set[str]]: Dictionary with the merged clusters. The keys are
            the cluster representatives, and the values are the members of the
            cluster. 
    """
    
    top_seqclusters_members = get_topcluster_members(seqclusters)
    top_strclusters_members = get_topcluster_members(strclusters)
    
    # Iterate over the top structure clusters, and see if they have elements in
    # common with the top sequence clusters. If they do, join them.
    joint_clusters = {}
    merged_seqclusters = []
    for rep, members in top_strclusters_members.items():
        merged_members = members
        for rep2, members2 in top_seqclusters_members.items():
            if not merged_members.isdisjoint(members2):
                merged_members = merged_members | members2
                merged_seqclusters.append(rep2)
        
        joint_clusters[rep] = merged_members

    # Merge the missing sequence clusters
    missing_seqclusters = set(top_seqclusters_members.keys()) - set(merged_seqclusters)
    if missing_seqclusters:
        for rep in missing_seqclusters:
            joint_clusters[rep] = top_seqclusters_members[rep]
            
    # Merge joint clusters further in cases where they have elements in common
    joint_clusters2 = merge_dict_values(joint_clusters)
    
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


def align_all(clusters:pd.DataFrame,
              pdbs_dir: Path, cpus:int=1) -> pd.DataFrame:
    """
    Align all vs all the members of every cluster

    Args:
        clusters (pd.DataFrame): DataFrame with the cluster representatives and
            members
        pdbs_dir (Path): Path to the directory with the models
        cpus (int, optional): Number of CPUs to use. Defaults to 1.

    Returns:
        pd.DataFrame: DataFrame with the alignment scores
    """
    # Filter out clusters with no merged_rep
    clusters_names = clusters[clusters.merged_rep.notna()].merged_rep.unique()
    len_clusters = len(clusters_names)
    
    aligned_dfs = []
    for i, cluster in enumerate(clusters_names):
        
        logging.info(f"Aligning cluster {i+1} of {len_clusters}...")
        
        members = list(clusters[clusters.merged_rep == cluster].member.values)
        
        logging.info(f"{len(members)} members.")
        
        member_combinations = combinations(members, 2)
        member_paths = [(pdbs_dir / (m1 + ".pdb"), pdbs_dir / (m2 + ".pdb")) \
                                        for m1, m2 in member_combinations]
        
        with multiprocessing.Pool(cpus) as pool:
            
            results = pool.starmap(calculate_tmscore, member_paths)
        
        
        aligned_scores = [(cluster, m1, m2, tmscore_m1, tmscore_m2, aligned_length, rmsd) \
                            for (m1, m2), (aligned_length, rmsd, tmscore_m1, tmscore_m2) \
                            in zip(member_combinations, results)]

        # Make dataframe
        columns = ['cluster', 'ref', 'member', 'tmscore_ref', 'tmscore_m',
                'aligned_length', 'rmsd']
        aligned_df = pd.DataFrame(aligned_scores, columns=columns)
        aligned_dfs.append(aligned_df)

    return pd.concat(aligned_dfs)


def medians_alignments(alignment_scores:pd.DataFrame,
                       clusters:pd.DataFrame) -> pd.DataFrame:
    """
    Get the median alignment scores for each cluster

    Args:
        alignment_scores (pd.DataFrame): DataFrame with the alignment scores
        clusters (pd.DataFrame): DataFrame with the cluster representatives and
                                 members
        pdbs_dir (Path): Path to the directory with the models
    Returns:
        pd.DataFrame: DataFrame with the median alignment scores
    """
    
    # Concatenate the scores for the reference and the member
    columns = ['cluster', 'ref', 'tmscore_ref', 'rmsd', 'aligned_length']
    scores1 = alignment_scores[columns].copy().rename(
                                columns={'ref': 'member', 'tmscore_ref': 'tmscore'})

    columns = ['cluster', 'member', 'tmscore_m', 'rmsd', 'aligned_length']
    scores2 = alignment_scores[columns].copy().rename(
                                columns={'tmscore_m': 'tmscore'})

    all_scores = pd.concat([scores1, scores2])
    
    # Create a column for cluster size
    all_scores['cluster_size'] = all_scores.cluster.map(
                                             clusters.merged_rep.value_counts())
    
    # Calculate the median values for each member of each cluster
    median_scores = all_scores.groupby(['cluster', 'member']).median().reset_index()

    return median_scores


def add_binder_fraction(median_scores:pd.DataFrame, pdbs_dir:Path) -> pd.DataFrame:
    """
    Add the fraction of the binder in the alignment to the median scores
    """

    # Create column for the fraction of the binder in the alignment
    median_scores['fraction_binder'] = 0.0
    for i in median_scores.index:
        m = median_scores.loc[i, 'member']
        m_pdb = pdbs_dir / (m + ".pdb")
        
        model = b.PDBModel(os.fspath(m_pdb))
        length_bait = len(model.takeChains([0]).sequence())
        length_binder = len(model.takeChains([1]).sequence())
        
        aligned_length = median_scores.loc[i, 'aligned_length']
        median_scores.loc[i, 'fraction_binder'] = ((aligned_length - length_bait)
                                                                / length_binder)

    return median_scores


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
    
    out_merged = args.destination / "merged_clusters"
    out_merged.mkdir(exist_ok=True)
    
    strclusters = joint_clusters_df(seqclusters, strclusters)
    strclusters.to_csv(out_merged / "scores_clusters.csv")
    
    logging.info("Aligning all vs all members of each cluster...")
    alignment_scores = align_all(strclusters, pdbs_dir, cpus=args.cpus)
    alignment_scores.to_csv(out_merged / "alignment_scores.csv", index=False)

    logging.info("Calculating median alignment scores...")
    median_scores = medians_alignments(alignment_scores, strclusters)
    median_scores = add_binder_fraction(median_scores, pdbs_dir)
    median_scores.to_csv(out_merged / "median_scores.csv", index=False)
    
    logging.info("Done!!")
