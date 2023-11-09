"""
Script to extract the regions of the candidate models that have a high PAE score
against the bait
"""

import argparse
import json
import logging
import os
import pickle
import re
from pathlib import Path
from typing import Tuple

import biskit as b
import numpy as np
import pandas as pd
from Bio import SeqIO

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
        d = Path(d)
        if not d.exists():
            d.mkdir()
        return d
    
    def validate_pae(n:str) -> float:
        """
        Validate that the provided pae number is between 0 and 30
        """
        n = float(n)
        if n < 0 or n > 30:
            raise ValueError("The PAE threshold must be between 0 and 30.")
        return n

    parser = argparse.ArgumentParser(description=('Reads all the quality scores'
                ' for the models in the given directory and ranks them. Outputs '
                'are a table with the quality scores, a fasta file with the '
                'best sequences and a text file with the random seeds for '
                'making the complexes again.'))

    parser.add_argument("--models_dir", 
        help=('Path where the quality scores of the models are saved.'),
        type=validate_dir, required=True)
    
    parser.add_argument("--out_dir",
        help=('Path where the outputs will be saved. Defaults to a `scores` '
              'director inside of `models_dir`.'), type=validate_out,
        default=None)
    
    parser.add_argument("--pae_threshold",
        help=('Threshold for the PAE score to select the region of the binder'
              ' that will be used for clustering. Defaults to 15.'),
        type=validate_pae, default=15)
 
    return parser.parse_args(args)


def get_pae(model_dir:Path) -> np.ndarray:
    """
    Read the results dictionary of the top model from the pickle file
    """
    with open(model_dir / 'ranking_debug.json', 'r') as f:
        model_rank = json.load(f)['order']
        
    pickle_name = model_dir / f'result_{model_rank[0]}.pkl'
    
    with open(pickle_name, 'rb') as f:
        results = pickle.load(f)

    pae = results['predicted_aligned_error']
    
    return pae


def get_lowpae_indices(pae:np.ndarray, slength:int, pthresh:float) -> Tuple[int, int]:
    """
    Obtain the indices of the first and last appearances of a pae score below
    the threshold.

    Args:
        pae (np.ndarray): _description_
        blength (int): _description_
        pthresh (float): _description_

    Returns:
        float: _description_
    """
    
    # Extract the rows corresponding to the first sequence and all the columns.
    # Then calculate the mean of each column.
    mean_pae1 = np.mean(pae[:slength,:], axis=0)
    # Extract only the values in the intersection with the second sequence
    mean_pae1 = mean_pae1[slength:]
    
    belowthresh_indices1 = np.where(mean_pae1 <= pthresh)
    btleft1 = belowthresh_indices1[0][0]
    btright1 = belowthresh_indices1[0][-1]

    # Extract the columns corresponding to the first sequence and all the rows.
    # Then calculate the mean of each row.
    mean_pae2 = np.mean(pae[:, :slength], axis=1)
    # Extract only the values in the intersection with the second sequence
    mean_pae2 = mean_pae2[slength:]
    
    belowthresh_indices2 = np.where(mean_pae2 <= pthresh)
    btleft2 = belowthresh_indices2[0][0]
    btright2 = belowthresh_indices2[0][-1]
    
    return min(btleft1, btleft2), max(btright1, btright2)


def extract_pdb_region(pdb:Path, leftind:int, rightind:int, destination:Path
                       ) -> None:
    """
    Extract the region of the second chain of the pdb that is between the left
    and right indices
    """
    
    m = b.PDBModel(os.fspath(pdb))
    chainb = m.takeChains([1])
    chainb = chainb.takeResidues(list(np.arange(leftind, rightind)))
    
    m2 = m.takeChains([0]).concat(chainb)
    
    # Get new name to write pdb
    uid = re.search(r'-1_(\w+)-1', pdb.parent.name).group(1)
    new_name = destination / (uid + ".pdb")
    
    m2.writePDB(new_name)


if __name__ == "__main__":
    
    args = parsing()
    pdbs_dir = args.destination / "pdbs_trimmed"
    
    trimmed_candidates = []
    candidates_regions = []
    
    count = 0
    for d in args.models_dir.iterdir():
        if d.isdir():
            top_pdb = d / "ranked_0.pdb"
            if top_pdb.exists():
                count += 1
                
                pae = get_pae(d)
                
                # Read sequences
                ffile = list(d.glob('*.fasta'))[0]
                sequences = list(SeqIO.parse(ffile, 'fasta'))
                
                # Get the indices of the region with low PAE
                leftind, rightind = get_lowpae_indices(pae, len(sequences[0]),
                                                       args.pae_threshold)
                
                # Read pdb and extract the region
                extract_pdb_region(top_pdb, leftind, rightind, pdbs_dir)
                
                # Save the sequences of the region
                seq_trimmed = sequences[1][leftind:rightind]
                trimmed_candidates.append(seq_trimmed)
                
                candidates_regions.append((seq_trimmed.id, leftind, rightind))
    
    logging.info(f"Processed {count} complexes.")
    
    logging.info("Writing trimmed sequences to fasta file.")
    SeqIO.write(trimmed_candidates, args.destination / "trimmed_candidates.fasta",
                'fasta')
    
    # Write the regions of the candidates to a file
    pd.DataFrame(candidates_regions, columns=['id', 'left', 'right']).to_csv(
        args.destination / "candidates_regions.csv", index=False)
