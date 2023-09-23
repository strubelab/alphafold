import re
import pandas as pd
import json
from Bio import SeqIO
from pathlib import Path
from typing import List
from Bio.SeqRecord import SeqRecord

from typing import Dict, Tuple

import logging
logging.getLogger().setLevel(logging.INFO)

######## Functions for make_features and make_complexes

def read_sequences(input:str) -> list:
    """
    Get an input of either a fasta file or a directory and read all the sequences
    in it
    """
    
    input = Path(input)
    
    if not input.exists():
        raise ValueError('Input file or directory does not exist.')
    
    if input.is_file():
        sequences = list(SeqIO.parse(input, 'fasta'))
    elif input.is_dir():
        sequences = []
        for f in input.glob('*.fasta'):
            sequences += list(SeqIO.parse(f, 'fasta'))
    else:
        raise ValueError('Input must be either a fasta file or a directory with fasta files.')
    
    return sequences

def get_id(seqid:str) -> str:
    """
    Process and return the id of the sequence

    Args:
        seqid (str): full id of the sequence

    Returns:
        str: processed id
    """
    names = seqid.split('|')
    
    if len(names) > 1:
        return names[1]
    else:
        return names[0]


def check_missing_sequences(out_dir:Path, sequences:List[List[SeqRecord]]
                            ) -> Tuple[List[List[SeqRecord]], List[str]]:
    """
    Check which sequences don't have features created yet

    Args:
        out_dir (Path): Directory where the features are saved
        sequences (List[List[SeqRecord]]): Sequences to check

    Returns:
        Tuple[List[List[SeqRecord]], List[str]]
    """
    
    logging.info(f'Checking for existing features in {out_dir}...')
    
    all_ids = [get_id(s[0].id) for s in sequences]
    # all_ids = [s[0].id.split('|')[1] for s in sequences]

    # Get the ids of the sequences that have a `features.pkl` file
    completed = []
    missing_ids = []
    for sid in all_ids:
        features_file = out_dir / sid / 'features.pkl'
        if features_file.exists():
            completed.append(sid)
        else:
            missing_ids.append(sid)
    
    logging.info(f'Found {len(completed)} sequences with features already calculated.')

    # Get the sequences that don't have features created yet
    missing_sequences = [s for s in sequences if s[0].id.split('|')[1] not in completed]
    
    return missing_sequences, missing_ids


def validate_models(models:list) -> list:
    """
    Validate the names of the models to be run
    """
    
    all_models = [
        'model_1_multimer_v3',
        'model_2_multimer_v3',
        'model_3_multimer_v3',
        'model_4_multimer_v3',
        'model_5_multimer_v3'
    ]
    
    if models == ['all']:
        return all_models
    elif models == ['two']:
        return ['model_3_multimer_v3','model_4_multimer_v3']
    else:
        # Check that the models are valid
        if not all([m in all_models for m in models]):
            raise ValueError('Invalid model names provided.')
        
        return models

def check_existing_features(features_dir: Path, sequences: List[SeqRecord],
                            bait: SeqRecord=None) -> List[str]:
    """
    Check if the features for the bait and the candidates already exist

    Args:
        features_dir (Path)
        bait (SeqRecord)
        sequences (List[SeqRecord]): candidate sequences

    Raises:
        ValueError: Error if the features for the bait are not found

    Returns:
        List[str]: List of the ids of the candidate sequences that already have
                   features
    """
    
    if bait:
        # Features file for bait
        bait_id = get_id(bait.id)
        bait_features_file = features_dir / bait_id / 'features.pkl'
        if not bait_features_file.exists():
            raise ValueError(f'Features for bait {bait.id} not found in {features_dir}.')
    
    # Features files for candidates
    # all_ids = [s.id.split('|')[1] for s in sequences]
    all_ids = [get_id(s.id) for s in sequences]

    # Get the ids of the sequences that have a `features.pkl` file
    completed = []
    for sid in all_ids:
        features_file = features_dir / sid / 'features.pkl'
        if features_file.exists():
            completed.append(sid)
    
    return completed


def check_missing_models(completed: List[str], out_dir: Path,
                         bait: SeqRecord,
                         sequences: List[SeqRecord],
                         stoich: List[int],
                         screen_mode: bool,
                         models_to_run: List[str],
                         multimer_predictions_per_model: int
                         ) -> Tuple[List[SeqRecord], List[str], List[str]]:
    """
    Obtain the list of sequences that don't have a model yet

    Args:
        completed (List): List of ids of the sequences that have features
        out_dir (Path): Output directory for the models
        bait (SeqRecord): Bait sequence
        sequences (List[SeqRecord]): Candidate sequences
        stoich (List[int]): Stoichiometry of the complexes in [n, n] format.
        screen_mode (bool): Whether to the run was only in screen mode or not.
        models_to_run (int): Names of the models to run.
        multimer_predictions_per_model (int): Number of predictions to make per model.
    Returns:
        List[SeqRecord]: List of candidate sequences that don't have a model yet
    """

    bait_id = get_id(bait.id)
    nbait = stoich[0]
    ncandidate = stoich[1]
    # Get the ids of the sequences that have a model already created
    modeled = []
    missing = []
    if screen_mode:
        # If in screen mode, only look for the 'iptms.json' file
        for sid in completed:
            model_dir = out_dir / f'{bait_id}-{nbait}_{sid}-{ncandidate}'
            model_scores = model_dir / 'iptms.json'
            if model_scores.exists():
                modeled.append(sid)
            else:
                missing.append(sid)
    else:
        # If making full models, count all the ranked*.pdb files
        for sid in completed:
            model_dir = out_dir / f'{bait_id}-{nbait}_{sid}-{ncandidate}'
            model_files = list(model_dir.glob('ranked_*.pdb'))
            nmodels = len(models_to_run) * multimer_predictions_per_model
            if len(model_files) == nmodels:
                modeled.append(sid)
            else:
                missing.append(sid)
            

    to_model = [s for s in completed if s not in modeled]

    # Get the sequences to model
    sequences_to_model = [s for s in sequences if get_id(s.id) in to_model]
    
    return sequences_to_model, modeled, missing


def check_missing_homomers(completed: List[str], out_dir: Path,
                          sequences: List[SeqRecord],
                          stoich:int,
                          screen_mode: bool,
                          models_to_run: List[str],
                          multimer_predictions_per_model: int,
                          ) -> Tuple[List[SeqRecord], List[str], List[str]]:
    """
    Obtain the list of sequences that don't have a model yet

    Args:
        completed (List): List of ids of the sequences that have features
        out_dir (Path): Output directory for the models
        sequences (List[SeqRecord]): Sequences to model
        stoich (int): Stoichiometry of the homomers
    Returns:
        List[SeqRecord]: List of sequences that don't have a model yet
    """
    # Get the ids of the sequences that have a model already created
    modeled = []
    missing = []
    if screen_mode:
        for sid in completed:
            # If in screen mode, only look for the 'iptms.json' file
            model_scores = out_dir / f'{sid}-{stoich}' / 'iptms.json'
            if model_scores.exists():
                modeled.append(sid)
            else:
                missing.append(sid)
    else:
        for sid in completed:
            # If making full models, count all the ranked*.pdb files
            model_dir = out_dir / f'{sid}-{stoich}'
            model_files = list(model_dir.glob('ranked_*.pdb'))
            nmodels = len(models_to_run) * multimer_predictions_per_model
            if len(model_files) == nmodels:
                modeled.append(sid)
            else:
                missing.append(sid)

    to_model = [s for s in completed if s not in modeled]

    # Get the sequences to model
    sequences_to_model = [s for s in sequences if get_id(s.id) in to_model]
    
    return sequences_to_model, modeled,  missing


######## Find random seeds

def get_scores(models_dir:Path) -> pd.DataFrame:
    
    dirs = list(models_dir.iterdir())
    
    dfs = []
    for d in dirs:
        ranking_file = d / "ranking_debug.json"
        if ranking_file.exists():
            with open(d / "ranking_debug.json") as f:
                iptm_ptms = json.load(f)['iptm+ptm']
            with open(d / "iptms.json") as f:
                iptms = json.load(f)['iptms']
            df = pd.DataFrame.from_dict(iptm_ptms, orient="index",
                                        columns=["iptm+ptm"])
            df["iptms"] = iptms
            df["complex"] = d.name
            # Reset the index
            df = df.reset_index(names="model")
            dfs.append(df)
    
    logging.info(f"Found {len(dfs)} model directories with quality scores.")

    # Concatenate all the DataFrames into one
    scores = pd.concat(dfs, ignore_index=True)

    scores = scores.sort_values(by=['complex','iptms'], ignore_index=True,
                            ascending=[True, False])

    return scores


def get_seeds(top_complexes: List[str], top_ligands: List[str],
              models_dir:Path) -> Dict[str, int]:
    """
    Read the random seeds from the stdout files of the given models

    Args:
        top_complexes (List[str]): Names of top complexes
        top_ligands (List[str]): Names of the top ligands
        models_dir (Path): Directory with the models and the `out_ibe` directory

    Returns:
        Dict[str, int]: Dictionary with {ligand: seed} pairs
    """
    # Find the files in which each top ligand is run, and get the random seed
    out_files = list((models_dir / 'out_ibex').glob('*.out'))
    seeds = {}
    for i, complex in enumerate(top_complexes):
        for fout in out_files:
            with open(fout, 'r') as f:
                for line in f:
                    if re.search(r'Using random seed', line):
                        seed = int(re.search(r'Using random seed (\d+) for the',
                                             line).group(1))
                        
                        # The name of the complex has to be in the next line
                        next_line = next(f)
                        if complex in next_line:
                            seeds[top_ligands[i]] = seed
                            break
            
            # Check if the seed has been found
            if top_ligands[i] in seeds:
                break
    
    return seeds


def get_errors(missing_models: List[str], models_dir:Path) -> Dict[str, int]:
    """
    Read the random seeds from the stdout files of the given models

    Args:
        missing_models (List[str]): Ids of the missing candidates
        top_ligands (List[str]): Names of the top ligands
        models_dir (Path): Directory with the models and the `out_ibe` directory

    Returns:
        Dict[str, int]: Dictionary with {ligand: seed} pairs
    """
    # Find the files in which each top ligand is run, and get the random seed
    out_files = list((models_dir / 'out_ibex').glob('*.out'))
    error_files = {}
    for candidate in missing_models:
        print(candidate)
        for fout in out_files:
            with open(fout, 'r') as f:
                lines = f.readlines()
                for line in lines:
                    if re.search(rf'{candidate}', line):
                        print('')
                        print(fout)
                        print(lines[-1])
                        error_files[candidate] = fout
                        break
            # Check if the candidate has been found
            if candidate in error_files:
                break
    
    return error_files