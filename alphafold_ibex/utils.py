import pickle
import re
import jax
import numpy as np

from Bio import SeqIO
from pathlib import Path
from typing import List
from Bio.SeqRecord import SeqRecord

######## PROCESS SEQUENCE INPUTS ########

def define_homooligomers(sequences:list):

    unique_sequences = []
    homooligomers = []
    unique_names = []
    for seq in sequences:
        s = str(seq.seq)
        if s not in unique_sequences:
            unique_sequences.append(s)
            unique_names.append(seq.name)
            homooligomers.append(1)
        else:
            ind = unique_sequences.index(s)
            homooligomers[ind] += 1
    
    ## set chainbreaks
    chain_breaks = []
    for seq,h in zip(unique_sequences, homooligomers):
        chain_breaks += [len(seq)] * h
    
    return chain_breaks, homooligomers, unique_names
    

######## PROCESS OUTPUTS ########

def parse_results(prediction_result):

    dist_bins = jax.numpy.append(0,prediction_result["distogram"]["bin_edges"])
    dist_logits = prediction_result["distogram"]["logits"]
    dist_mtx = dist_bins[dist_logits.argmax(-1)]
    contact_mtx = jax.nn.softmax(dist_logits)[:,:,dist_bins < 8].sum(-1)

    plddt = prediction_result['plddt']
    
    to_np = lambda a: np.asarray(a)
    out = {
        "plddt": to_np(plddt),
        "pLDDT": to_np(plddt.mean()),
        "dists": to_np(dist_mtx),
        "adj": to_np(contact_mtx),
        "pae": to_np(prediction_result['predicted_aligned_error']),
        "pTMscore": to_np(prediction_result['ptm'])
            }

    return out


def process_outputs(features_files:list):

    prediction_results = {}

    for file in features_files:
        name = re.search(r'model_.+', file.stem).group()
        with open(file, 'rb') as f:
            features = pickle.load(f)
            prediction_results[name]=features

    outs = {key : parse_results(value) for key, value in \
            prediction_results.items()}
    
    # Rank models according to average pLDDT
    # model_rank = list(outs.keys())
    # model_rank = [model_rank[i] for i in \
    #               np.argsort([outs[x]['pLDDT'] for x in model_rank])[::-1]]

    return prediction_results, outs#, model_rank


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


def check_missing_sequences(out_dir:Path, sequences:List[List[SeqRecord]]):
    
    print(f'Checking for existing features in {out_dir}...')
    
    all_ids = [get_id(s[0].id) for s in sequences]
    # all_ids = [s[0].id.split('|')[1] for s in sequences]

    # Get the ids of the sequences that have a `features.pkl` file
    completed = []
    for sid in all_ids:
        features_file = out_dir / sid / 'features.pkl'
        if features_file.exists():
            completed.append(sid)
    
    print(f'Found {len(completed)} sequences with features already calculated.')

    # Get the sequences that don't have features created yet
    missing_sequences = [s for s in sequences if s[0].id.split('|')[1] not in completed]
    
    return missing_sequences


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
    
    print(f'Checking for existing features in {features_dir}...')
    
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
    
    print(f'Found {len(completed)} features for {len(sequences)} candidate sequences.')
    
    return completed


def check_missing_models(completed: List[str], out_dir: Path, bait: SeqRecord,
                          sequences: List[SeqRecord]) -> List[SeqRecord]:
    """
    Obtain the list of sequences that don't have a model yet

    Args:
        completed (List): List of ids of the sequences that have features
        out_dir (Path): Output directory for the models
        bait (SeqRecord): Bait sequence
        sequences (List[SeqRecord]): Candidate sequences

    Returns:
        List[SeqRecord]: List of candidate sequences that don't have a model yet
    """
    
    print(f"Checking for existing models in {out_dir}...")
    bait_id = get_id(bait.id)
    # Get the ids of the sequences that have a model already created
    modeled = []
    for sid in completed:
        model_scores = out_dir / f'{bait_id}-1_{sid}-1' / 'iptms.json'
        if model_scores.exists():
            modeled.append(sid)

    print(f'Found {len(modeled)} models for {len(sequences)} candidate sequences.')

    to_model = [s for s in completed if s not in modeled]

    # Get the sequences to model
    sequences_to_model = [s for s in sequences if get_id(s.id) in to_model]
    
    return sequences_to_model