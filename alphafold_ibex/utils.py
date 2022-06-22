import pickle
import re
import jax
import numpy as np


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
    model_rank = list(outs.keys())
    model_rank = [model_rank[i] for i in \
                  np.argsort([outs[x]['pLDDT'] for x in model_rank])[::-1]]

    return prediction_results, outs, model_rank