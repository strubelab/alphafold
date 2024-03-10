'''
Script and functions to score the interfaces of multimeric models.
Adapted from:
- https://github.com/KosinskiLab/AlphaPulldown/blob/main/alphapulldown/analysis_pipeline/get_good_inter_pae.py
- https://github.com/KosinskiLab/AlphaPulldown/blob/main/alphapulldown/analysis_pipeline/calculate_mpdockq.py
'''

import os
import numpy as np
import pickle
import json
from collections import defaultdict
import math
import pandas as pd

from typing import Union, Tuple, List, Dict


def parse_atm_record(line):
    '''Get the atm record
    '''
    record = defaultdict()
    record['name'] = line[0:6].strip()
    record['atm_no'] = int(line[6:11].strip())
    record['atm_name'] = line[12:16].strip()
    record['atm_alt'] = line[17]
    record['res_name'] = line[17:20].strip()
    record['chain'] = line[21]
    record['res_no'] = int(line[22:26].strip())
    record['insert'] = line[26].strip()
    record['resid'] = line[22:29]
    record['x'] = float(line[30:38])
    record['y'] = float(line[38:46])
    record['z'] = float(line[46:54])
    record['occ'] = float(line[54:60])
    record['B'] = float(line[60:66])

    return record


def read_pdb(pdbfile:str) -> Tuple[Dict[str,List[str]],
                                   Dict[str,List[List[float]]],
                                   Dict[str,List[int]],
                                   Dict[str,List[int]],
                                   Dict[str,np.ndarray]]:
    """
    Read a pdb file per chain

    Args:
        pdbfile (str): Path for the pdb file

    Returns:
        Tuple[dict,dict,dict,dict]: Dictionaries with various information per chain
    
    - pdb_chains: Dictionary with the lines of the pdb file for each chain. The
            keys are the chain ids and the values are lists of strings with
            the lines
    - chain_coords: Dictionary with the coordinates for each chain. The keys are
            the chain ids and the values are lists of lists with the x,y,z
            coordinates for each atom
    - chain_CA_inds: Dictionary with the indices of the CA atoms for each chain.
            The keys are the chain ids and the values are lists of integers
            with the indices
    - chain_CB_inds: Dictionary with the indices of the CB atoms for each chain.
            The keys are the chain ids and the values are lists of integers
            with the indices
    - res_indices: Dictionary with the residue numbers for each chain. The keys
            are the chain ids and the values are numpy arrays with the residue
            index for each atom
    """
    pdb_chains = {}
    chain_coords = {}
    chain_CA_inds = {}
    chain_CB_inds = {}
    res_indices = {}

    with open(pdbfile) as file:
        for line in file:
            if 'ATOM' in line:
                record = parse_atm_record(line)
                if record['chain'] in [*pdb_chains.keys()]:
                    pdb_chains[record['chain']].append(line)
                    chain_coords[record['chain']].append([record['x'],record['y'],record['z']])
                    res_indices[record['chain']].append(record['res_no']-1)
                    coord_ind+=1
                    if record['atm_name']=='CA':
                        chain_CA_inds[record['chain']].append(coord_ind)
                    if record['atm_name']=='CB' or (record['atm_name']=='CA' and record['res_name']=='GLY'):
                        chain_CB_inds[record['chain']].append(coord_ind)


                else:
                    pdb_chains[record['chain']] = [line]
                    chain_coords[record['chain']]= [[record['x'],record['y'],record['z']]]
                    res_indices[record['chain']] = [record['res_no']-1]
                    chain_CA_inds[record['chain']]= []
                    chain_CB_inds[record['chain']]= []
                    #Reset coord ind
                    coord_ind = 0

    # Convert residue numbers from list to np.array
    for k,v in res_indices.items():
        res_indices[k] = np.array(v)

    return pdb_chains, chain_coords, chain_CA_inds, chain_CB_inds,res_indices 


def get_best_plddt(work_dir:str) -> np.ndarray:
    """
    Gets the array with pLDDT scores for the best model
    
    Args:
        work_dir (str): Directory with the results from AlphaFold
    
    Returns:
        np.ndarray: Array with the pLDDT scores for the best model
    """
    json_path = os.path.join(work_dir,'ranking_debug.json')
    best_model = json.load(open(json_path,'r'))['order'][0]
    best_plddt = pickle.load(open(os.path.join(work_dir,"result_{}.pkl".format(best_model)),'rb'))['plddt']
    
    return best_plddt


def read_plddt(best_plddt:np.ndarray, chain_CA_inds:Dict[str,List[int]]
               ) -> Dict[str,np.ndarray]:
    """
    Get the pLDDT scores per chain
    
    Args:
        best_plddt (np.ndarray): Array with the pLDDT scores for the best model
        chain_CA_inds (dict): Dictionary with the indices of the CA atoms for
                each chain. The keys are the chain ids and the values are lists
                of integers with the indices
    
    Returns:
        Dict[str,np.ndarray]: Dictionary with the pLDDT scores per chain. The
                keys are the chain ids and the values are numpy arrays with
                the pLDDT scores
    """
    chain_names = chain_CA_inds.keys()
    chain_lengths = dict()
    for name in chain_names:
        curr_len = len(chain_CA_inds[name])
        chain_lengths[name] = curr_len
    
    plddt_per_chain = dict()
    curr_len = 0
    for k,v in chain_lengths.items():
        curr_plddt = best_plddt[curr_len:curr_len+v]
        plddt_per_chain[k] = curr_plddt
        curr_len += v 
    return plddt_per_chain


def score_complex(path_coords:Dict[str,List[List[float]]],
                  path_CB_inds:Dict[str,List[int]],
                  path_plddt:Dict[str,np.ndarray]) -> Tuple[float,int]:
    """
    Score all interfaces in the current complex

    Modified from the score_complex() function in MoLPC repo: 
    https://gitlab.com/patrickbryant1/molpc/-/blob/main/src/complex_assembly/score_entire_complex.py#L106-154
    
    Args:
        path_coords (dict): Dictionary with the coordinates for each chain. The
                keys are the chain ids and the values are lists of lists with
                the x,y,z coordinates for each atom
        path_CB_inds (dict): Dictionary with the indices of the CB atoms for
                each chain. The keys are the chain ids and the values are lists
                of integers with the indices
        path_plddt (dict): Dictionary with the pLDDT scores per chain. The
                keys are the chain ids and the values are numpy arrays with
                the pLDDT scores
    
    Returns:
        Tuple[float,int]: Tuple with the complex score and the number of chains
    """

    chains = [*path_coords.keys()]
    chain_inds = np.arange(len(chains))
    complex_score = 0
    #Get interfaces per chain
    for i in chain_inds:
        chain_i = chains[i]
        chain_coords = np.array(path_coords[chain_i])
        chain_CB_inds = path_CB_inds[chain_i]
        l1 = len(chain_CB_inds)
        chain_CB_coords = chain_coords[chain_CB_inds]
        chain_plddt = path_plddt[chain_i]
 
        for int_i in np.setdiff1d(chain_inds, i):
            int_chain = chains[int_i]
            int_chain_CB_coords = np.array(path_coords[int_chain])[path_CB_inds[int_chain]]
            int_chain_plddt = path_plddt[int_chain]
            #Calc 2-norm
            mat = np.append(chain_CB_coords,int_chain_CB_coords,axis=0)
            a_min_b = mat[:,np.newaxis,:] -mat[np.newaxis,:,:]
            dists = np.sqrt(np.sum(a_min_b.T ** 2, axis=0)).T
            contact_dists = dists[:l1,l1:]
            contacts = np.argwhere(contact_dists<=8)
            #The first axis contains the contacts from chain 1
            #The second the contacts from chain 2
            if contacts.shape[0]>0:
                av_if_plDDT = np.concatenate((chain_plddt[contacts[:,0]],
                                              int_chain_plddt[contacts[:,1]])).mean()
                complex_score += np.log10(contacts.shape[0]+1)*av_if_plDDT

    return complex_score, len(chains)


def calculate_mpDockQ(complex_score:float) -> float:
    """
    A function that returns a complex's mpDockQ score after 
    calculating complex_score
    
    Args:
        complex_score (float): The complex score calculated by score_complex()
    
    Returns:
        float: The mpDockQ score
    """
    L = 0.827
    x_0 = 261.398
    k = 0.036
    b = 0.221
    mpdockq = L/(1+math.exp(-1*k*(complex_score-x_0))) + b
    return (mpdockq, np.nan, np.nan) # To match the output of the pdockq function


def calc_pdockq(chain_coords:Dict[str,List[List[float]]],
                chain_plddt:Dict[str,np.ndarray],
                t:int,
                res_indices:Dict[str,np.ndarray]) -> Tuple[float,float,int]:
    """
    Calculate the pDockQ scores
    pdockQ = L / (1 + np.exp(-k*(x-x0)))+b
    L= 0.724 x0= 152.611 k= 0.052 and b= 0.018

    Modified from the calc_pdockq() from FoldDock repo: 
    https://gitlab.com/ElofssonLab/FoldDock/-/blob/main/src/pdockq.py#L62
    
    Args:
        chain_coords (dict): Dictionary with the coordinates for each chain. The
                keys are the chain ids and the values are lists of lists with
                the x,y,z coordinates for each atom
        chain_plddt (dict): Dictionary with the pLDDT scores per chain. The
                keys are the chain ids and the values are numpy arrays with
                the pLDDT scores
        t (int): Distance threshold for the interface
        res_indices (dict): Dictionary with the residue numbers for each chain.
                The keys are the chain ids and the values are numpy arrays with
                the residue index for each atom
    
    Returns:
        Tuple[float,float,int]: Tuple with the pDockQ score, the average pLDDT
                score for the interface and the number of interface contacts
    """

    #Get coords and plddt per chain
    ch1, ch2 = [*chain_coords.keys()]
    coords1, coords2 = chain_coords[ch1], chain_coords[ch2]
    plddt1, plddt2 = chain_plddt[ch1], chain_plddt[ch2]

    #Calc 2-norm
    mat = np.append(coords1, coords2,axis=0)
    a_min_b = mat[:,np.newaxis,:] -mat[np.newaxis,:,:]
    dists = np.sqrt(np.sum(a_min_b.T ** 2, axis=0)).T
    l1 = len(coords1)
    contact_dists = dists[:l1,l1:] #upper triangular --> first dim = chain 1
    contacts = np.argwhere(contact_dists<=t)

    if contacts.shape[0]<1:
        pdockq=0
    else:
        #Get the average interface plDDT
        residue_indices_A = np.unique(res_indices[ch1][contacts[:,0]])
        residue_indices_B = np.unique(res_indices[ch2][contacts[:,1]])
        avg_if_plddt = np.average(np.concatenate([plddt1[residue_indices_A], 
                                                  plddt2[residue_indices_B]]))
        #Get the number of interface contacts
        # Convert the contacts to residue indexes, and remove duplicates
        n_if_contacts = pd.DataFrame(
            [[res_indices['A'][a], res_indices['B'][b]] for a,b in contacts]
            ).drop_duplicates().shape[0]
        x = avg_if_plddt*np.log10(n_if_contacts)
        pdockq = 0.724 / (1 + np.exp(-0.052*(x-152.611)))+0.018

    return pdockq, avg_if_plddt, n_if_contacts


def obtain_mpdockq(work_dir:str) -> Union[Tuple[float,float,int], str]:
    """
    Returns mpDockQ if more than two chains otherwise return pDockQ

    Args:
        work_dir (str): Directory with the results from AlphaFold

    Returns:
        Union[Tuple[float,float,int], str]: Quality scores for the complex
    """
    pdb_path = os.path.join(work_dir,'ranked_0.pdb')
    pdb_chains, chain_coords, chain_CA_inds, chain_CB_inds, res_indices = read_pdb(pdb_path)
    best_plddt = get_best_plddt(work_dir)
    plddt_per_chain = read_plddt(best_plddt,chain_CA_inds)
    complex_score,num_chains = score_complex(chain_coords,chain_CB_inds,plddt_per_chain)
    if complex_score is not None and num_chains>2:
        mpDockq_or_pdockq = calculate_mpDockQ(complex_score)
    elif complex_score is not None and num_chains==2:
        mpDockq_or_pdockq = calc_pdockq(chain_coords,plddt_per_chain,8,res_indices)
    else:
        mpDockq_or_pdockq = "None"
    return mpDockq_or_pdockq


# test case
if __name__ == '__main__':
    pdbdir = ("/Volumes/weka_projects/c2217/alphafold_strube/runs/denovo_screen/"
              "dn47_top_full/Osjap04g06970.1-1_A0A0N7KMK5-1")
    
    scores = obtain_mpdockq(pdbdir)
    print(scores)