#!/usr/bin/env python3

"""
Run this script to process the alphafold results


Usage:

./process_results.py [fasta_file] [af_outputs]

fasta_file: path to the same fasta file that was provided to alphafold
af_outputs: path to the directory with the outputs from alphafold (the one that
            contains the pdb files)

For help:

./process_results.py --help
"""


import sys
import numpy as np
import pickle
import re
import jax
import argparse
import logging

from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib.patheffects as path_effects
import matplotlib

from matplotlib import collections as mcoll
from matplotlib.figure import Figure

from Bio import SeqIO

from string import ascii_uppercase,ascii_lowercase

alphabet_list = list(ascii_uppercase+ascii_lowercase)



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
    out = {
        "plddt": plddt,
        "pLDDT": plddt.mean(),
        "dists": dist_mtx,
        "adj": contact_mtx,
        "pae": prediction_result['predicted_aligned_error'],
        "pTMscore": prediction_result['ptm']
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


######## PROPERTY PLOTTING ########

def plot_ticks(chain_breaks):
    Ln = sum(chain_breaks)
    L_prev = 0
    for L_i in chain_breaks[:-1]:
        L = L_prev + L_i
        L_prev += L_i
        plt.plot([0,Ln],[L,L],color="black")
        plt.plot([L,L],[0,Ln],color="black")
    ticks = np.cumsum([0]+chain_breaks)
    ticks = (ticks[1:] + ticks[:-1])/2
    plt.yticks(ticks,alphabet_list[:len(ticks)])
  

def plot_paes(paes, chain_breaks=None, dpi=100, fig=True, savefile:Path=None):
    num_models = len(paes)
    if fig: plt.figure(figsize=(3*num_models,2), dpi=dpi)
    for n,pae in enumerate(paes):
        plt.subplot(1,num_models,n+1)
        plt.title(f"rank_{n+1}")
        Ln = pae.shape[0]
        plt.imshow(pae,cmap="bwr",vmin=0,vmax=30,extent=(0, Ln, Ln, 0))
        if chain_breaks is not None and len(chain_breaks) > 1:
            plot_ticks(chain_breaks)
        plt.colorbar()

    if savefile:
        plt.savefig(savefile, dpi=dpi)
        plt.close()

def plot_adjs(adjs, chain_breaks=None, dpi=100, fig=True, savefile:Path=None):
    num_models = len(adjs)
    if fig: plt.figure(figsize=(3*num_models,2), dpi=dpi)
    for n,adj in enumerate(adjs):
        plt.subplot(1,num_models,n+1)
        plt.title(f"rank_{n+1}")
        Ln = adj.shape[0]
        plt.imshow(adj,cmap="binary",vmin=0,vmax=1,extent=(0, Ln, Ln, 0))
        if chain_breaks is not None and len(chain_breaks) > 1:
            plot_ticks(chain_breaks)
        plt.colorbar()
    
    if savefile:
        plt.savefig(savefile, dpi=dpi)
        plt.close()


def plot_dists(dists, chain_breaks=None, dpi=100, fig=True, savefile:Path=None):
    num_models = len(dists)
    if fig: plt.figure(figsize=(3*num_models,2), dpi=dpi)
    for n,dist in enumerate(dists):
        plt.subplot(1,num_models,n+1)
        plt.title(f"rank_{n+1}")
        Ln = dist.shape[0]
        plt.imshow(dist,extent=(0, Ln, Ln, 0))
        if chain_breaks is not None and len(chain_breaks) > 1:
            plot_ticks(chain_breaks)
        plt.colorbar()

    if savefile:
        plt.savefig(savefile, dpi=dpi)
        plt.close()


def plot_plddts(plddts, chain_breaks=None, dpi=100, fig=True, savefile:Path=None):
    if fig: plt.figure(figsize=(8,5),dpi=dpi)
    plt.title("Predicted lDDT per position")
    for n,plddt in enumerate(plddts):
        plt.plot(plddt,label=f"rank_{n+1} ({plddt.mean():.2f})")
    if chain_breaks is not None:
        L_prev = 0
        for L_i in chain_breaks[:-1]:
          L = L_prev + L_i
          L_prev += L_i
          plt.plot([L,L],[0,100],color="black")
    
    plt.legend()
    plt.ylim(0,100)
    plt.ylabel("Predicted lDDT")
    plt.xlabel("Positions")

    if savefile:
        plt.savefig(savefile, dpi=dpi)
        plt.close()



######## STRUCTURE PLOTTING ########


pymol_color_list = ["#33ff33","#00ffff","#ff33cc","#ffff00","#ff9999","#e5e5e5","#7f7fff","#ff7f00",
                    "#7fff7f","#199999","#ff007f","#ffdd5e","#8c3f99","#b2b2b2","#007fff","#c4b200",
                    "#8cb266","#00bfbf","#b27f7f","#fcd1a5","#ff7f7f","#ffbfdd","#7fffff","#ffff7f",
                    "#00ff7f","#337fcc","#d8337f","#bfff3f","#ff7fff","#d8d8ff","#3fffbf","#b78c4c",
                    "#339933","#66b2b2","#ba8c84","#84bf00","#b24c66","#7f7f7f","#3f3fa5","#a5512b"]

pymol_cmap = matplotlib.colors.ListedColormap(pymol_color_list)


def kabsch(a, b, weights=None, return_v=False):
    a = np.asarray(a)
    b = np.asarray(b)
    if weights is None: weights = np.ones(len(b))
    else: weights = np.asarray(weights)
    B = np.einsum('ji,jk->ik', weights[:, None] * a, b)
    u, s, vh = np.linalg.svd(B)
    if np.linalg.det(u @ vh) < 0: u[:, -1] = -u[:, -1]
    if return_v: return u
    else: return u @ vh

def plot_pseudo_3D(xyz, c=None, ax=None, chainbreak=5,
                                     cmap="gist_rainbow", line_w=2.0,
                                     cmin=None, cmax=None, zmin=None, zmax=None):

    def rescale(a,amin=None,amax=None):
        a = np.copy(a)
        if amin is None: amin = a.min()
        if amax is None: amax = a.max()
        a[a < amin] = amin
        a[a > amax] = amax
        return (a - amin)/(amax - amin)

    # make segments
    xyz = np.asarray(xyz)
    seg = np.concatenate([xyz[:-1,None,:],xyz[1:,None,:]],axis=-2)
    seg_xy = seg[...,:2]
    seg_z = seg[...,2].mean(-1)
    ord = seg_z.argsort()

    # set colors
    if c is None: c = np.arange(len(seg))[::-1]
    else: c = (c[1:] + c[:-1])/2
    c = rescale(c,cmin,cmax)  

    if isinstance(cmap, str):
        if cmap == "gist_rainbow": c *= 0.75
        colors = matplotlib.cm.get_cmap(cmap)(c)
    else:
        colors = cmap(c)
    
    if chainbreak is not None:
        dist = np.linalg.norm(xyz[:-1] - xyz[1:], axis=-1)
        colors[...,3] = (dist < chainbreak).astype(np.float)

    # add shade/tint based on z-dimension
    z = rescale(seg_z,zmin,zmax)[:,None]
    tint, shade = z/3, (z+2)/3
    colors[:,:3] = colors[:,:3] + (1 - colors[:,:3]) * tint
    colors[:,:3] = colors[:,:3] * shade

    set_lim = False
    if ax is None:
        fig, ax = plt.subplots()
        fig.set_figwidth(5)
        fig.set_figheight(5)
        set_lim = True
    else:
        fig = ax.get_figure()
        if ax.get_xlim() == (0,1):
            set_lim = True
            
    if set_lim:
        xy_min = xyz[:,:2].min() - line_w
        xy_max = xyz[:,:2].max() + line_w
        ax.set_xlim(xy_min,xy_max)
        ax.set_ylim(xy_min,xy_max)

    ax.set_aspect('equal')
        
    # determine linewidths
    width = fig.bbox_inches.width * ax.get_position().width
    linewidths = line_w * 72 * width / np.diff(ax.get_xlim())

    lines = mcoll.LineCollection(seg_xy[ord], colors=colors[ord], linewidths=linewidths,
                                                             path_effects=[path_effects.Stroke(capstyle="round")])
    
    return ax.add_collection(lines)


def add_text(text, ax):
    return plt.text(0.5, 1.01, text, horizontalalignment='center',
                                    verticalalignment='bottom', transform=ax.transAxes)


def plot_protein(protein:dict, Ls:list=None, dpi:int=100,
    best_view:bool=True, line_w=2.0) -> Figure:
    """
    Plot the protein in 2D

    Args:
        protein (dict, optional):
            Dictionary with the prediction results of a single protein.
            Defaults to None.
        Ls (list, optional):
            List with the amino acid indexes of chain breaks with respect to
            the full sequence. Defaults to None.
        dpi (int, optional):
            Dots per inch in the figure. Defaults to 100.
        best_view (bool, optional):
            Whether to calculate the best view/orientation for the molecule?.
            Defaults to True.
        line_w (float, optional):
            Line width?. Defaults to 2.0.

    Returns:
        matplotlib.figure.Figure
    """
    
    pos = np.asarray(protein['structure_module']['final_atom_positions'][:,1,:])
    plddt = np.asarray(protein['plddt'])

    # get best view
    if best_view:
        weights = plddt/100
        pos = pos - (pos * weights[:,None]).sum(0,keepdims=True) / weights.sum()
        pos = pos @ kabsch(pos, pos, weights, return_v=True)

    fig, (ax1, ax2) = plt.subplots(1,2)
    fig.set_figwidth(6); fig.set_figheight(3)
    ax = [ax1, ax2]
        
    fig.set_dpi(dpi)
    fig.subplots_adjust(top = 0.9, bottom = 0.1, right = 1, left = 0, hspace = 0, wspace = 0)

    xy_min = pos[...,:2].min() - line_w
    xy_max = pos[...,:2].max() + line_w
    for a in ax:
        a.set_xlim(xy_min, xy_max)
        a.set_ylim(xy_min, xy_max)
        a.axis(False)

    if Ls is None or len(Ls) == 1:
        # color N->C
        c = np.arange(len(pos))[::-1]
        plot_pseudo_3D(pos,  line_w=line_w, ax=ax1)
        add_text("colored by N→C", ax1)
    else:
        # color by chain
        c = np.concatenate([[n]*L for n,L in enumerate(Ls)])
        if len(Ls) > 40:   plot_pseudo_3D(pos, c=c, line_w=line_w, ax=ax1)
        else:              plot_pseudo_3D(pos, c=c, cmap=pymol_cmap, cmin=0, cmax=39, line_w=line_w, ax=ax1)
        add_text("colored by chain", ax1)
        
    if plddt is not None:
        # color by pLDDT
        plot_pseudo_3D(pos, c=plddt, cmin=50, cmax=90, line_w=line_w, ax=ax2)
        add_text("colored by pLDDT", ax2)

    return fig


########## PARSING COMMAND LINE INPUTS

def parsing(args: list=None) -> argparse.Namespace:
    """
    Creates the argument parser instance and applies it to the command line
    input

    Input
    -----
    args : list
        List of the arguments to be parsed (only to be used for testing). If
        none is provided, it is taken from sys.argv
    """

    def validate_input(input: str) -> Path:
        """
        Validate that input is an existing file or directory

        Args:
            input (str): input file or directory

        """
        inp = Path(input)
        if not inp.exists():
            raise ValueError

        return inp


    parser = argparse.ArgumentParser(description=('Program to process the '
                'outputs from Alphafold and return plots with the pLDDT, PAE, '
                'predicted contacts, predicted distances, and 2D images of '
                'protein structure colored by pLDDT.'))

    parser.add_argument("sequence", help=('Fasta file that was provided as '
        'input to Alphafold.'), type=validate_input)

    parser.add_argument("af_outputs", help=('Directory with the alphafold '
        'outputs, containing the pdb files.'), type=validate_input)

    return parser.parse_args(args)



if __name__ == "__main__":

    logging.basicConfig(format='%(levelname)s:%(message)s',
                level=logging.INFO)

    args = parsing()
    fasta_file = args.sequence
    af_outputs = args.af_outputs
    dpi=200

    af_outputs = af_outputs/fasta_file.stem

    sequences = list(SeqIO.parse(fasta_file, 'fasta'))

    chain_breaks, homooligomers, unique_names = (define_homooligomers(sequences))

    # Read otuputs
    features_files = list(af_outputs.glob('result_model_*'))

    prediction_results, outs, model_rank = (process_outputs(features_files))

    # Plot properties
    logging.info(f'Making directory for plots in {af_outputs}/plots')
    plots_dir = af_outputs / 'plots'
    plots_dir.mkdir()

    logging.info('Plotting PAEs...')
    plot_paes([outs[k]["pae"] for k in model_rank],
                chain_breaks=chain_breaks, dpi=dpi,
                savefile=plots_dir/'pae.png')

    logging.info('Plotting predicted contacts...')
    plot_adjs([outs[k]["adj"] for k in model_rank],
                chain_breaks=chain_breaks, dpi=dpi,
                savefile=plots_dir/'predicted_contacts.png')

    logging.info('Plotting predicted distances...')
    plot_dists([outs[k]["dists"] for k in model_rank],
                chain_breaks=chain_breaks, dpi=dpi,
                savefile=plots_dir/'predicted_distances.png')

    logging.info('Plotting pLDDTs...')
    plot_plddts([outs[k]["plddt"] for k in model_rank],
                chain_breaks=chain_breaks, dpi=dpi,
                savefile=plots_dir/'plddts.png')

    # Plot structures
    logging.info('Drawing proteins in 2D...')
    for i,name in enumerate(model_rank):
        plot_protein(prediction_results[name], chain_breaks, dpi=dpi)
        plt.suptitle(f'Rank {i+1}: {name}, '
                        f'pLDDT={outs[name]["pLDDT"]:.2f}, '
                        f'pTM={outs[name]["pTMscore"]:.2f}')

        plt.tight_layout()
        plt.savefig(plots_dir/f'rank_{i+1}_{name}.png', dpi=dpi)
        plt.close()
        
    logging.info('Done.')

