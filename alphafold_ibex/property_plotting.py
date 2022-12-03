import matplotlib
matplotlib.use('Agg')

import numpy as np
import matplotlib.pyplot as plt

from pathlib import Path

from string import ascii_uppercase,ascii_lowercase

alphabet_list = list(ascii_uppercase+ascii_lowercase)

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
        plt.title(f"rank_{n}")
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
        plt.title(f"rank_{n}")
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
        plt.title(f"rank_{n}")
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
        plt.plot(plddt,label=f"rank_{n} ({plddt.mean():.2f})")
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

