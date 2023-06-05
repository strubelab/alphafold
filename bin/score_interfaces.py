'''
Script to relax a protein structure. Have to use it with the python installed
in the alphafold module
'''
import os
import sys
import pickle
from pathlib import Path

sys.path.append(os.fspath(Path(__file__).parent.parent))

import argparse

from absl import logging
from alphafold_ibex.scoring_functions import obtain_mpdockq

logging.set_verbosity(logging.INFO)


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
    
    def validate_dir(finput:str) -> Path:
        """
        Validate that the input is an existing directory

        Args:
            input (str): Input file

        Returns:
            Path
        """
        if not Path(finput).is_dir():
            raise ValueError

        return finput

    parser = argparse.ArgumentParser(description=('Calculate the interface '
                'scores (pDockQ/mpDockQ, number of interface residues and '
                'average interface pLDDT) for a given complex.'))
    
    parser.add_argument('-i', '--input', type=validate_dir, required=True,
                        help='Directory with the AlphaFold models.')
    
    return parser.parse_args(args)


if __name__ == '__main__':
    
    args = parsing()

    logging.info(f'Calculating interface scores for {args.input}...')
    
    pdockq, avg_if_plddt, n_if_contacts = obtain_mpdockq(args.input)
    
    logging.info(f'pDockQ: {pdockq:.3f}')
    logging.info(f'Average interface pLDDT: {avg_if_plddt:.3f}')
    logging.info(f'Number of interface residues: {n_if_contacts}')
    
    # Save to pickle
    with open(os.path.join(args.input, 'interface_scores.pkl'), 'wb') as f:
        pickle.dump({'pDockQ': pdockq,
                     'avg_if_plddt': avg_if_plddt,
                     'n_if_contacts': n_if_contacts}, f)
    
    logging.info('Done!')