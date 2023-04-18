'''
Script to relax a protein structure. Have to use it with the python installed
in the alphafold module
'''
import os
import sys
from pathlib import Path

sys.path.append(os.fspath(Path(__file__).parent.parent))

import argparse

from absl import logging
from alphafold.relax import relax
from alphafold.common import protein

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
    
    def validate_file(finput:str) -> Path:
        """
        Validate that the input is an existing file

        Args:
            input (str): Input file

        Returns:
            Path
        """
        inp = Path(finput)
        if not inp.is_file():
            raise ValueError

        return inp

    parser = argparse.ArgumentParser(description='Relax a protein structure')
    
    parser.add_argument('-i', '--input', type=validate_file, required=True,
                        help='Input PDB file')
    
    parser.add_argument('-o', '--output', type=Path, required=True,
                        help='Output PDB file')
    
    return parser.parse_args(args)


if __name__ == '__main__':
    
    # Constant values from run_alphafold.py
    RELAX_MAX_ITERATIONS = 0
    RELAX_ENERGY_TOLERANCE = 2.39
    RELAX_STIFFNESS = 10.0
    RELAX_EXCLUDE_RESIDUES = []
    RELAX_MAX_OUTER_ITERATIONS = 3
    
    args = parsing()
    
    with open(args.input, 'r') as f:
        pdb_str = f.read()
    
    protein = protein.from_pdb_string(pdb_str)
    
    # Initialize the relaxer.
    amber_relaxer = relax.AmberRelaxation(
        max_iterations=RELAX_MAX_ITERATIONS,
        tolerance=RELAX_ENERGY_TOLERANCE,
        stiffness=RELAX_STIFFNESS,
        exclude_residues=RELAX_EXCLUDE_RESIDUES,
        max_outer_iterations=RELAX_MAX_OUTER_ITERATIONS,
        use_gpu=True)
    
    relaxed_pdb_str, _, violations = amber_relaxer.process(prot=protein)
    
    with open(args.output, 'w') as f:
        f.write(relaxed_pdb_str)
