import unittest
import pandas as pd
import os, sys
from pathlib import Path
import re


sys.path.append(os.fspath(Path(__file__).parent.parent))

from alphafold_ibex.utils import (get_seeds, get_scores)

class CheckGetSeeds(unittest.TestCase):
    
    def test_get_seeds(self) -> None:
        """
        Test that the seeds are correctly extracted from the output text files
        """
        seeds_dir = Path(__file__).parent / 'test_outputs' / 'seeds'
        highest_scores = pd.read_pickle(seeds_dir / 'highest_scores.pkl')
        top_complexes = list(highest_scores.complex)
        top_ligands = list(highest_scores.complex.apply(
                   lambda x: re.search(r'_(\w+)-1$', x).group(1)))

        seeds = get_seeds(top_complexes, top_ligands, seeds_dir)
        
        test_seeds = {
            'Q6ZI90': 3547371928582286275,
            'Q8LNT5': 183306296791308092,
            'A0A0P0UXP2': 4113323630389952870,
            'Q5KQK4': 508900553639612678,
            'A0A0N7KQW9': 2836163092372925646
        }
        
        self.assertTrue(test_seeds == seeds)

class CheckGetScores(unittest.TestCase):
    
    def test_get_seeds(self) -> None:
        """
        Test that the scores are correctly extracted from the output text files
        """
        models_dir = Path(__file__).parent / 'test_outputs' / 'models'

        scores = get_scores(models_dir)
        
        test_scores = pd.read_pickle(models_dir / 'scores.pkl')
        
        self.assertTrue(scores.equals(test_scores))


if __name__ == '__main__':
    unittest.main()
