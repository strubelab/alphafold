import unittest

import os, sys, shutil
import tempfile
from pathlib import Path
import pickle

from Bio import SeqIO

sys.path.append(os.fspath(Path(__file__).parent.parent / 'bin'))

from make_features import check_missing_sequences

class CheckMissingSequencesTest(unittest.TestCase):
    """
    Class for testing the function to check missing sequences
    """

    def test_missing_sequences(self) -> None:
        """
        Test that the correct number of sequences is identified
        """
        out_dir = Path(__file__).parent / 'test_outputs' / 'features_test'
        sequences = list(SeqIO.parse(Path(__file__).parent / 
                                     'test_outputs' /
                                     'test_rice.fasta', 'fasta'))
        sequences = [[s] for s in sequences]
        
        missing_sequences = check_missing_sequences(out_dir, sequences)
        
        self.assertEqual(len(missing_sequences), 3)
        
        missing_ids = ['A0A0P0UZH5', 'A0A0P0V7D8', 'A0A0P0V7I2']
        test_missing_sequences = [s for s in sequences \
                                  if s[0].id.split('|')[1] in missing_ids]
        
        self.assertEqual(missing_sequences, test_missing_sequences)
        

if __name__ == '__main__':
    unittest.main()
