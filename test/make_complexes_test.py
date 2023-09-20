import unittest

import os, sys
from pathlib import Path

from Bio import SeqIO

sys.path.append(os.fspath(Path(__file__).parent.parent))

from alphafold_ibex.utils import (validate_models, check_existing_features,
                                  check_missing_models, check_missing_homomers)

class ValidateModelsTest(unittest.TestCase):
    """
    Class for testing the function to check model names
    """

    def test_all(self) -> None:
        input_models = ['all']
        all_models = [
        'model_1_multimer_v3',
        'model_2_multimer_v3',
        'model_3_multimer_v3',
        'model_4_multimer_v3',
        'model_5_multimer_v3'
        ]
        
        validated_models = validate_models(input_models)
        
        self.assertEqual(validated_models, all_models)
    
    def test_two(self) -> None:
        input_models = ['two']
        
        validated_models = validate_models(input_models)
        
        self.assertEqual(validated_models,
                         ['model_3_multimer_v3', 'model_4_multimer_v3'])
    
    def test_one(self) -> None:
        input_models = ['model_1_multimer_v3']
        
        validated_models = validate_models(input_models)
        
        self.assertEqual(validated_models, ['model_1_multimer_v3'])
    
    def test_invalid(self) -> None:
        input_models = ['1', '2']
        
        self.assertRaises(ValueError, validate_models, input_models)


class CheckExistingFeaturesTest(unittest.TestCase):
    
    def setUp(self) -> None:
        self.features_dir = Path(__file__).parent / 'test_outputs' / 'features_test'
        self.sequences_dir = Path(__file__).parent / 'test_outputs'
        self.sequences = list(SeqIO.parse(
                            self.sequences_dir / 'test_rice.fasta', 'fasta'))
    
    def test_existing_features(self) -> None:
        """
        Test that the correct number of features is identified
        """
        bait = SeqIO.read(self.sequences_dir / 'bait.fasta', 'fasta')
        
        existing_features = check_existing_features(self.features_dir,
                                                    self.sequences, bait)
        
        self.assertEqual(len(existing_features), 7)
        
        test_existing = ['A0A0N7KDK5', 'A0A0N7KED1', 'A0A0N7KHT0', 'A0A0N7KJD4',
                        'A0A0N7KJF0', 'A0A0N7KKX8', 'A0A0N7KPF2']
        
        self.assertEqual(existing_features, test_existing)
    
    def test_nobait_features(self) -> None:
        """
        Test that the error is raised when the bait has no features
        """
        bait = SeqIO.read(self.sequences_dir / 'bait2.fasta', 'fasta')
        
        self.assertRaises(ValueError, check_existing_features, self.features_dir,
                          self.sequences, bait)
    
    def test_existing_features_baitnull(self) -> None:
        """
        Test that the correct number of features is identified, no bait provided
        """
        
        existing_features = check_existing_features(self.features_dir,
                                                    self.sequences)
        
        self.assertEqual(len(existing_features), 7)
        
        test_existing = ['A0A0N7KDK5', 'A0A0N7KED1', 'A0A0N7KHT0', 'A0A0N7KJD4',
                        'A0A0N7KJF0', 'A0A0N7KKX8', 'A0A0N7KPF2']
        
        self.assertEqual(existing_features, test_existing)


class CheckMissingModels(unittest.TestCase):
    
    def test_missing_models(self) -> None:
        """
        Test that the right missing models are identified
        """
        features_dir = Path(__file__).parent / 'test_outputs' / 'features_test'
        sequences_dir = Path(__file__).parent / 'test_outputs'
        sequences = list(SeqIO.parse(sequences_dir / 'test_rice.fasta', 'fasta'))
        bait = SeqIO.read(sequences_dir / 'bait.fasta', 'fasta')
        
        completed = check_existing_features(features_dir, sequences, bait)
        
        sequences_to_model = check_missing_models(completed, features_dir, bait,
                                                  sequences)
        
        self.assertTrue(len(sequences_to_model)==4)
        
        missing_models = ['A0A0N7KJD4', 'A0A0N7KJF0', 'A0A0N7KKX8', 'A0A0N7KPF2']
        
        sequences_to_model_ids = [s.id.split('|')[1] for s in sequences_to_model]
        
        self.assertEqual(set(sequences_to_model_ids), set(missing_models))


class CheckMissingHomomers(unittest.TestCase):
    
    def test_missing_homomers(self) -> None:
        """
        Test that the right missing homomers are identified
        """
        features_dir = Path(__file__).parent / 'test_outputs' / 'features_test'
        sequences_dir = Path(__file__).parent / 'test_outputs'
        sequences = list(SeqIO.parse(sequences_dir / 'test_rice.fasta', 'fasta'))
        
        completed = check_existing_features(features_dir, sequences)
        
        sequences_to_model = check_missing_homomers(completed, features_dir,
                                                  sequences, 2)
        
        self.assertTrue(len(sequences_to_model)==4)
        
        missing_models = ['A0A0N7KJD4', 'A0A0N7KJF0', 'A0A0N7KKX8', 'A0A0N7KPF2']
        
        sequences_to_model_ids = [s.id.split('|')[1] for s in sequences_to_model]
        
        self.assertEqual(set(sequences_to_model_ids), set(missing_models))


if __name__ == '__main__':
    unittest.main()
