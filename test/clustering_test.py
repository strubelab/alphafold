import unittest
import os
from pathlib import Path
import sys
import pandas as pd

from unittest.mock import patch

sys.path.append(os.fspath(Path(__file__).parent.parent))

from bin.clustering import (get_topcluster_members, joint_cluster, merge_dict_values,
                            joint_clusters_df)

class GetTopClusterMembersTest(unittest.TestCase):
    """
    Class to test the get_topcluster_members function
    """
    
    def test_topstrclusters(self):
        
        strclusters_path = (Path(__file__).parent / \
                            'test_outputs/skp1_clusters/strclusters_unmerged.csv')
        
        strclusters = pd.read_csv(strclusters_path)
        
        top_clusters_members = get_topcluster_members(strclusters)
        
        self.assertEqual(len(top_clusters_members), 73)
        
        test_set = {'A0A0P0X4R4', 'Q2R0K5', 'Q0J0N5', 'Q7XLA5', 'A0A0P0XGK8',
                    'A0A0P0WF19', 'A0A0P0YB14', 'A0A0P0XQK2'}
        self.assertEqual(top_clusters_members['A0A0P0XGK8.pdb_B'], test_set)
    
    def test_topseqclusters(self):
        
        seqclusters_path = (Path(__file__).parent / \
                            'test_outputs/skp1_clusters/seqclusters_unmerged.csv')
        
        seqclusters = pd.read_csv(seqclusters_path)
        
        top_clusters_members = get_topcluster_members(seqclusters)
        
        self.assertEqual(len(top_clusters_members), 127)
        
        test_set = {'Q5VR67', 'Q9LG67'}
        self.assertEqual(top_clusters_members['Q5VR67'], test_set)
    


class JointClusterTest(unittest.TestCase):
    """
    Class to test the joint_cluster function
    """

    def test_non_overlapping_clusters1(self):
        """
        No overlap. Return the same four clusters.
        """
        # Initialize input dataframes
        seqclusters = pd.DataFrame()
        strclusters = pd.DataFrame()
        
        # Assuming that the get_topcluster_members function works correctly,
        # the input data should look similar to this:
        seqclusters_members = {'seq1': {'A', 'B'}, 'seq2': {'C', 'D'}}
        strclusters_members = {'str1': {'E', 'F'}, 'str2': {'G', 'H'}}
        expected_result = {'seq1': {'A', 'B'},
                           'seq2': {'C', 'D'},
                           'str1': {'E', 'F'},
                           'str2': {'G', 'H'}}
        
        # Mock the get_topcluster_members function and give the expected
        # result as return value (side_effect)
        with patch('bin.clustering.get_topcluster_members',
                   side_effect=[seqclusters_members, strclusters_members]):
            result = joint_cluster(seqclusters, strclusters)
        self.assertEqual(result, expected_result)
        
    def test_non_overlapping_clusters2(self):
        """
        No overlap. Return the same clusters.
        """
        # Initialize input dataframes
        seqclusters = pd.DataFrame()
        strclusters = pd.DataFrame()
        
        # Assuming that the get_topcluster_members function works correctly,
        # the input data should look similar to this:
        seqclusters_members = {'seq1': {'A', 'B', 'C'}, 'seq2': {'D', 'E', 'F'}}
        strclusters_members = {'str1': {'G', 'H', 'I'}, 'str2': {'J', 'K', 'L'}}
        expected_result = {'seq1': {'A', 'B', 'C'},
                           'seq2': {'D', 'E', 'F'},
                           'str1': {'G', 'H', 'I'},
                           'str2': {'J', 'K', 'L'}}
        
        # Mock the get_topcluster_members function and give the expected
        # result as return value (side_effect)
        with patch('bin.clustering.get_topcluster_members',
                   side_effect=[seqclusters_members, strclusters_members]):
            result = joint_cluster(seqclusters, strclusters)
        self.assertEqual(result, expected_result)
    
    def test_overlapping_clusters1(self):
        """
        Return one big cluster.
        """
        # Initialize input dataframes
        seqclusters = pd.DataFrame()
        strclusters = pd.DataFrame()
        
        # Assuming that the get_topcluster_members function works correctly,
        # the input data should look similar to this:
        seqclusters_members = {'seq1': {'A', 'B'}, 'seq2': {'C', 'D'}}
        strclusters_members = {'str1': {'B', 'C'}, 'str2': {'D', 'E'}}
        expected_result = {'str2': {'A', 'B', 'C', 'D', 'E'}}
        
        # Mock the get_topcluster_members function and give the expected
        # result as return value (side_effect)
        with patch('bin.clustering.get_topcluster_members',
                   side_effect=[seqclusters_members, strclusters_members]):
            result = joint_cluster(seqclusters, strclusters)
        self.assertEqual(result, expected_result)
    
    def test_overlapping_clusters2(self):
        """
        Return two clusters.
        """
        # Initialize input dataframes
        seqclusters = pd.DataFrame()
        strclusters = pd.DataFrame()
        
        # Assuming that the get_topcluster_members function works correctly,
        # the input data should look similar to this:
        seqclusters_members = {'seq1': {'A', 'B'}, 'seq2': {'D', 'E'}}
        strclusters_members = {'str1': {'B', 'C'}, 'str2': {'D', 'F'}}
        expected_result = {'str1': {'A', 'B', 'C'},
                           'str2': {'D', 'E', 'F'}}
        
        # Mock the get_topcluster_members function and give the expected
        # result as return value (side_effect)
        with patch('bin.clustering.get_topcluster_members',
                   side_effect=[seqclusters_members, strclusters_members]):
            result = joint_cluster(seqclusters, strclusters)
        self.assertEqual(result, expected_result)
        
        
    def test_overlapping_clusters3(self):
        """
        Return three clusters, with one sequence cluster intact.
        """
        # Initialize input dataframes
        seqclusters = pd.DataFrame()
        strclusters = pd.DataFrame()
        
        # Assuming that the get_topcluster_members function works correctly,
        # the input data should look similar to this:
        seqclusters_members = {'seq1': {'A', 'B'}, 'seq2': {'D', 'E'}, 'seq3': {'F', 'G'}}
        strclusters_members = {'str1': {'B', 'C'}, 'str2': {'D', 'H'}}
        expected_result = {'str1': {'A', 'B', 'C'},
                           'str2': {'D', 'E', 'H'},
                           'seq3': {'F', 'G'}}
        
        # Mock the get_topcluster_members function and give the expected
        # result as return value (side_effect)
        with patch('bin.clustering.get_topcluster_members',
                   side_effect=[seqclusters_members, strclusters_members]):
            result = joint_cluster(seqclusters, strclusters)
        self.assertEqual(result, expected_result)


    def test_overlapping_clusters4(self):
        """
        Return four clusters, with one sequence and one structure cluster intact.
        """
        # Initialize input dataframes
        seqclusters = pd.DataFrame()
        strclusters = pd.DataFrame()
        
        # Assuming that the get_topcluster_members function works correctly,
        # the input data should look similar to this:
        seqclusters_members = {'seq1': {'A', 'B'}, 'seq2': {'D', 'E'}, 'seq3': {'F', 'G'}}
        strclusters_members = {'str1': {'B', 'C'}, 'str2': {'D', 'H'}, 'str3': {'I', 'J'}}
        expected_result = {'str1': {'A', 'B', 'C'},
                           'str2': {'D', 'E', 'H'},
                           'str3': {'I', 'J'},
                           'seq3': {'F', 'G'}}
        
        # Mock the get_topcluster_members function and give the expected
        # result as return value (side_effect)
        with patch('bin.clustering.get_topcluster_members',
                   side_effect=[seqclusters_members, strclusters_members]):
            result = joint_cluster(seqclusters, strclusters)
        self.assertEqual(result, expected_result)
    
    def test_overlapping_clusters5(self):
        """
        Return merged clusters.
        """
        # Initialize input dataframes
        seqclusters = pd.DataFrame()
        strclusters = pd.DataFrame()
        
        # Assuming that the get_topcluster_members function works correctly,
        # the input data should look similar to this:
        seqclusters_members = {'seq1': {'A', 'B', 'C', 'D'},
                               'seq2': {'E', 'F', 'G'},
                               'seq3': {'H', 'I'},
                               'seq4': {'J', 'K', 'L'}}
        strclusters_members = {'str1': {'B', 'C', 'E', 'L'},
                               'str2': {'D', 'H'},
                               'str3': {'I', 'M'},
                               'str4': {'N', 'O', 'P'}}
        
        expected_result = {'str1': {'A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J', 'K', 'L', 'M'},
                           'str4': {'N', 'O', 'P'}}
        
        # Mock the get_topcluster_members function and give the expected
        # result as return value (side_effect)
        with patch('bin.clustering.get_topcluster_members',
                   side_effect=[seqclusters_members, strclusters_members]):
            result = joint_cluster(seqclusters, strclusters)
        self.assertEqual(result, expected_result)


    def test_overlapping_clusters6(self):
        """
        Return merged clusters.
        """
        # Initialize input dataframes
        seqclusters = pd.DataFrame()
        strclusters = pd.DataFrame()
        
        # Assuming that the get_topcluster_members function works correctly,
        # the input data should look similar to this:
        seqclusters_members = {'seq1': {'A', 'B', 'C', 'D'},
                               'seq2': {'E', 'F', 'G'},
                               'seq3': {'H', 'I'},
                               'seq4': {'J', 'K', 'L'},
                               'seq5': {'Q', 'R', 'S', 'T'}}
        strclusters_members = {'str1': {'B', 'C', 'E', 'L'},
                               'str2': {'D', 'H'},
                               'str3': {'I', 'M'},
                               'str4': {'N', 'O', 'P'}}
        
        expected_result = {'str1': {'A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J', 'K', 'L', 'M'},
                           'str4': {'N', 'O', 'P'},
                           'seq5': {'Q', 'R', 'S', 'T'}}
        
        # Mock the get_topcluster_members function and give the expected
        # result as return value (side_effect)
        with patch('bin.clustering.get_topcluster_members',
                   side_effect=[seqclusters_members, strclusters_members]):
            result = joint_cluster(seqclusters, strclusters)
        self.assertEqual(result, expected_result)


class MergeDictValues(unittest.TestCase):
    """
    Class to test the merge_dict_values function
    """
    
    def test_no_overlap(self):
        """
        Test a dictionary with no overlapping values
        """
        
        test_dict = {'A': {'a', 'b', 'c'},
                     'B': {'d', 'e', 'f'}}
        
        merged_dict = merge_dict_values(test_dict)
        
        self.assertEqual(len(merged_dict), 2)
        
        self.assertEqual(merged_dict['A'], {'a', 'b', 'c'})
        self.assertEqual(merged_dict['B'], {'d', 'e', 'f'})


    def test_overlap(self):
        """
        Test a dictionary with overlapping values
        """
        
        test_dict = {'A': {'a', 'b', 'c'},
                     'B': {'b', 'c', 'd'}}
        
        merged_dict = merge_dict_values(test_dict)
        
        self.assertEqual(len(merged_dict), 1)
        
        self.assertEqual(merged_dict['B'], {'a', 'b', 'c', 'd'})
    
    def test_overlap2(self):
        """
        Test a dictionary with overlapping values
        """
        
        test_dict = {'A': {'a', 'b', 'c'},
                     'B': {'b', 'c', 'd'},
                     'C': {'a', 'b', 'e', 'f'},
                     'D': {'g', 'h', 'i'},
                     'E': {'i', 'k', 'l'},
                     'F': {'m', 'n', 'o'}}
        
        merged_dict = merge_dict_values(test_dict)
        
        self.assertEqual(len(merged_dict), 3)
        self.assertEqual(merged_dict['C'], {'a', 'b', 'c', 'd', 'e', 'f'})
        self.assertEqual(merged_dict['E'], {'g', 'h', 'i', 'k', 'l'})
        self.assertEqual(merged_dict['F'], {'m', 'n', 'o'})
        
    
    def test_overlap3(self):
        """
        Test a dictionary with overlapping values
        """
        
        test_dict = {'str1': {'A', 'B', 'C', 'D', 'E', 'F', 'G', 'J', 'K', 'L'},
                     'str2': {'A', 'B', 'C', 'D', 'H', 'I'},
                     'str3': {'H', 'I', 'M'},
                     'str4': {'N', 'O', 'P'}}
        
        merged_dict = merge_dict_values(test_dict)
        
        self.assertEqual(len(merged_dict), 2)
        self.assertEqual(merged_dict['str1'],
                         {'A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J', 'K', 'L', 'M'})
        self.assertEqual(merged_dict['str4'], {'N', 'O', 'P'})


if __name__ == '__main__':
    unittest.main()