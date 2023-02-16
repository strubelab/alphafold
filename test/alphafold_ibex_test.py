import unittest

import os, sys, shutil
import tempfile
from pathlib import Path
import configparser
import pickle

from Bio import SeqIO

sys.path.append(os.fspath(Path(__file__).parent.parent))

from alphafold_ibex.alphafold_ibex import AlphafoldIbex

class AlphafoldIbexTest(unittest.TestCase):
    """
    Class for testing the Ibex module
    """

    def setUp(self) -> None:
        """
        Example case where there the test input is a fasta file with many
        sequences
        """
        self.sequences = [SeqIO.read(
            Path(__file__).parent / 'test_outputs' / 'sequence.fasta',
            'fasta')] * 3
        self.tempdir = Path(tempfile.gettempdir(),
            self.__class__.__name__.lower())
        self.out_dir = self.tempdir
        self.out_ibex = self.out_dir / 'out_ibex'
        self.sequences_dir = self.out_dir / 'sequences'
        self.recycles=5
        self.models_to_relax = 'best'
        self.multimer_predictions_per_model = 4
        self.use_precomputed_msas = True
        self.conda_env = Path(__file__).parent.parent / 'env'

        self.jobname = 'AlphafoldIbex_unittest'

        self.script_file = self.out_ibex / 'script.sh'
        
        self.alphafoldibex = AlphafoldIbex(self.sequences, out_dir=self.out_dir,
            tempdir=self.tempdir, jobname=self.jobname,
            recycles=self.recycles,
            multimer_predictions_per_model=self.multimer_predictions_per_model,
            use_precomputed_msas=self.use_precomputed_msas,
            models_to_relax=self.models_to_relax)
    

    def test_sequences_written(self) -> None:
        """
        Test that the sequence files were creating according to the number of
        sequences per job
        """
        
        file_names = [self.sequences_dir / f'sequences{i}.pkl' \
                        for i in range(3)]
        self.alphafoldibex.prepare()

        for i, file in enumerate(file_names):
            self.assertTrue(file.exists())
            with open(file, 'rb') as f:
                seq = pickle.load(f)[0]
            self.assertEqual(str(self.sequences[i].seq), str(seq.seq))

    def test_script_made(self) -> None:
        """
        Test that the script for ibex is made correctly
        """
        self.alphafoldibex.prepare()

        self.python_file = (Path(__file__).parent.parent.resolve()
                                / 'alphafold_ibex'
                                / 'run_wrapper.py')
        
        self.python_command = (
            f'{self.conda_env}/bin/python {self.python_file} '
            '${seq_file} '
            f'{self.models_to_relax} {self.out_dir} {self.recycles} '
            f'{self.multimer_predictions_per_model} '
            f'{self.use_precomputed_msas} v100\n'
        )

        self.script = (
            '#!/bin/bash -l\n'
            '#SBATCH -N 1\n'
            f'#SBATCH --partition=batch\n'
            f'#SBATCH --job-name=AlphafoldIbex_unittest\n'
            f'#SBATCH --output={self.out_ibex}/%x.%j.out\n'
            f'#SBATCH --error={self.out_ibex}/%x.%j.out\n'
            f'#SBATCH --time=02:00:00\n'
            '#SBATCH --mem=64G\n'
            '#SBATCH --gres=gpu:1\n'
            f'#SBATCH --cpus-per-task=8\n'
            f'#SBATCH --constraint=[v100]\n'
            f'#SBATCH --array=0-2\n'
            '\n'
            f'module load alphafold/2.3.1/python3\n'
            'export CUDA_VISIBLE_DEVICES=0,1,2,3\n'
            'export TF_FORCE_UNIFIED_MEMORY=1\n'
            'export XLA_PYTHON_CLIENT_MEM_FRACTION=0.5\n'
            'export XLA_PYTHON_CLIENT_ALLOCATOR=platform\n'
            '\n'
            f'conda activate {self.conda_env}\n'
            '\n'
            f'seq_file="{self.sequences_dir.resolve()}/'
            'sequences${SLURM_ARRAY_TASK_ID}.pkl"\n'
            f'echo "{self.python_command}"\n'
            f'time {self.python_command}\n'
        )

        self.assertEqual(self.script, self.alphafoldibex.script)
        
        self.assertTrue(self.script_file.exists())
        with open(self.script_file, 'r') as f:
            script = f.read()
        self.assertEqual(self.script, script)

    def tearDown(self):
        '''If setUp() succeeded, tearDown() will be run whether the test method
        succeeded or not.'''
        if os.path.isdir(self.tempdir):
            shutil.rmtree(self.tempdir)

if __name__ == '__main__':
    unittest.main()
