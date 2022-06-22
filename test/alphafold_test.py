import unittest
import os, sys, shutil
from  pathlib import Path
from Bio import SeqIO
import tempfile
import configparser
import pandas as pd

from datetime import date

sys.path.append(os.fspath(Path(__file__).parent.parent))

from alphafold_ibex.alphafold_wrapper import AlphaFold

class AlphafoldTest(unittest.TestCase):

    def setUp(self):
        '''Called for every method'''
        config = configparser.ConfigParser()
        config.read(Path(__file__).parent.parent/'alphafold_ibex/config.ini')
        self.ALPHAFOLD_DATA = Path(config['user.env']['AF_DATA'])
        self.ALPHAFOLD_SCRIPT = Path(__file__).parent.parent / 'run_alphafold.py'

        self.SeqRec = SeqIO.read(Path(__file__).parent / 'test_outputs/sequence.fasta', 'fasta')
        self.tempdir = Path(tempfile.gettempdir(),
            self.__class__.__name__.lower())
        self.out_dir = self.tempdir
        self.db_preset = 'full_dbs'
        self.model_preset = 'monomer_ptm'
        self.max_template_date = date.today().isoformat()
        self.recycles = 6
        self.run_relax = True
        self.use_gpu_relax = True
        self.multimer_predictions_per_model = 4

        self.fasta_path = self.out_dir / f'{self.SeqRec.name}-1' / f'{self.SeqRec.name}-1.fasta'

        self.alphafold = AlphaFold([self.SeqRec],
                            out_dir=self.out_dir, recycles=self.recycles,
                            keep_tempdir=True)

    def test_command_created(self):
        """
        Test that the proper command is created in the __init__ method
        """

        args = (
            f'python3 '
            f'{self.ALPHAFOLD_SCRIPT} '
            f'--data_dir={self.ALPHAFOLD_DATA} '
            f'--output_dir={(self.out_dir / f"{self.SeqRec.name}-1")} '
            f'--fasta_paths={self.fasta_path} '
            f'--db_preset={self.db_preset} '
            f'--bfd_database_path={self.ALPHAFOLD_DATA}/bfd/'
                'bfd_metaclust_clu_complete_id30_c90_final_seq.sorted_opt '
            f'--uniclust30_database_path={self.ALPHAFOLD_DATA}/uniclust30/'
                'uniclust30_2018_08/uniclust30_2018_08 '
            f'--uniref90_database_path={self.ALPHAFOLD_DATA}/uniref90/'
                'uniref90.fasta '
            f'--mgnify_database_path={self.ALPHAFOLD_DATA}/mgnify/'
                'mgy_clusters_2018_12.fa '
            f'--template_mmcif_dir={self.ALPHAFOLD_DATA}/pdb_mmcif/'
                'mmcif_files '
            f'--model_preset={self.model_preset} '
            f'--obsolete_pdbs_path={self.ALPHAFOLD_DATA}/pdb_mmcif/'
                'obsolete.dat '
            f'--max_template_date={self.max_template_date} '
            f'--recycles={self.recycles} '
            f'--run_relax=true '
            f'--use_gpu_relax=true '
            f'--pdb70_database_path={self.ALPHAFOLD_DATA}/pdb70/pdb70 '
        ).split()

        self.assertEqual(self.alphafold.args, args)

        

    def test_creates_input_fasta(self):
        """
        Test that the prepare method creates the input fasta file
        """
        self.alphafold.prepare()

        self.assertTrue(self.fasta_path.is_file())

        seq = SeqIO.read(self.fasta_path, 'fasta')

        self.assertEqual(str(seq.seq), str(self.SeqRec.seq))


    def tearDown(self):
        '''If setUp() succeeded, tearDown() will be run whether the test method
        succeeded or not.'''
        
        if self.tempdir.exists():
            shutil.rmtree(self.tempdir)




if __name__ == '__main__':
    unittest.main()
