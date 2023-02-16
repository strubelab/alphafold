import unittest
import os, sys, shutil
from  pathlib import Path
from Bio import SeqIO
import tempfile
import configparser

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

        self.SeqRec = SeqIO.read(Path(__file__).parent /
                                 'test_outputs/sequence.fasta', 'fasta')
        self.tempdir = Path(tempfile.gettempdir(),
            self.__class__.__name__.lower())
        self.out_dir = self.tempdir
        self.db_preset = 'full_dbs'
        self.max_template_date = date.today().isoformat()
        self.recycles = 6
        self.run_relax = True
        self.multimer_predictions_per_model = 4
        self.use_precomputed_msas = True


    def test_command_monomer(self):
        """
        Test that the proper command is created in the __init__ method
        """
        self.fasta_path = (self.out_dir / f'{self.SeqRec.name}-1' /
                           f'{self.SeqRec.name}-1.fasta')

        self.models_to_relax = 'all'
        self.model_preset = 'monomer_ptm'
        self.alphafold = AlphaFold([self.SeqRec],
                            out_dir=self.out_dir, recycles=self.recycles,
                            keep_tempdir=True,
                            models_to_relax=self.models_to_relax,
                            use_precomputed_msas=self.use_precomputed_msas)

        args = (
            f'python3 '
            f'{self.ALPHAFOLD_SCRIPT} '
            f'--fasta_paths={self.fasta_path} '
            f'--data_dir={self.ALPHAFOLD_DATA} '
            f'--output_dir={self.out_dir} '
            f'--uniref90_database_path={self.ALPHAFOLD_DATA}/uniref90/'
                'uniref90.fasta '
            f'--mgnify_database_path={self.ALPHAFOLD_DATA}/mgnify/'
                'mgy_clusters_2022_05.fa '
            f'--bfd_database_path={self.ALPHAFOLD_DATA}/bfd/'
                'bfd_metaclust_clu_complete_id30_c90_final_seq.sorted_opt '
            f'--uniref30_database_path={self.ALPHAFOLD_DATA}/uniref30/'
                'UniRef30_2021_03 '
            f'--template_mmcif_dir={self.ALPHAFOLD_DATA}/pdb_mmcif/'
                'mmcif_files '
            f'--max_template_date={self.max_template_date} '
            f'--obsolete_pdbs_path={self.ALPHAFOLD_DATA}/pdb_mmcif/'
                'obsolete.dat '
            f'--db_preset={self.db_preset} '
            f'--model_preset={self.model_preset} '
            f'--use_precomputed_msas=true '
            f'--models_to_relax={self.models_to_relax} '
            f'--use_gpu_relax=true '
            f'--recycles={self.recycles} '
            f'--pdb70_database_path={self.ALPHAFOLD_DATA}/pdb70/pdb70 '
        ).split()

        self.assertEqual(self.alphafold.args, args)

    def test_command_multimer(self):
        """
        Test that the proper command is created in the __init__ method
        """
        self.fasta_path = (self.out_dir / f'{self.SeqRec.name}-3' /
                           f'{self.SeqRec.name}-3.fasta')

        self.models_to_relax = 'best'
        self.model_preset = 'multimer'
        self.multimer_predictions_per_model = 4
        self.alphafold = AlphaFold([self.SeqRec]*3,
                            out_dir=self.out_dir, recycles=self.recycles,
                            keep_tempdir=True,
                            use_precomputed_msas=self.use_precomputed_msas,
                            multimer_predictions_per_model=
                            self.multimer_predictions_per_model)

        args = (
            f'python3 '
            f'{self.ALPHAFOLD_SCRIPT} '
            f'--fasta_paths={self.fasta_path} '
            f'--data_dir={self.ALPHAFOLD_DATA} '
            f'--output_dir={self.out_dir} '
            f'--uniref90_database_path={self.ALPHAFOLD_DATA}/uniref90/'
                'uniref90.fasta '
            f'--mgnify_database_path={self.ALPHAFOLD_DATA}/mgnify/'
                'mgy_clusters_2022_05.fa '
            f'--bfd_database_path={self.ALPHAFOLD_DATA}/bfd/'
                'bfd_metaclust_clu_complete_id30_c90_final_seq.sorted_opt '
            f'--uniref30_database_path={self.ALPHAFOLD_DATA}/uniref30/'
                'UniRef30_2021_03 '
            f'--template_mmcif_dir={self.ALPHAFOLD_DATA}/pdb_mmcif/'
                'mmcif_files '
            f'--max_template_date={self.max_template_date} '
            f'--obsolete_pdbs_path={self.ALPHAFOLD_DATA}/pdb_mmcif/'
                'obsolete.dat '
            f'--db_preset={self.db_preset} '
            f'--model_preset={self.model_preset} '
            f'--use_precomputed_msas=true '
            f'--models_to_relax={self.models_to_relax} '
            f'--use_gpu_relax=true '
            f'--recycles={self.recycles} '
            f'--uniprot_database_path={self.ALPHAFOLD_DATA}/uniprot/'
                    'uniprot.fasta '
            f'--pdb_seqres_database_path={self.ALPHAFOLD_DATA}/pdb_seqres/'
                'pdb_seqres.txt'
            f'--num_multimer_predictions_per_model='
            f'{self.multimer_predictions_per_model} '
        ).split()

        self.assertEqual(self.alphafold.args, args)

    def test_creates_input_fasta(self):
        """
        Test that the prepare method creates the input fasta file
        """
        self.fasta_path = (self.out_dir / f'{self.SeqRec.name}-1' /
                           f'{self.SeqRec.name}-1.fasta')

        self.models_to_relax = 'best'
        self.model_preset = 'monomer_ptm'
        self.alphafold = AlphaFold([self.SeqRec],
                            out_dir=self.out_dir, recycles=self.recycles,
                            keep_tempdir=True,
                            models_to_relax=self.models_to_relax,
                            use_precomputed_msas=self.use_precomputed_msas)

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
