"""
Wrapper for AlphaFold2
Starting from an amino acid sequence, you get the models and plots
related to MSA depth, pLDDT and PAE scores.
"""
import matplotlib
matplotlib.use('Agg')

from Bio import SeqIO
from pathlib import Path
from datetime import date
import configparser
import json
from matplotlib import pyplot as plt

from executor.executor import Executor

from utils import define_homooligomers
from utils import process_outputs

from property_plotting import plot_paes, plot_adjs, plot_dists, plot_plddts
from structure_plotting import plot_protein

import logging


class AlphaFold(Executor):
    """
    Class to execute AlphaFold on a single amino acid sequence, given
    as a SeqRecord
    """

    def __init__(self, sequences:list,
                 model_preset:str='monomer_ptm',
                 recycles:int=3,
                 db_preset:str='full_dbs',
                 models_to_relax:str='best',
                 use_gpu_relax:bool=True,
                 max_template_date:str=date.today().isoformat(),
                 out_dir:Path=None,
                 target_name:str=None,
                 multimer_predictions_per_model:int=5,
                 use_precomputed_msas:bool=False,
                 gpu_type:str='v100',
                 **kw):
        """
        Instantiate variables

        Args:
            sequences (list):
                List of sequences to analyze. If more than one is provided,
                `model_preset` must be set to `multimer`
            af_script (Path):
                Path to the `run_alphafold.py` script from the
                `alphafold_strube` repository
            model_preset (str, optional):
                Model present to use. Can be one of 'monomer',
                'monomer_casp14', 'monomer_ptm', 'multimer'. Defaults to
                'monomer_ptm'.
            recycles (int, optional):
                Number of times to recycle the output through the network.
                Defaults to 3.
            db_preset (str, optional):
                Preset MSA database configuration. `reduced_dbs` for smaller
                genetic database or `full_dbs` for full genetic database.
                Defaults to 'full_dbs'.
            max_template_date (str, optional):
                Maximum template release date to consider, in `YYYY-MM-DD`
                format. Defaults to today's date.
            out_dir (Path, optional):
                Output directory for alphafold's results. Defaults to ./out
            tempdir (Path, optional):
                Temporary directory to write the fasta sequence. Defaults to
                None.
        """
        config = configparser.ConfigParser()
        config.read(Path(__file__).parent/'config.ini')
        if gpu_type=='v100':
            self.ALPHAFOLD_DATA = Path(config['user.env']['AF_DATA'])
        else:
            self.ALPHAFOLD_DATA = Path(config['user.env']['AF_DATA_A100'])
            
        self.ALPHAFOLD_SCRIPT = Path(__file__).parent.parent / 'run_alphafold.py'

        self.SeqRecs = sequences if isinstance(sequences, list) else [sequences]
        # Reduce sequence id if it is in uniprot format
        for seq in self.SeqRecs:
            sid = seq.id.split('|')
            if len(sid)>1:
                seq.name = seq.description = seq.id = sid[1]

        self.model_preset = model_preset
        self.recycles = recycles
        self.db_preset = db_preset
        self.max_template_date = max_template_date
        self.models_to_relax = models_to_relax 
        self.use_gpu_relax = use_gpu_relax
        self.multimer_predictions_per_model = multimer_predictions_per_model
        self.use_precomputed_msas = use_precomputed_msas

        if len(self.SeqRecs) > 1:
            self.model_preset = 'multimer'

        self.chain_breaks, self.homooligomers, self.unique_names = (
                                            define_homooligomers(self.SeqRecs))

        if target_name is None:
            self.target_name = '_'.join(
                                [f'{name}-{h}' for name, h in \
                                zip(self.unique_names, self.homooligomers)])

        if out_dir is None:
            self.out_dir = Path(f'{self.target_name}')
        else:
            self.out_dir = out_dir

        self.out_model = self.out_dir / self.target_name
        
        self.fasta_path = self.out_model / f'{self.target_name}.fasta'

        self.make_args()

        # Call the parent __init__ method from the Executor class
        # with the rest of the `kw` parameters (see source executor.py file)
        super().__init__(self.args, out_dir=self.out_dir,
                         **kw)

    
    def make_args(self):
        """
        Make the appropriate arguments for the monomer or multimer case
        """
        self.args = (
            f'python3 '
            f'{self.ALPHAFOLD_SCRIPT} '
            f'--fasta_paths={self.fasta_path} '
            f'--data_dir={self.ALPHAFOLD_DATA} '
            f'--output_dir={self.out_dir} '
            f'--uniref90_database_path={self.ALPHAFOLD_DATA}/uniref90/'
                'uniref90.fasta '
            f'--mgnify_database_path={self.ALPHAFOLD_DATA}/mgnify/'
                'mgy_clusters_2018_12.fa '
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
            f'--use_precomputed_msas={str(self.use_precomputed_msas).lower()} '
            f'--models_to_relax={self.models_to_relax} '
            f'--use_gpu_relax={str(self.use_gpu_relax).lower()} '
            f'--recycles={self.recycles} '
        ).split()

        if 'multimer' in self.model_preset:
            self.args = self.args + (
                f'--uniprot_database_path={self.ALPHAFOLD_DATA}/uniprot/'
                    'uniprot.fasta '
                f'--pdb_seqres_database_path={self.ALPHAFOLD_DATA}/pdb_seqres/'
                    'pdb_seqres.txt'
                f'--num_multimer_predictions_per_model={self.multimer_predictions_per_model} '
                
            ).split()
        
        else:
            self.args = self.args + (
                f'--pdb70_database_path={self.ALPHAFOLD_DATA}/pdb70/pdb70 '
            ).split()

    def prepare(self):
        """
        Create output and temporary directories
        """
        if not self.out_dir.exists():
            self.out_dir.mkdir(parents=True)

        if not self.out_model.exists():
            self.out_model.mkdir(parents=True)

        if not self.fasta_path.exists():
            SeqIO.write(self.SeqRecs, self.fasta_path, 'fasta')

    
    def fail(self):
        """
        1. Check how many feature files were generated and do the plots
        2. Generate the error message and raise the corresponding error
        """

        features_files = list(self.out_model.glob('result_model_*'))
        
        if len(features_files)>0:
            self.finish()

        error_string = \
            f"\n{self.program} EXECUTION FAILED.\n"+\
            f"{len(features_files)} FEATURES GENERATED.\n"+\
            f"Command: {' '.join(self.args)}\n"
        if self.completed_process:
            error_string += \
                f"Returncode: {self.completed_process.returncode}\n"+\
                f"stdout: \n"+\
                self.completed_process.stdout
        
        if self.verbose:
            logging.error(error_string)

        if self.catch_out:
            with open(self.f_stdout, 'w') as f:
                f.write(error_string)

        if self.failed_message:
            raise self.error(self.failed_message)
        else:
            raise self.error
    
            
    def finish(self):
        """
        Read output from your program, e.g. parse into a DataFrame or
        save to output file. Be sure to save the output in a location other
        than `self.tempdir`, which will be removed in the cleanup() method of
        the parent Executor class.
        """

        # Read otuputs
        features_files = list(self.out_model.glob('result_model_*'))

        self.prediction_results, self.outs = (
                                            process_outputs(features_files))
        
        # Read model ranking from 'ranking_debug.json'
        with open(self.out_model/'ranking_debug.json', 'r') as f:
            self.model_rank = json.load(f)['order']

        # Plot properties
        self.plots_dir = self.out_model / 'plots'
        self.plots_dir.mkdir()

        plot_paes([self.outs[k]["pae"] for k in self.model_rank],
                  chain_breaks=self.chain_breaks, dpi=200,
                  savefile=self.plots_dir/'pae.png')

        plot_adjs([self.outs[k]["adj"] for k in self.model_rank],
                  chain_breaks=self.chain_breaks, dpi=200,
                  savefile=self.plots_dir/'predicted_contacts.png')

        plot_dists([self.outs[k]["dists"] for k in self.model_rank],
                  chain_breaks=self.chain_breaks, dpi=200,
                  savefile=self.plots_dir/'predicted_distances.png')
        
        plot_plddts([self.outs[k]["plddt"] for k in self.model_rank],
                  chain_breaks=self.chain_breaks, dpi=200,
                  savefile=self.plots_dir/'plddts.png')

        # Plot structures
        for i,name in enumerate(self.model_rank):
            try:
                plot_protein(self.prediction_results[name], self.chain_breaks)
                plt.suptitle(f'Rank {i}: {name}, '
                         f'pLDDT={self.outs[name]["pLDDT"]:.2f}, '
                         f'pTM={self.outs[name]["pTMscore"]:.2f}')

                plt.tight_layout()
                plt.savefig(self.plots_dir/f'rank_{i}_{name}.png', dpi=200)
                plt.close()
            except:
                logging.error(f"Could not generate protein plot for {name}...")
    
