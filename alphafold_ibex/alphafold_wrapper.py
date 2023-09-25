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
from typing import Union
import shutil
import pandas as pd
import numpy as np
from typing import List, Tuple
import pickle

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
                 old_uniclust:bool=False,
                 only_features_chain: Union[str, None] = None,
                 keep_msas: bool = False,
                 features_dir: Union[Path, None] = None,
                 only_pae_interaction: bool = False,
                 model_names: List[str] = None,
                 make_plots: bool = True,
                 screen_mode: bool = False,
                 random_seed: int = None,
                 get_quality_scores: bool = False,
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
            only_features_chain (str, optional):
                If provided, only the features for this sequence will be
                calculated, with the provided chain ID. Defaults to None.
            keep_msas (bool, optional):
                If True, the MSA files will be kept after calculating the 
                features. Only used in features-only mode. Defaults to False.
            features_dir (Path, optional):
                Directory where to find the features, if they are precomputed.
            only_pae_interaction (bool, optional):
                If True, the mean of the PAE quadrant for the interaction of the
                best model will be calculated and saved to a file, and all the
                pickled results will be erased, except for the best model.
                Defaults to False.
            model_names (list, optional):
                List of models to run. If None, all models will be run.
            make_plots (bool, optional):
                If True, the plots will be generated. Defaults to True.
            screen_mode (bool, optional):
                If True, only the quality scores of the models will be written
                to a text file. No output pickle or pdbs will be saved.
            random_seed (int, optional):
                Random seed to use for the data pipeline. Doesn't guarantee
                deterministic results. Defaults to None.
            get_quality_scores (bool, optional):
                If True, the quality scores will be extracted from the models
                and saved to a file. The pickle files will be erased except for
                the one with the highest score. Defaults to False.
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
        self.old_uniclust = old_uniclust
        self.only_features_chain = only_features_chain
        self.features_dir = features_dir
        self.keep_msas = keep_msas
        self.only_pae_interaction = only_pae_interaction
        self.make_plots = make_plots
        self.screen_mode = screen_mode
        self.get_quality_scores = get_quality_scores
        
        if model_names is None:
            self.model_names_str = ''
        else:
            self.model_names_str = f"--model_names={','.join(model_names)} "
        
        if random_seed is None:
            self.random_seed_str = ''
        else:
            self.random_seed_str = f"--random_seed={random_seed} "
        
        if (not self.old_uniclust):
            self.uniref30 = self.ALPHAFOLD_DATA / 'uniref30/UniRef30_2022_02'
        else:
            self.uniref30 = self.ALPHAFOLD_DATA / 'uniref30/UniRef30_2021_03'

        if len(self.SeqRecs) > 1 or self.only_features_chain or self.features_dir:
            self.model_preset = 'multimer'

        self.chain_breaks, self.homooligomers, self.unique_names = (
                                            define_homooligomers(self.SeqRecs))

        # Set the name of the target protein model, which will be used as the
        # name of the output directory for the results from AF
        if target_name is None:
            if self.model_preset == 'monomer_ptm' or self.only_features_chain:
                self.target_name = self.unique_names[0]
            elif self.model_preset == 'multimer':
                self.target_name = '_'.join(
                                    [f'{name}-{h}' for name, h in \
                                    zip(self.unique_names, self.homooligomers)])
            else:
                raise ValueError(f'model_preset=={self.model_preset} not supported')

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
                'mgy_clusters_2022_05.fa '
            f'--bfd_database_path={self.ALPHAFOLD_DATA}/bfd/'
                'bfd_metaclust_clu_complete_id30_c90_final_seq.sorted_opt '
            f'--uniref30_database_path={self.uniref30} '
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
            f'{self.model_names_str}'
            f'{self.random_seed_str}'
            f'--screen_mode={str(self.screen_mode).lower()} '
        ).split()

        if 'multimer' in self.model_preset:
            self.args = self.args + (
                f'--uniprot_database_path={self.ALPHAFOLD_DATA}/uniprot/'
                    'uniprot.fasta '
                f'--pdb_seqres_database_path={self.ALPHAFOLD_DATA}/pdb_seqres/'
                    'pdb_seqres.txt '
                f'--num_multimer_predictions_per_model='
                f'{self.multimer_predictions_per_model} '
                f'--only_features_chain={self.only_features_chain} '
                f'--features_dir={self.features_dir} '
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
        
    def process_pae(self, pae:np.ndarray) -> Tuple[float, float]:
        """
        Calculate the minimum average PAE for the quadrants with interactions,
        and the minimum rolling window average for the rows and columns of both
        quadrants.
        
        Write the results to two files
        - mean_pae_best.txt: the minimum average PAE for the quadrants
        - min_rolling_mean_pae.txt: the minimum rolling window average

        Args:
            pae (np.ndarray): PAE matrix of the best model
        """
        # Calculate the mean of the PAE quadrants for the interaction for the
        # best model and save it to a file
        len_seq1 = len(self.SeqRecs[0])
        q2 = pae[0:len_seq1, len_seq1:]
        q3 = pae[len_seq1:, 0:len_seq1]
        
        # Report only the minimum of the two quadrants' means
        min_mean = min(q2.mean(), q3.mean())
        
        # Calculate the minimum rolling mean of the PAE quadrants for both
        # rows and columns
        q2_df = pd.DataFrame(q2)
        min_q2_columns = q2_df.rolling(window=7, axis=0).mean().min().min()
        min_q2_rows = q2_df.rolling(window=7, axis=1).mean().min().min()
        
        q3_df = pd.DataFrame(q3)
        min_q3_columns = q3_df.rolling(window=7, axis=0).mean().min().min()
        min_q3_rows = q3_df.rolling(window=7, axis=1).mean().min().min()
        
        min_rolling = min(min_q2_columns, min_q2_rows,
                            min_q3_columns, min_q3_rows)
        
        return min_mean, min_rolling
    
            
    def finish(self):
        """
        Read output from your program, e.g. parse into a DataFrame or
        save to output file. Be sure to save the output in a location other
        than `self.tempdir`, which will be removed in the cleanup() method of
        the parent Executor class.
        """
        
        # If only features are generated, remove MSAs and finish
        if self.only_features_chain:
            if not self.keep_msas:
                shutil.rmtree(self.out_model/'msas')
            
            return None
        
        if self.screen_mode:
            return None

        # Read outputs
        features_files = list(self.out_model.glob('result_model_*'))

        self.prediction_results, self.outs = (
                                            process_outputs(features_files))
        
        # Read model ranking from 'ranking_debug.json'
        with open(self.out_model/'ranking_debug.json', 'r') as f:
            self.model_rank = json.load(f)['order']

        
        if self.make_plots:
            # Plot properties
            self.plots_dir = self.out_model / 'plots'
            self.plots_dir.mkdir(exist_ok=True)

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
                    raise
        
        if self.only_pae_interaction:
            # Calculate the minimum average PAE for the quadrants with
            # interactions, and the minimum rolling window average
            pae = self.outs[self.model_rank[0]]['pae']
            min_mean, min_rolling = self.process_pae(pae)
            iptm_best = self.prediction_results[self.model_rank[0]]['iptm']
            
            # Write the scores to a file
            with open(self.out_model/'scores.txt', 'w') as f:
                f.write(f"min_mean:{min_mean:.2f}\n")
                f.write(f"min_rolling:{min_rolling:.2f}\n")
                f.write(f"iptm:{iptm_best:.2f}\n")
            
            # Erase all .pkl files except for the best model
            pickle_files = list(self.out_model.glob('*.pkl'))
            pickle_files.remove(self.out_model/f'result_{self.model_rank[0]}.pkl')
            for pf in pickle_files:
                pf.unlink()
        
        if self.get_quality_scores:
            # Get the quality scores from the models and save them to a file
            for key in self.outs:
                # Delete some values to save space
                del self.outs[key]['adj']
                del self.outs[key]['dists']
            
            # Save the results to a pickle
            with open(self.out_model / 'results_all_models.pkl', 'wb') as f:
                pickle.dump(self.outs, f)
            
            # Erase all .pkl files except for the best model
            pickle_files = list(self.out_model.glob('*.pkl'))
            pickle_files.remove(self.out_model / f'result_{self.model_rank[0]}.pkl')
            pickle_files.remove(self.out_model / 'results_all_models.pkl')
            for pf in pickle_files:
                pf.unlink()