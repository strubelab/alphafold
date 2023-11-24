"""
Module to run a set of sequences through Program in an ibex job array
"""

import pickle
import numpy as np
from pathlib import Path
from datetime import date
from executor.ibex import IbexRun
from typing import Union, List


class AlphafoldIbex(IbexRun):
    """
    Class to run a set of sequences through raptorx in an ibex job array
    """

    def __init__(self, sequences:list, out_dir:Path, time_per_command:int='auto',
        jobname:str='AlphafoldIbex', cpus:int=8, recycles:int=6, mem:int='auto',
        gpus:int='auto', gpu_type:str='v100', models_to_relax:str='best',
        mail:str=None, multimer_predictions_per_model:int=1,
        use_precomputed_msas:bool=False, old_uniclust:bool=False,
        max_template_date:str=date.today().isoformat(),
        only_features_chain: Union[str, None]=None,
        features_dir: Union[Path, None]=None,
        only_pae_interaction: bool = False,
        model_names: List[str] = None,
        make_plots: bool = True,
        screen_mode: bool = False,
        random_seeds: List[int] = None,
        get_quality_scores: bool = False,
        **kw):
        """
        Defines the variables for the ibex job array to run Program.

        Args:
            sequences (list):
                Lisf of lists of SeqRecords to predict Alphafold models
            out_dir (Path):
                Directory to save the output for each protein sequence
            time_per_command (int, optional):
                Time to allocate for each run of Program. Defaults to 15.
            cpus (int, optional):
                Number of CPUs that each run of RaptorX will require.
                Defaults to 4.
            jobname (str, optional):
                Name of the job for ibex
            mem (int, optional):
                Specify the amount of RAM to request for the job
            gpus (int, optional):
                Specify the number of gpus that you want for the job
            only_pae_interaction (bool, optional):
                If True, only the interaction between the two chains will be
                evaluated with the mean of the PAE for the second quadrant. All
                pickled results will be erased. Defaults to False.
            model_names (list, optional):
                List of names for the models to be run. If None, all five models
                will be run. Defaults to None.
            make_plots (bool, optional):
                If True, the plots will be generated. Defaults to True.
            screen_mode (bool, optional):
                If True, only the quality scores of the models will be written
                to a text file. No output pickle or pdbs will be saved.
            random_seed (List[int], optional):
                Random seeds to use for the data pipeline. One integer per model.
                Doesn't guarantee deterministic results. Defaults to None.
            get_quality_scores (bool, optional):
                If True, the quality scores will be extracted from the models
                and saved to a file. The pickle files will be erased except for
                the one with the highest score. Defaults to False.
        """
        self.sequences = sequences
        
        # self.out_dir will contain the output file(s) for each sequence
        self.out_dir = out_dir
        
        # self.out_ibex will contain the stdout for each ibex job
        self.out_ibex = self.out_dir / 'out_ibex'
        
        # self.sequences_dir will contain the files with the sequences to be
        # taken for each job
        self.sequences_dir = self.out_dir / 'sequences'
        
        self.jobname = jobname
        self.ncommands = len(sequences)
        self.cpus_per_task = cpus
        self.recycles = recycles
        self.models_to_relax = models_to_relax
        self.multimer_predictions_per_model = multimer_predictions_per_model
        self.use_precomputed_msas = use_precomputed_msas
        self.gpu_type = gpu_type
        self.old_uniclust = old_uniclust
        self.max_template_date = max_template_date
        self.only_features_chain = only_features_chain
        self.features_dir = features_dir
        self.only_pae_interaction = only_pae_interaction
        self.make_plots = make_plots
        self.screen_mode = screen_mode
        self.random_seeds = random_seeds
        self.get_quality_scores = get_quality_scores
        
        if self.random_seeds:
            assert len(self.sequences) == len(self.random_seeds), (
                'The number of random seeds must be equal to the number of '
                'sequences')

        if model_names is None:
            self.model_names_str = 'None'
        else:
            self.model_names_str = ','.join(model_names)
            
        if mail:
            self.mail_string = (f'#SBATCH --mail-user={mail}\n'
                                f'#SBATCH --mail-type=ALL\n')
        else:
            self.mail_string = ''

        self.get_gpus_hours()
        if mem != 'auto':
            self.mem = mem
        if gpus != 'auto':
            self.gpus = gpus
        if time_per_command != 'auto':
            self.time_per_command = time_per_command

        self.python_file = Path(__file__).parent.resolve() / 'run_wrapper.py'
        self.conda_env = Path(__file__).parent.parent / 'env'

        super().__init__(time_per_command=self.time_per_command,
            out_ibex=self.out_ibex, ncommands=self.ncommands,
            jobname=self.jobname, cpus_per_task=self.cpus_per_task,
            out_dir=self.out_dir, **kw)

    def get_gpus_hours(self):
        """
        Calcualte the amount of GPUs and how many hours to allocate to the job.
        It looks at the 90th percentile of the lengths from all the sequences/models
        provided.
        """
        total_lengths = [sum([len(s) for s in seqs]) for seqs in self.sequences]
        # Calculate 90th percentile of lengths
        len90 = np.percentile(total_lengths, 90)
        
        self.gpus = 1

        # Set different times for features-only or models-only modes
        if self.only_features_chain:
            if len90<500:
                self.mem = 64
                self.time_per_command = 180
            elif len90<1000:
                self.mem = 64
                self.time_per_command = 300
            else:
                self.mem = 64
                self.time_per_command = 600
        
        elif self.features_dir:
            if len90<500:
                self.mem = 64
                self.time_per_command = 60
            elif len90<1000:
                self.mem = 64
                self.time_per_command = 120
            elif len90<2000:
                self.mem = 128
                self.time_per_command = 240
            else:
                self.mem = 128
                self.time_per_command = 600
        
        else:
            if len90<200:
                self.mem = 64
                self.time_per_command = 120
            elif len90<500:
                self.mem = 64
                self.time_per_command = 300
            elif len90<1000:
                self.mem = 128
                self.time_per_command = 600
            else:
                self.mem = 128
                self.time_per_command = 1440


    def write_sequences(self):
        """
        Write the sequences in separate files, according to the number of
        commands to be run per job.
        """
        seq_ind=0
        
        for job_num in range(self.njobs):
            job_seqs = (
                self.sequences[ seq_ind : seq_ind + self.commands_per_job ])

            seqs_file = self.sequences_dir / f'sequences{job_num}.pkl'
            with open(seqs_file, 'wb') as f:
                pickle.dump(job_seqs, f)
            
            seq_ind += self.commands_per_job
    
    
    def write_random_seeds(self):
        """
        Write the random seeds in separate files, according to the number of
        commands to be run per job.
        """
        seq_ind=0
        
        for job_num in range(self.njobs):
            job_seeds = (
                self.random_seeds[ seq_ind : seq_ind + self.commands_per_job ])

            seeds_file = self.sequences_dir / f'sequences{job_num}.seeds.pkl'
            with open(seeds_file, 'wb') as f:
                pickle.dump(job_seeds, f)
            
            seq_ind += self.commands_per_job


    def prepare(self):
        """
        Generate the output directories and the script to be run. By default,
        the script file is saved in `{self.out_ibex}/script.sh`.
        """
        if not self.sequences_dir.exists():
            self.sequences_dir.mkdir(parents=True)
        if not self.out_ibex.exists():
            self.out_ibex.mkdir(parents=True)

        self.write_sequences()
        
        if self.random_seeds:
            self.write_random_seeds()
        
        self.python_command = (
            f'python {self.python_file} '
            '${seq_file} '
            f'{self.models_to_relax} {self.out_dir} {self.recycles} '
            f'{self.multimer_predictions_per_model} '
            f'{self.use_precomputed_msas} {self.gpu_type} {self.old_uniclust} '
            f'{self.max_template_date} {self.only_features_chain} '
            f'{self.features_dir} {self.only_pae_interaction} '
            f'{self.model_names_str} {self.make_plots} {self.screen_mode} '
            f'{self.random_seeds is not None} '
            f'{self.get_quality_scores}'
        )
        
        if self.only_features_chain:
            # Write a script for CPU only to calculate the features
            self.script = (
                '#!/bin/bash -l\n'
                '#SBATCH -N 1\n'
                f'#SBATCH --partition=batch\n'
                f'#SBATCH --job-name={self.jobname}\n'
                f'#SBATCH --output={self.out_ibex}/%x.%j.out\n'
                f'#SBATCH --time={self.time_per_job}\n'
                f'#SBATCH --mem={self.mem}G\n'
                f'#SBATCH --cpus-per-task={self.cpus_per_task}\n'
                f'#SBATCH --array=0-{self.njobs-1}\n'
                f'{self.mail_string}'
                '\n'
                f'module load alphafold/2.1.1/python3\n'
                '\n'
                f'conda activate {self.conda_env}\n'
                '\n'
                f'seq_file="{self.sequences_dir.resolve()}/'
                'sequences${SLURM_ARRAY_TASK_ID}.pkl"\n'
                f'echo "{self.python_command}"\n'
                f'time {self.python_command}\n'
            )
            
        else:
            # Regular script for GPU to make models
            self.script = (
                '#!/bin/bash -l\n'
                '#SBATCH -N 1\n'
                f'#SBATCH --partition=batch\n'
                f'#SBATCH --job-name={self.jobname}\n'
                f'#SBATCH --output={self.out_ibex}/%x.%j.out\n'
                f'#SBATCH --time={self.time_per_job}\n'
                f'#SBATCH --mem={self.mem}G\n'
                f'#SBATCH --gres=gpu:{self.gpus}\n'
                f'#SBATCH --cpus-per-task={self.cpus_per_task}\n'
                f'#SBATCH --constraint={self.gpu_type}\n'
                f'#SBATCH --array=0-{self.njobs-1}\n'
                f'{self.mail_string}'
                '\n'
                'module load cuda/11.8\n'
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

        with open(self.script_file, 'w') as f:
            f.write(self.script)

