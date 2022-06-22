"""
Module to run a set of sequences through Program in an ibex job array
"""

import pickle
import numpy as np
from pathlib import Path

from executor.ibex import IbexRun


class AlphafoldIbex(IbexRun):
    """
    Class to run a set of sequences through raptorx in an ibex job array
    """

    def __init__(self, sequences:list, out_dir:Path, time_per_command:int='auto',
        jobname:str='AlphafoldIbex', cpus:int=8, recycles:int=6, mem:int='auto',
        gpus:int='auto', run_relax:bool=True, mail:str=None,
        multimer_predictions_per_model:int=5,
        use_precomputed_msas:bool=False, **kw):
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
        self.run_relax = run_relax
        self.multimer_predictions_per_model = multimer_predictions_per_model
        self.use_precomputed_msas = use_precomputed_msas
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
        It looks at the median model length from all the groups of sequences
        provided.
        """
        total_lengths = [sum([len(s) for s in seqs]) for seqs in self.sequences]
        med_len = np.median(total_lengths)
        
        if med_len<200:
            self.gpus = 1
            self.mem = 64
            self.time_per_command = 120
        elif med_len<500:
            self.gpus = 1
            self.mem = 64
            self.time_per_command = 300
        elif med_len<1000:
            self.gpus = 1
            self.mem = 128
            self.time_per_command = 600
        else:
            self.gpus = 1
            self.mem = 256
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
            f'#SBATCH --constraint=[v100]\n'
            f'#SBATCH --array=0-{self.njobs-1}\n'
            f'{self.mail_string}'
            '\n'
            f'module load alphafold/2.1.1/python3_jupyter cuda/11.2.2\n'
            'export CUDA_VISIBLE_DEVICES=0,1,2,3\n'
            'export TF_FORCE_UNIFIED_MEMORY=1\n'
            'export XLA_PYTHON_CLIENT_MEM_FRACTION=0.5\n'
            'export XLA_PYTHON_CLIENT_ALLOCATOR=platform\n'
            '\n'
            f'conda activate {self.conda_env}\n'
            '\n'
            f'seq_file="{self.sequences_dir.resolve()}/'
            'sequences${SLURM_ARRAY_TASK_ID}.pkl"\n'
            f'time {self.conda_env}/bin/python {self.python_file} '
            '${seq_file} '
            f'{self.run_relax} {self.out_dir} {self.recycles} '
            f'{self.multimer_predictions_per_model} '
            f'{self.use_precomputed_msas}\n'
        )

        with open(self.script_file, 'w') as f:
            f.write(self.script)

