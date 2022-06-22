"""
Script that will be called for each job in ibex
To be used only along with the ibex wrapper
"""

from subprocess import CalledProcessError
import sys
from alphafold_wrapper import AlphaFold
from pathlib import Path
import logging
import pickle

from executor.executor import RunError

logging.basicConfig(format='%(levelname)s:%(message)s', level=logging.INFO)

# Take the file with sequences and the output directory from the command-line inputs
seqs_file = Path(sys.argv[1])
run_relax = sys.argv[2] == 'True'
out_dir = Path(sys.argv[3])
recycles = int(sys.argv[4])
multimer_predictions_per_model = int(sys.argv[5])

with open(seqs_file, 'rb') as f:
    sequences = pickle.load(f)


# Run the Program wrapper for every sequence
for seqs in sequences:
    try:
        exe = AlphaFold(seqs, out_dir=out_dir, recycles=recycles,
                run_relax=run_relax,
                multimer_predictions_per_model=multimer_predictions_per_model)
        logging.info(f"Running AlphaFold for target {exe.target_name}...")
        exe.run()
    
    except (RunError, MemoryError, CalledProcessError) as e:
        features_files = list((exe.out_dir/'model').glob('result_model_*'))
        logging.error(f"{len(features_files)} FEATURES CLACULATED FOR "
                      f"{exe.target_name}")
        raise
        
