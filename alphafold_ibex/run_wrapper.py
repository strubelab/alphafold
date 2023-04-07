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

# The last time I installed it (28Nov2022) this doesn't do anything... I think
# that in one of the updated libraries there is already a logger handler being
# set, so this function doesn't have an effect anymore. Have to set the level
# of the current logger instead
# logging.basicConfig(format='%(levelname)s:%(message)s', level=logging.INFO)
logging.getLogger().setLevel(logging.INFO)
# Could add a new handler with different formatting, but have to change the
# level of the currently existing handler to ignore INFO messages, otherwise
# the messages will be duplicated
# ch = logging.StreamHandler()
# ch.setLevel(logging.INFO)
# formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
# ch.setFormatter(formatter)
# logging.getLogger().addHandler(ch)

# Take the file with sequences and the output directory from the command-line inputs
seqs_file = Path(sys.argv[1])
models_to_relax = sys.argv[2]
out_dir = Path(sys.argv[3])
recycles = int(sys.argv[4])
multimer_predictions_per_model = int(sys.argv[5])
use_precomputed_msas = sys.argv[6] == 'True'
gpu_type = sys.argv[7]
old_uniclust = sys.argv[8] == 'True'
max_template_date = sys.argv[9]
only_features_chain = sys.argv[10]
if only_features_chain == 'None':
    only_features_chain = None

with open(seqs_file, 'rb') as f:
    sequences = pickle.load(f)


# Run the Program wrapper for every sequence
for seqs in sequences:
    try:
        exe = AlphaFold(seqs, out_dir=out_dir, recycles=recycles,
                models_to_relax=models_to_relax,
                multimer_predictions_per_model=multimer_predictions_per_model,
                use_precomputed_msas=use_precomputed_msas, gpu_type=gpu_type,
                old_uniclust=old_uniclust, max_template_date=max_template_date,
                only_features_chain=only_features_chain)
        logging.info(f"Running AlphaFold for target {exe.target_name}...")
        exe.run()
    
    except (RunError, MemoryError, CalledProcessError) as e:
        features_files = list((exe.out_dir/'model').glob('result_model_*'))
        logging.error(f"{len(features_files)} FEATURES CLACULATED FOR "
                      f"{exe.target_name}")
        raise
