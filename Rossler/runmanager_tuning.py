import subprocess
import pandas as pd
import subprocess
from concurrent.futures import ProcessPoolExecutor
import itertools
import logging
import random

def run_command(epochs,
                batchsize,
                batchlength,
                learningrate,
                layers,
                depthlayers,
                rtol,
                seed,
                runname,
                dataset,
                validation="none"
                ):

    command = (
    f"python training.py "
    f"--epochs {epochs} "
    f"--batchsize {batchsize} "
    f"--batchlength {batchlength} "
    f"--layers {layers} "
    f"--depthlayers {depthlayers} "
    f"--learningrate {learningrate} "
    f"--rtol {rtol} "
    f"--trackloss True "
    f"--runname {runname} "
    f"--seed {seed} "
    f"--dataset {dataset} "
    f"--validation_set {validation}"
)
    try:
        subprocess.run(command, check=True)
    except subprocess.CalledProcessError as e:
        logging.error(f"Command '{command}' failed with error: {e}")

if __name__ == "__main__":

    """
    |Runmanager that was used for hyperparameter tuning. 
    """

    dataset     = "rossler_training"
    validation  = "none"

    list_epochs=[15001]
    list_batchsize=[10]
    list_batchlength=[40]
    list_learningrate=[0.001]
    list_layers=[2]
    list_depth=[64]
    list_rtol=[1e-5]
    # Use three different seeds for each combination
    list_seeds=[121,232,343,454,565]

    runname = "best"

    modelnumber=1

    # Create all possible combinations
    all_combinations = list(itertools.product(
        list_epochs,
        list_batchsize,
        list_batchlength,
        list_learningrate,
        list_layers,
        list_depth,
        list_rtol,
    ))

    # Randomly sample 20 unique combinations
    sampled_combinations = random.sample(all_combinations, min(16, len(all_combinations)))

    with ProcessPoolExecutor(max_workers=8) as executor:
        futures = []
        for combo in sampled_combinations:
            for seed in list_seeds:
                epochs, batchsize, batchlength, learningrate, layers, dep, rtol = combo
                futures.append(
                    executor.submit(
                        run_command,
                        epochs, batchsize, batchlength, learningrate, layers, dep, rtol, seed,
                        f"{runname}{modelnumber}", dataset, validation
                    )
                )
            modelnumber += 1
        for future in futures:
            try:
                future.result()
            except Exception as e:
                logging.error(f"Error occurred: {e}")
