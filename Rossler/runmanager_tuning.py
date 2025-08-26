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
    validation  = "yes"

    list_epochs=[5000]
    list_batchsize=[4,8,20]
    list_batchlength=[10,20]
    list_learningrate=[0.0001,0.001,0.01]
    list_layers=[1,2,3]
    list_depth=[32,64]
    list_rtol=[1e-5]
    list_seed=[121]

    runname = "test"

    modelnumber=0

    # Create all possible combinations
    all_combinations = list(itertools.product(
        list_epochs,
        list_batchsize,
        list_batchlength,
        list_learningrate,
        list_layers,
        list_depth,
        list_rtol,
        list_seed
    ))

    # Randomly sample 20 unique combinations
    sampled_combinations = random.sample(all_combinations, min(20, len(all_combinations)))

    with ProcessPoolExecutor(max_workers=8) as executor:
        futures = []
        for combo in sampled_combinations:
            modelnumber += 1
            epochs, batchsize, batchlength, learningrate, layers, dep, rtol, seed = combo
            futures.append(
                executor.submit(
                    run_command,
                    epochs, batchsize, batchlength, learningrate,  layers, dep, rtol, seed,
                    f"{runname}{modelnumber}", dataset, validation
                )
            )
            
        for future in futures:
            try:
                future.result()
            except Exception as e:
                logging.error(f"Error occurred: {e}")
