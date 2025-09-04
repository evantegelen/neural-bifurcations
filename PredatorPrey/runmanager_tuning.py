import subprocess
import pandas as pd
import subprocess
from concurrent.futures import ProcessPoolExecutor
import itertools
import logging

def run_command(epochs,
                batchsize,
                batchlength,
                learningrate,
                regulation,
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
    f"--regulation {regulation} "
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

    dataset     = "tuning_training"
    validation  = "tuning_validation"

    list_epochs=[5000]
    list_batchsize=[5]
    list_batchlength=[20]
    list_learningrate=[0.0001,0.001]
    list_regulation=[0.1,0.01,0.001]
    list_layers=[1,2,3]
    list_depth=[32,64]
    list_rtol=[1e-5]
    list_seed=[121,232,343]

    runname = "tuning"

    modelnumber=0

    with ProcessPoolExecutor(max_workers=8) as executor:
            futures = []
            
            # Iterate over all combinations of the parameters
            for epochs,batchsize,batchlength, learningrate,regulation,layers, dep,rtol,seed in itertools.product(list_epochs,list_batchsize,list_batchlength,list_learningrate,list_regulation,list_layers,list_depth,list_rtol,list_seed):
                futures.append(executor.submit(run_command, epochs,batchsize,batchlength,learningrate,regulation,layers,dep,rtol,seed,f"{runname}{int(modelnumber/3)}",dataset,validation))
                modelnumber += 1

            # Wait for all the tasks to complete
            for future in futures:
                try:
                    future.result()  # Wait for each task to complete and raise any exception
                except Exception as e:
                    logging.error(f"Error occurred: {e}")                    

