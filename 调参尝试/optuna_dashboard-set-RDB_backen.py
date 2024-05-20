#  Copyright (c) Patar my copyright message. 2024-2024. All rights reserved.

import optuna
import logging
import sys


def objective(trial):
    x = trial.suggest_float("x", -100, 100)
    y = trial.suggest_categorical("y", [-1, 0, 1])
    return x ** 2 + y


if __name__ == "__main__":
    # Add stream handler of stdout to show the messages
    optuna.logging.get_logger("optuna").addHandler(logging.StreamHandler(sys.stdout))
    study_name = "example-study"  # Unique identifier of the study.
    storage_name = "sqlite:///{}.db".format(study_name)
    study = optuna.create_study(study_name=study_name, storage=storage_name, load_if_exists=True)
    df = study.trials_dataframe(attrs=("number", "value", "params", "state"))
    print(df)
    study.optimize(objective, n_trials=100)

    print(f"Best value: {study.best_value} (params: {study.best_params})")
