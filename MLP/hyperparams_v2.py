import os 
import sys 
import datetime

import torch
import optuna

from optuna.samplers import TPESampler
from sklearn.model_selection import ParameterGrid
from optuna.samplers import RandomSampler

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from utils.helpers import data_loading
from utils.train import pt_objective, skl_objective
import json


NOW = datetime.datetime.now()
PATH = os.getcwd()
RUN=False
DEVICE = torch.device("cuda:1" if torch.cuda.is_available() else "cpu") 
SAMPLER = TPESampler()

storage_dir = os.path.join(PATH, "storage")
STORAGE = f"sqlite:///{storage_dir}/db_{NOW.month}_{NOW.day}_{NOW.hour}_{NOW.minute}.sqlite3" 

data_params_config = {
    "featurisation": ["mtencoder_512", "mtencoder_256", "composition", "matscibert", "matminer"],
    "use_only_target": [False], #[False, True]
    "model_type": ["GNN"], #"XGB", 
    "task": ["sint"] #["calc", "sint"]
}

pooling_params_config = {
    "concat": [False]
}

results = []
data_params_grid = ParameterGrid(data_params_config)
pooling_params_grid = ParameterGrid(pooling_params_config)
# df = pd.read_csv(PATH + "/../dataprocessing_MTE/cleaned_sintering_dataset_encoded.csv")
data_path = "/home/students/code/MLP/Menelaos/reaction_conditions/dataprocessing_MTE/dataset_v7"

for data_params in data_params_grid:

    featurisation = data_params['featurisation']
    use_only_target = data_params['use_only_target']
    model_type = data_params['model_type']
    task = data_params['task']

    results_logger = {}
    results_logger['params'] = data_params
    results_logger['train_maes'] = []
    results_logger['test_maes'] = []
    results_logger['val_maes'] = []
    #results_logger['predictions'] = []
    
    n_trials = 20

    if model_type == "GNN":
        use_graph =True
    else:
        use_graph =False

    if model_type == "NN" or model_type == "XGB":
        for pooling_param in pooling_params_grid:
            concat = pooling_param['concat']
            data_params.update({'concat': pooling_param['concat']})
            print(f'Configuration: {data_params}')
            results_logger['params'] = data_params
            
            all_datasets = data_loading(data_path, 
                                        task=task,
                                        use_only_target=use_only_target, 
                                        concat=concat,
                                        featurisation=featurisation,
                                        use_graph=use_graph)
            for split in range(5):
                print(f"split{split}")
                datasets_x = (all_datasets[f'train{split}'], all_datasets[f'val{split}'], 
                              all_datasets[f'test{split}'])         

                if model_type == "NN":
                    study_name = f"{str(data_params)}_split={split}"
                    study = optuna.create_study(direction="minimize", 
                                                storage=STORAGE, 
                                                study_name=study_name, 
                                                load_if_exists=False,
                                                sampler=SAMPLER) 

                    def wrapped_objective(trial):
                        return pt_objective(trial, datasets_x, use_graph, use_test=False, device=DEVICE)

                    study.optimize(wrapped_objective, n_trials=n_trials)

                    print("Best hyperparameters: ", study.best_params)
                    train_mae, val_mae, test_mae, test_outputs, test_truth = pt_objective(study.best_trial, datasets_x, use_graph, use_test=True, device=DEVICE)
                    print(f"Test_{split} MAE: {test_mae}")
                    results_logger['train_maes'].append(float(train_mae))
                    results_logger['val_maes'].append(float(val_mae))
                    results_logger['test_maes'].append(float(test_mae))
                
                else:
                    trial_train_losses = []
                    trial_test_losses = []
                    study_name = f"{str(data_params)}_split={split}"
                    study = optuna.create_study(direction="minimize", 
                                                sampler=SAMPLER,
                                                storage=STORAGE,  
                                                study_name=study_name,
                                                load_if_exists=False,
                                                )
                    def wrapped_objective(trial):
                        return skl_objective(trial, datasets_x, use_test=False)
                    
                    study.optimize(wrapped_objective, n_trials=n_trials)
                    
                    print("Best hyperparameters: ", study.best_params)
                    train_mae, val_mae, test_mae = skl_objective(study.best_trial, datasets_x, use_test=True)
                    results_logger['train_maes'].append(float(train_mae))
                    results_logger['val_maes'].append(float(val_mae))
                    results_logger['test_maes'].append(float(test_mae))
                    print(f"Test_{split} MAE: {test_mae}")

            # Append results_logger for this parameter combination
            results.append(results_logger)

            # Save results incrementally after each combination
            with open(f'results/results_{NOW}.json', 'w') as f:
                json.dump(results, f, indent=4)

    else:
        print(f'Configuration: {data_params}')
        results_logger['params'] = data_params
        all_datasets = data_loading(data_path, 
                                    task=task,
                                    use_only_target=use_only_target, 
                                    featurisation=featurisation, 
                                    use_graph=use_graph)
        for split in range(5):
            print(f"split{split}")
            datasets_x = (all_datasets[f'train{split}'], all_datasets[f'val{split}'], 
                            all_datasets[f'test{split}'])  
            study_name = f"{str(data_params)}_split={split}"
            study = optuna.create_study(direction="minimize", 
                                        storage=STORAGE, 
                                        study_name=study_name, 
                                        load_if_exists=False,
                                        sampler=SAMPLER) 

            def wrapped_objective(trial):
                return pt_objective(trial, datasets_x, use_graph, use_test=False, device=DEVICE)

            study.optimize(wrapped_objective, n_trials=n_trials)

            print("Best hyperparameters: ", study.best_params)
            train_mae, val_mae, test_mae, test_outputs, test_truth = pt_objective(study.best_trial, datasets_x, use_graph, use_test=True, device=DEVICE)
            print(f"Test_{split} MAE: {test_mae}")
            results_logger['train_maes'].append(float(train_mae))
            results_logger['val_maes'].append(float(val_mae))
            results_logger['test_maes'].append(float(test_mae))

        # Append results_logger for this parameter combination
        results.append(results_logger)

        # Save results incrementally after each combination
        with open(f'results/results_{NOW}.json', 'w') as f:
            json.dump(results, f, indent=4)

study_summaries = optuna.study.get_all_study_summaries(storage=STORAGE)
for study_summary in study_summaries:
    print(study_summary.study_name)
