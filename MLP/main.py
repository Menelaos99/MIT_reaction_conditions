import os
import copy
import sys

import torch 
import numpy as np
import pandas as pd
import torch.nn as nn
import torch.optim as optim
import optuna
import xgboost as xgb
from torch.utils.data import DataLoader, Subset, TensorDataset
from sklearn.model_selection import train_test_split, ParameterGrid, cross_val_score
from sklearn.metrics import mean_absolute_error, make_scorer
from optuna.samplers import TPESampler

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from utils.helpers import get_data
from utils.model import NN1, xavier_init
from utils.train import train, validate, test
from utils.plot import plot_pred_vs_true, plot_loss

path = os.getcwd()
df = pd.read_csv(path + "/../dataprocessing_MTE/cleaned_sintering_dataset_encoded.csv")

data_params_config = {
    "concat": [False, True],
    "model_type": ["NN"], # [["NN", "XGB"]]
    "task": ["calc"], # [["calc", "sint"]] 
}

train_params_config = {
    "epochs": [1],
    "learning_rate": [0.1],
    "sampling_size": [100, 200] # defines the size of the validation set
}
        
data_pamams_grid = ParameterGrid(data_params_config)
train_params_grid = ParameterGrid(train_params_config)

for data_params in data_pamams_grid:
    model_type = data_params['model_type']
    concat = data_params['concat']
    task = data_params['task']
    
    X, y= get_data(df, task = task, concat=concat)
    
    if model_type == "NN":    
        nn_params_config = {
            "hidden_layers": [3],
            "hidden_dim": [32]
        }

        nn_params_grid = ParameterGrid(nn_params_config)

        # num_devices = 2  
        # device_ids = list(range(num_devices))

        validation_losses = []
        training_losses = []

        best_vals =[]
        best_models = []
        best_tests = []

        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

        data_list = TensorDataset(X, y) 
        # print(f"Data List: {len(data_list)}") #only 733 out of 2653  due to a lot of temps being NaN
        for train_params in train_params_grid:
            sampling_size = train_params["sampling_size"]
            learning_rate = train_params["learning_rate"]
            epochs = train_params["epochs"]
            
            train_test_split_index = int(0.9 * len(data_list))

            train_val_data = Subset(data_list, range(train_test_split_index))
            test_data = Subset(data_list, range(train_test_split_index, len(data_list)))

            test_loader = DataLoader(test_data, batch_size=len(test_data), shuffle=False)

            data_len = len(train_val_data)
            num_batches = data_len // sampling_size
            indices = np.arange(data_len)
            
            for model_params in nn_params_grid:
                hidden_layers = model_params['hidden_layers']
                hidden_dim = model_params['hidden_dim']
                for i in range(num_batches):
                    epoch_training_losses = []
                    epoch_validation_losses = []
                    best_val_loss = np.inf

                    model = NN1(input_size=X.shape[1], hidden_size=hidden_dim, num_hidden_layers=hidden_layers, output_size=1)
                    # model = nn.DataParallel(model, device_ids=device_ids)
                    model.apply(xavier_init)

                    start_index = i * sampling_size

                    val_indices = indices[start_index:start_index + sampling_size]
                    train_indices = np.setdiff1d(indices, val_indices)
                    # print(f"Train Indices: {len(train_indices)}", f"Validation Indices: {len(val_indices)}")

                    train_data = Subset(train_val_data, train_indices)
                    val_data = Subset(train_val_data, val_indices)

                    # print(f"Batch {i+1}/{num_batches}")#, Train Size: {len(train_data)}, Validation Size: {len(val_data)}

                    train_loader = DataLoader(train_data, batch_size=8, shuffle=True)
                    validation_loader = DataLoader(val_data, batch_size=16, shuffle=False)

                    model.to(device)

                    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
                    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min',
                                                                        factor=0.8, patience=5,
                                                                        min_lr=0.0000001)
                    criterion = nn.L1Loss()
                    best_validation_loss = float('inf')

                    for epoch in range(1, epochs+1):
                        train_loss = train(train_loader, device, model, optimizer, criterion)
                        # print(f"Epoch {epoch}, Train Loss: {train_loss:.4f}")
                        scheduler.step(train_loss)

                        validation_loss, validation_output, validation_truth_temp = validate(validation_loader, device, model, criterion)
                        # print(f"Epoch {epoch+1}, Train Loss: {loss:.4f}, Validation Loss: {validation_loss:.4f}")
                        if validation_loss < best_validation_loss:
                            best_validation_loss = validation_loss
                            best_model_state = copy.deepcopy(model.state_dict())
                            best_model = model
                            best_val_loss = validation_loss
                            # print(f"Best Validation Loss: {best_val_loss:.4f}")

                        # Store the best validation loss for the current epoch

                        epoch_validation_losses.append(validation_loss)
                        epoch_training_losses.append(train_loss)
                    if  i == num_batches-1:
                        epoch_training_losses = np.array(epoch_training_losses)
                        training_losses.append(epoch_training_losses)

                        epoch_validation_losses = np.array(epoch_validation_losses)
                        validation_losses.append(epoch_validation_losses)

                #TODO needs to be changed in order to match the different model_params and train_params 
                # training_losses = np.array(training_losses).reshape(-1)
                # validation_losses = np.array(validation_losses).reshape(-1)
                model.load_state_dict(best_model_state)
                test_loss, test_outputs, test_truth = test(test_loader, device, model, criterion)

                # print(f"Test Loss: {test_loss:.4f}")
                # print(f"validation losses: {validation_losses}")
    else:   
        use_optuna = True
        if use_optuna:
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
            def objective(trial):
                param = {
                    'booster': trial.suggest_categorical('booster', ['gbtree', 'gblinear']),
                    'lambda': trial.suggest_loguniform('lambda', 1e-8, 1.0),
                    'alpha': trial.suggest_loguniform('alpha', 1e-8, 1.0),
                    'subsample': trial.suggest_uniform('subsample', 0.5, 1.0),
                    'colsample_bytree': trial.suggest_uniform('colsample_bytree', 0.5, 1.0),
                    'max_depth': trial.suggest_int('max_depth', 1, 9),
                    'eta': trial.suggest_loguniform('eta', 1e-8, 1.0),
                    'n_estimators': trial.suggest_int('n_estimators', 100, 1000)
                }

                model = xgb.XGBRegressor(**param)

                mae_scorer = make_scorer(mean_absolute_error, greater_is_better=False)

                mae = cross_val_score(model, X_train, y_train, cv=5, scoring=mae_scorer).mean()

                return -mae
            
            study = optuna.create_study(direction='minimize')
            study.optimize(objective, n_trials=50)
            
            print("Best hyperparameters: ", study.best_params)
            
            best_model = xgb.XGBRegressor(**study.best_params)
            best_model.fit(X_train, y_train)
            y_pred = best_model.predict(X_test)
            mae = mean_absolute_error(y_test, y_pred)
            
            print("Test MAE: ", mae)
        else:
            xgb_params_config = {
                "max_depth": [6],
                "eta": [0.1],
                "subsample": [0.8],
                "colsample_bytree": [0.8],
                "num_boost_round": [1]
            }
            xgb_params_grid = ParameterGrid(xgb_params_config)

            X_train, X_temp, y_train, y_temp = train_test_split(X, y, test_size=0.3, random_state=42)
            X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=0.5, random_state=42)

            dtrain = xgb.DMatrix(X_train, label=y_train)
            dval = xgb.DMatrix(X_val, label=y_val)
            dtest = xgb.DMatrix(X_test, label=y_test)

            for model_params in xgb_params_grid:
                print(model_params)
                params = {
                    'objective': 'reg:absoluteerror',
                    'max_depth': model_params['max_depth'],
                    'eta': model_params['eta'],
                    'subsample': model_params['subsample'],
                    'colsample_bytree': model_params['colsample_bytree']
                }

                evals = [(dtrain, 'train'), (dval, 'eval')]
                num_boost_round = model_params["num_boost_round"]
                early_stopping_rounds = None 
                nfold = 5

                model = xgb.train(params, dtrain, num_boost_round, evals, early_stopping_rounds=early_stopping_rounds, verbose_eval=True)
                cv_results = xgb.cv(params,
                                dtrain,
                                num_boost_round=num_boost_round,
                                nfold=nfold,
                                early_stopping_rounds=early_stopping_rounds,
                                metrics='mae',
                                as_pandas=True,
                                seed=42
                                )

                model = xgb.train(params,
                                dtrain,
                                num_boost_round=len(cv_results),
                                evals=[(dtrain, 'train'), (dval, 'eval')],
                                early_stopping_rounds=early_stopping_rounds,
                                verbose_eval=True
                                )

                y_pred = model.predict(dtest)
                test_rmse = mean_absolute_error(y_test, y_pred)
                # print(f'Test RMSE: {test_rmse:.4f}')
                print(cv_results)
                training_losses = cv_results['train-mae-mean']
                validation_losses = cv_results['test-mae-mean']
                test_truth = np.copy(y_test)
                test_outputs = np.copy(y_pred)

    # plot_pred_vs_true(y_test=test_truth, y_pred=test_outputs, path=path, model_name=model_type)
    # plot_loss(training_losses=training_losses, validation_losses=validation_losses, path=path, model_name=model_type)