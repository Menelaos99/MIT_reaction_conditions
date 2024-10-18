import os 
import sys 
import copy

import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import xgboost as xgb
from tqdm import trange
from torch_geometric.loader import DataLoader as GeoLoader
from torch.utils.data import DataLoader, TensorDataset
from sklearn.metrics import mean_absolute_error
from sklearn.preprocessing import StandardScaler

from utils.helpers import scale_node_attr
# from torch.amp import GradScaler, autocast in the case we try to do data parallel 

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from utils.model import NN, GNN

def train_epoch(train_loader, device, model, optimizer, criterion, use_graph):
    model.train()
    running_loss = 0.0
    # scaler = GradScaler()
    if use_graph:
        for batch in train_loader:
            optimizer.zero_grad()
            batch = batch.to(device)

            output = model(batch.x, batch.edge_index, batch.batch)  
            loss = criterion(output, batch.y.view(-1,1))  
            # print(f"output:{output}, true:{batch.y}")
            # print("loss", loss)
            loss.backward()  
            optimizer.step()  
            running_loss += loss.item()

    else:    
        for inputs, labels in train_loader:
            inputs, labels = inputs.to(device), labels.to(device)

            outputs = model(inputs).view(-1,1)
            loss = criterion(outputs, labels)

            loss.backward()
            optimizer.step()
            optimizer.zero_grad()
            running_loss += loss.item()

    average_loss = running_loss / len(train_loader)
    #print("Running train loss", average_loss)
    # test_outputs_destd = scaler_y.inverse_transform(test_outputs)
    return average_loss

def validate(val_loader, device, model, criterion, use_graph):
    model.eval()
    running_loss = 0.0
    validation_outputs = []
    validation_truth = []

    with torch.no_grad():
        if use_graph:
            for batch in val_loader:
                batch = batch.to(device)    
                outputs = model(batch.x, batch.edge_index, batch.batch)
                val_loss = criterion(outputs, batch.y.view(-1,1))
                running_loss += val_loss.item()
                validation_outputs.append(outputs.clone().detach())
                validation_truth.append(batch.y.clone().detach())
        else:   
            for inputs, labels in val_loader:
                inputs, labels = inputs.to(device), labels.to(device)

                # with autocast(device_type='cuda', enabled=True):
                outputs = model(inputs).view(-1,1)
                val_loss = criterion(outputs, labels)

                running_loss += val_loss.item()
                validation_outputs.append(outputs.clone().detach())
                validation_truth.append(labels.clone().detach())

    average_loss = running_loss / len(val_loader)
    
    y_pred = torch.cat(validation_outputs)
    y_true = torch.cat(validation_truth)

    return y_pred, y_true


def test(test_loader, device, model, criterion, use_graph):
    model.eval()
    running_loss = 0.0
    test_outputs = []
    test_truth = []
    with torch.no_grad():
        if use_graph:
            for batch in test_loader: 
                batch = batch.to(device)  

                outputs = model(batch.x, batch.edge_index, batch.batch)
                test_loss = criterion(outputs, batch.y.view(-1,1))
                running_loss += test_loss.item()
                test_outputs.append(outputs.cpu().clone().detach().numpy())
                test_truth.append(batch.y.cpu().clone().detach().numpy())
        else:
            for inputs, targets in test_loader:
                inputs, targets = inputs.to(device), targets.to(device)

                # with autocast(device_type='cuda', enabled=True):
                outputs = model(inputs).view(-1,1)
                loss = criterion(outputs, targets)

                running_loss += loss.item()
                test_outputs.append(outputs.cpu().clone().detach().numpy())
                test_truth.append(targets.cpu().clone().detach().numpy())

    average_loss = running_loss / len(test_loader)
    test_outputs = np.concatenate(test_outputs)
    test_truth = np.concatenate(test_truth)
    # test_outputs_destd = scaler_y.inverse_transform(test_outputs)
    return test_outputs, test_truth


def pt_objective(trial, datasets, use_graph, use_test=False, device='cuda:0'):
    train_dataset, val_dataset, test_dataset = datasets

    train_dataset_copy = copy.deepcopy(train_dataset)
    val_dataset_copy = copy.deepcopy(val_dataset)
    test_dataset_copy = copy.deepcopy(test_dataset)

    use_scaler = True
    epochs = trial.suggest_int("epochs", 20, 50)
    lr = 1e-4 #trial.suggest_float('lr', 1e-5, 1e-2, log=True)
    base_layer_dim = trial.suggest_int("base_layer_dim_log", 6, 10) # log2 scale, 2^5 = 32, 2^6 = 64
    batch_size_temp = trial.suggest_int("batch_size", 4, 10)
    batch_size = 2 ** batch_size_temp
    if use_graph:
        nn_layers = trial.suggest_int("nn_layers", 1, 4)
        layer_dims = 2 ** base_layer_dim 
        nn_units = [int(layer_dims / (2 ** i)) for i in range(nn_layers)]
        
        n_layers = trial.suggest_int("n_layers", 1, 4)
        n_units = 2 ** base_layer_dim 
        conv_steps = trial.suggest_int("conv_steps", 1, 4)
    else:
        n_layers = trial.suggest_int("n_layers", 2, 5)
        layer_dims = 2 ** base_layer_dim 
        n_units = [int(layer_dims / (2 ** i)) for i in range(n_layers)] # decreasing by factor of 2


    if use_graph:
        print('Using GNN')
        if use_scaler:
            y_scaler = StandardScaler()
            # get y from train dataset
            node_features = []
            for data in train_dataset_copy:
                node_features.append(data.y.numpy())  
            all_features = np.vstack(node_features)
            y_scaler.fit(all_features)

            train_dataset_scaled = scale_node_attr(train_dataset_copy, y_scaler)
            val_dataset_scaled  = scale_node_attr(val_dataset_copy, y_scaler)
            test_dataset_scaled = scale_node_attr(test_dataset_copy, y_scaler)
        
        train_loader = GeoLoader(train_dataset_scaled, batch_size=batch_size, shuffle=True)
        val_loader = GeoLoader(val_dataset_scaled, batch_size=len(val_dataset), shuffle=False)
        test_loader = GeoLoader(test_dataset_scaled, batch_size=len(test_dataset), shuffle=False)

        model = GNN(in_channels=train_dataset[0].x.shape[1], hidden_channels=n_units, nn_units=nn_units, nn_layers=nn_layers, 
                    conv_steps=conv_steps, conv_type='GAT')
    else:
        print('Using FFNN')
        X_train, y_train = train_dataset[0], train_dataset[1], 
        X_val, y_val = val_dataset[0], val_dataset[1], 
        X_test, y_test = test_dataset[0], test_dataset[1], 
        # X_train, y_train = torch.tensor(train_dataset[0]), torch.tensor(train_dataset[1])
        # X_val, y_val = torch.tensor(val_dataset[0]), torch.tensor(val_dataset[1])
        # X_test, y_test = torch.tensor(test_dataset[0]), torch.tensor(test_dataset[1])
        if use_scaler:
            y_scaler = StandardScaler()
            y_train = torch.tensor(y_scaler.fit_transform(y_train), dtype=torch.float32).view(-1, 1)
            y_val = torch.tensor(y_scaler.transform(y_val), dtype=torch.float32).view(-1, 1)
            y_test = torch.tensor(y_scaler.transform(y_test), dtype=torch.float32).view(-1, 1)

        else:
            y_train = torch.tensor(y_train, dtype=torch.float32).view(-1, 1)
            y_val = torch.tensor(y_val, dtype=torch.float32).view(-1, 1)
            y_test = torch.tensor(y_test, dtype=torch.float32).view(-1, 1)

        X_train = torch.tensor(X_train, dtype=torch.float32)
        X_val = torch.tensor(X_val, dtype=torch.float32)
        X_test = torch.tensor(X_test, dtype=torch.float32)

        train_dataset = TensorDataset(X_train, y_train)
        val_dataset = TensorDataset(X_val, y_val)
        test_dataset = TensorDataset(X_test, y_test)

        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
        val_loader = DataLoader(val_dataset, batch_size=len(val_dataset))
        test_loader = DataLoader(test_dataset, batch_size=len(test_dataset))
        print(train_dataset[0][0].shape[0])
        model = NN(train_dataset[0][0].shape[0], n_layers=n_layers, n_units=n_units)
    
    #device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu") #use from hyperparams.py
    criterion = nn.L1Loss()
    optimizer = optim.Adam(model.parameters(), lr=lr)
    model.to(device)
    
    for _ in trange(1, epochs + 1, desc='Training Epochs'):
        train_loss = train_epoch(train_loader, device, model, optimizer, criterion, use_graph)
        y_val_pred, y_val_true = test(val_loader, device, model, criterion, use_graph)

        if use_scaler:
            y_val_pred = y_scaler.inverse_transform(y_val_pred.reshape(-1, 1))
            y_val_true = y_scaler.inverse_transform(y_val_true.reshape(-1, 1))
        
        validation_loss = mean_absolute_error(y_val_true, y_val_pred)
        #print(f"Validation Loss: {validation_loss}")

    y_test_pred, y_test_true = test(test_loader, device, model, criterion, use_graph)
    if use_scaler:
        y_test_pred = y_scaler.inverse_transform(y_test_pred.reshape(-1, 1))
        y_test_true = y_scaler.inverse_transform(y_test_true.reshape(-1, 1))
        """print(y_test_pred)
        print(y_test_true)"""


    test_loss = mean_absolute_error(y_test_true, y_test_pred)
    print(f"Validation Loss: {validation_loss}, Test Loss: {test_loss}")
    #trial.report(best_validation_loss, epoch)

    if use_test:
        return train_loss, validation_loss, test_loss, y_test_pred, y_test_true
    else:
        return validation_loss

def skl_objective(trial, datasets, use_test=False):
    train_dataset, val_dataset, test_dataset = datasets

    X_train, y_train = train_dataset[0], train_dataset[1]
    X_val, y_val = val_dataset[0], val_dataset[1]
    X_test, y_test = test_dataset[0], test_dataset[1]

    print(f"X_train: {X_train.shape}, y_train: {y_train.shape}")
    dtrain = xgb.DMatrix(X_train, label=y_train)
    dval = xgb.DMatrix(X_val, label=y_val)
    dtest = xgb.DMatrix(X_test, label=y_test)
    
    params = {
        'booster': trial.suggest_categorical('booster', ['gbtree', 'gblinear']),
        'lambda': trial.suggest_float('lambda', 1e-8, 1.0, log=True),
        'subsample': trial.suggest_float('subsample', 0.5, 1.0),
        'eta': trial.suggest_float('eta', 1e-8, 1.0, log=True),
        'learning_rate': trial.suggest_float('lr', 1e-5, 1e-2, log=True),
        'objective': 'reg:squarederror', 
        'eval_metric': 'mae'
    }
    num_boost_round = 200
    progress = dict()

    bst = xgb.train(params, dtrain, num_boost_round, evals=[(dtrain, 'train'), (dval, 'validation')],
                    early_stopping_rounds=50, 
                    evals_result=progress, 
                    verbose_eval=False)
    train_mae = np.mean(np.array(progress["train"]["mae"]))
    val_mae = np.mean(np.array(progress["validation"]["mae"]))
    
    outputs = bst.predict(dtest)
    test_mae = mean_absolute_error(y_test, outputs)
    
    if use_test:
        return float(train_mae), float(val_mae), float(test_mae)
    else:
        return float(val_mae)

