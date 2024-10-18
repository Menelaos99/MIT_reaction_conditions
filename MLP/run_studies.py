import optuna
import sys
import os 
import re
import datetime
import pandas as pd 
import xgboost as xgb
import torch 
import torch.nn as nn 
import torch.optim as optim
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error
from torch.utils.data import DataLoader, TensorDataset

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from utils.helpers import get_data
from utils.model import NN
from utils.train import train, test
from utils.plot import plot_pred_vs_true

NOW = datetime.datetime.now()
path = os.getcwd()
df = pd.read_excel(path + '/data/reaction_entries_filtered.xlsx')
device = torch.device("cuda:1" if torch.cuda.is_available() else "cpu") 

path_add = f"plots/study{NOW.month}_{NOW.day}"
new_path = os.path.join(path, path_add) 
os.makedirs(new_path, exist_ok=True)

storage_dir = os.path.join(path, "storage")
LOAD_STORAGE = f"sqlite:///{storage_dir}/db_8_12_7_57.sqlite3"
study_summaries = optuna.study.get_all_study_summaries(storage=LOAD_STORAGE)
for i, study_summary in enumerate(study_summaries):
    name = study_summary.study_name
    try:
        model_type = name.split("_")[0] 
        task = name.split("_")[1]
        pattern = r'concat=(True|False)'
        
        match = re.search(pattern, name)
        if match:
            extracted = match.group(0)
            bool_string = extracted.split("=")[1]
            concat = bool_string == 'True'
        
        if concat:
            output_string = "concat"
        else:
            output_string = "mean_pool"
        
        data_params = [task, output_string]
    except:
        print(name)
        continue
    
    X, y= get_data(df, task = task, concat=concat)
    
    scaler_X = StandardScaler()
    scaler_y = StandardScaler()

    # X = scaler_X.fit_transform(X)
    # y = scaler_y.fit_transform(y.reshape(-1, 1)).ravel()

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
    
    study = optuna.create_study(
    direction="minimize", 
    storage=LOAD_STORAGE,  
    study_name=name,
    load_if_exists=True
    )
    if model_type == "NN":
        best_params = study.best_params
        n_layers = best_params['n_layers']
        n_units = [best_params[f'n_units_l{i}'] for i in range(n_layers)]
        best_lr = best_params['lr']
        epochs = best_params['epochs'] 

        X_train, X_test = X_train.clone().detach().requires_grad_(True), X_test.clone().detach().requires_grad_(True)
        y_train, y_test = y_train.clone().detach().requires_grad_(True).view(-1,1), y_test.clone().detach().requires_grad_(True).view(-1,1)
        
        best_model = NN(input_dim=X.shape[1], n_layers=n_layers, n_units=n_units)
        best_model.to(device)
        # best_model.n_layers(xavier_init)
        criterion = nn.L1Loss()
        optimizer = optim.Adam(best_model.parameters(), lr=best_lr)
        
        train_set = TensorDataset(X_train, y_train)
        test_set = TensorDataset(X_test, y_test)
        train_loader = DataLoader(train_set, batch_size=8, shuffle=True)
        test_loader = DataLoader(test_set, batch_size=8)
        
        for epoch in range(1, epochs):
            train_loss = train(train_loader, device, best_model, optimizer, criterion)
            print(f"Epoch{epoch}-Train loss: {train_loss}")
        test_loss, y_pred, y_test = test(test_loader, device, best_model, criterion)
        print("Test Loss:", test_loss)
    else:
        
        best_model = xgb.XGBRegressor(**study.best_params)
        best_model.fit(X_train, y_train, 
                       verbose=True)
        y_pred = best_model.predict(X_test)
        # y_pred_destd = scaler_y.inverse_transform(y_pred.reshape(-1, 1))
        
        test_loss = mean_absolute_error(y_test, y_pred)
        print("Test MAE: ", test_loss)
    
    model_path = os.path.join(new_path, f"{model_type}")
    print(model_path)
    os.makedirs(model_path, exist_ok=True)
    plot_pred_vs_true(y_test=y_test, y_pred=y_pred, path=model_path, params=data_params)
    
    with open("output.txt", "a") as f:
        if i == 0:
            f.write(f"\n\nExp-{NOW}")    
        f.write(f"\n{model_type}-{task}-{output_string}-test_loss: {test_loss}")
        