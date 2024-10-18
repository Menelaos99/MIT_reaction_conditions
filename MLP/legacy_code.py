    
"""
Previous implementation of hyperparams.py was changed, since GNN doesnt use any concat option. Now we distinguish between the models 
that use concat or mean pool first and then proceed 
"""
    # all_datasets = data_loading(data_path, task=task, use_only_target=use_only_target, 
    #                             concat=concat, featurisation=featurisation, use_graph=use_graph)

    # for split in range(5):
    #     print(f"split{split}")
    #     datasets_x = (all_datasets[f'train{split}'], all_datasets[f'val{split}'], 
    #                   all_datasets[f'test{split}'])         
        
    #     if model_type == "NN" or model_type == "GNN":
    #         # study_name = f"{model_type}_{task}_split={split}_(concat={concat})-{NOW.day}-{NOW.month}_{NOW.hour}:{NOW.minute}"
    #         study_name = f"{str(data_params)}_split={split}"
    #         study = optuna.create_study(direction="minimize", 
    #                                     storage=STORAGE, 
    #                                     study_name=study_name, 
    #                                     load_if_exists=False,
    #                                     sampler=SAMPLER) 

    #         def wrapped_objective(trial):
    #             return pt_objective(trial, datasets_x, use_graph, use_test=False, device=DEVICE)

    #         study.optimize(wrapped_objective, n_trials=1)
            
    #         print("Best hyperparameters: ", study.best_params)
    #         train_mae, val_mae, test_mae, test_outputs, test_truth = pt_objective(study.best_trial, datasets_x, use_graph, use_test=True, device=DEVICE)
    #         print(f"Test_{split} MAE: {test_mae}")

    #         """best_params = study.best_params
    #         n_layers = best_params['n_layers']
    #         n_units = [best_params[f'n_units_l{i}'] for i in range(n_layers)]
    #         best_lr = best_params['lr']"""
        
    #     else:
    #         trial_train_losses = []
    #         trial_test_losses = []
    #         # study_name = f"{model_type}_{task}_split={split}_(concat={concat})-{NOW.day}-{NOW.month}_{NOW.hour}:{NOW.minute}"
    #         study_name = f"{str(data_params)}_split={split}"
    #         study = optuna.create_study(direction="minimize", 
    #                                     sampler=SAMPLER,
    #                                     storage=STORAGE,  
    #                                     study_name=study_name,
    #                                     load_if_exists=False,
    #                                     )
    #         def wrapped_objective(trial):
    #             return skl_objective(trial, datasets_x, use_test=False)
            
    #         study.optimize(wrapped_objective, n_trials=20)
            
    #         print("Best hyperparameters: ", study.best_params)
    #         train_mae, val_mae, test_mae = skl_objective(study.best_trial, datasets_x, use_test=True)
    #         print(f"Test_{split} MAE: {test_mae}")