import optuna
import xgboost as xgb
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from optuna.samplers import TPESampler
# Define the objective function
def objective(trial):
    # Load the dataset
    data = load_iris()
    X_train, X_valid, y_train, y_valid = train_test_split(data.data, data.target, test_size=0.2, random_state=42)
    # Suggest hyperparameters to be optimized
    param = {
        "verbosity": 0,
        "objective": "multi:softmax",  # Classification task
        "num_class": 3,                # Number of classes
        "eval_metric": "mlogloss",     # Metric used for evaluation
        "max_depth": trial.suggest_int("max_depth", 2, 10),
        "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.1),
        "n_estimators": trial.suggest_int("n_estimators", 50, 200),
        "gamma": trial.suggest_float("gamma", 0, 1),
        "subsample": trial.suggest_float("subsample", 0.5, 1.0),
        "colsample_bytree": trial.suggest_float("colsample_bytree", 0.5, 1.0),
    }
    # Train the model with the suggested hyperparameters
    model = xgb.XGBClassifier(**param)
    model.fit(X_train, y_train)
    # Predict and calculate accuracy
    preds = model.predict(X_valid)
    accuracy = accuracy_score(y_valid, preds)
    return accuracy  # Optuna tries to maximize this objective value

# Create a study object and optimize the objective function
study = optuna.create_study(direction="maximize", sampler=TPESampler())
study.optimize(objective, n_trials=100)
# Print the best trial
best_trial = study.best_trial
print(f"Best trial's accuracy: {best_trial.value}")
print(f"Best hyperparameters: {best_trial.params}")