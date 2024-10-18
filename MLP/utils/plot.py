import matplotlib.pyplot as plt

# Plot predicted values vs. true values
def plot_pred_vs_true(y_test=None, y_pred=None, path=None, params=None):

    plt.figure(figsize=(12, 6))
    plt.scatter(y_test, y_pred, color='blue', label='Predicted vs True')
    plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'k--', lw=2, label='Ideal')
    plt.title('Predicted vs. True Values')
    plt.xlabel('True Values')
    plt.ylabel('Predicted Values')
    plt.legend()
    plt.grid(True)
    plt.savefig(path + f'/{params[0]}_{params[1]}_predicted_vs_true.png')

# Plotting validation loss and training loss vs. epochs
def plot_loss(training_losses=None, validation_losses=None, path=None, model_name=None):
    plt.figure(figsize=(12, 6))
    plt.plot(range(1, len(training_losses) + 1), training_losses, marker='o', linestyle='-', color='r', label='Training Loss')
    if validation_losses is not None:
        plt.plot(range(1, len(validation_losses) + 1), validation_losses, marker='o', linestyle='-', color='b', label='Validation Loss')
    
    plt.title('Training and Validation Loss vs. Epochs')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()
    plt.grid(True)
    plt.savefig(path + f'/train_validation_loss.png')