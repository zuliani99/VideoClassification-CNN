import torch
import random
import timer
import numpy as np
import matplotlib.pyplot as plt
from torch.utils.data import random_split
from torchvision import transforms
from torch.backends import cudnn

import logging
logger = logging.getLogger(__name__)

def get_device() -> torch.device:
    """Get the available torch device (GPU if available, else CPU).

    Returns:
        torch.device: The device to use for computation.
    """
    if torch.cuda.is_available():
        logger.info(f'Using GPU')
        device = torch.device('cuda')
    else:
        logger.info('CUDA is not available. Using CPU')
        device = torch.device('cpu')
    logger.info(f'Application running on {device}\n')
    return device


def set_seeds(seed: int = 42) -> None:
    """Set random seeds for reproducibility across numpy, random, and torch.

    Args:
        seed (int, optional): The seed value to use. Defaults to 42.
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    cudnn.deterministic = True
    cudnn.benchmark = False


def accuracy_score(output, label):
    '''
    PORPOUSE:
      Perform Accuracy Score

    TAKES:
      - output: model output
      - label: ground truth

    RETURNS:
      Accuracy score
    '''

    output_class = torch.argmax(torch.softmax(output, dim=1), dim=1)
    return (output_class == label).sum().item()/len(output)


def plot_loss_curves(results_info, path: str):
    '''
    PORPOUSE:
      Plot Loss and Score curves

    TAKES:
      - results_info: dictionary containing the results and model name

    RETURNS:
      None
    '''

    res = results_info['results']
    epochs = range(1, len(res['train_loss']) + 1)

    _, ax = plt.subplots(nrows = 1, ncols = 2, figsize = (18,8))
    minloss_val = min(res['val_loss'])
    minloss_ep = res['val_loss'].index(minloss_val) + 1

    

    ax[0].plot(epochs, res['train_loss'], label = 'train_loss')
    ax[0].plot(epochs, res['val_loss'], label = 'val_loss')
    ax[0].axvline(minloss_ep, linestyle='--', color='r', label='Early Stopping Checkpoint')

    ax[0].axhline(minloss_val, linestyle='--', color='r')
    ax[0].set_title('Loss - Epochs')
    ax[0].set_xlabel('Epochs')
    ax[0].set_ylabel('Loss')
    ax[0].grid()
    ax[0].legend()

    ax[1].plot(epochs, res['train_accuracy'], label = 'train_accuracy_score')
    ax[1].plot(epochs, res['val_accuracy'], label = 'val_accuracy_score')
    ax[1].set_title('Accuracy Score - Epochs')
    ax[1].set_ylabel('Accuracy Score')
    ax[1].set_xlabel('Epochs')
    ax[1].grid()
    ax[1].legend()
    plt.suptitle(results_info['model_name'], fontsize = 30)
    plt.savefig(path)



def train_evaluate(model, test, exp_path, epochs=80):
    '''
    PORPOUSE:
      Perform the training and evaluation plotting the results

    TAKES:
      - model: CNN model to use
      - test: test dataloader
      - EPOCHS: numer of times that our model will see the data

    RETURNS:
      Train history, results from test evaluation, test and training time
    '''

    history, training_time = model.fit(epochs) 
    # Train the model, it returns ->  {'model_name': model_name, 'results': results}, elapsed_train

    logger.info(f'Total training time: {(training_time / 3600):.3f} hours')

    plot_loss_curves(history, exp_path) # Compare the results between train and validation set

    start_time = timer()
    result = model.evaluate(test) # Evaluate the model in the Tran dataloader
    # It returns -> {'model_name': model_name, 'model_loss': val_loss.item(), 'model_accuracy': val_accuracy.item()}
    end_time = timer()
    testing_time = (end_time - start_time) / 60


    logger.info(f'Total evaluation time: {testing_time:.3f} minutes\n')
    logger.info(f"TEST Results for {result['model_name']} -> loss: {result['model_loss']} accuracy-score: {result['model_accuracy']}")
    
    return (history, result, testing_time, training_time / 3600)