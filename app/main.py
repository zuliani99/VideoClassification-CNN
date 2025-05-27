import os, argparse, datetime, torch
import torch.nn as nn
from utils import set_seeds, get_device, accuracy_score, train_evaluate
from get_dataset import get_dataset, check_consistency
from models import get_model_dict, CNN_Architecture
from dataset_fn import get_torch_Dataloader

import logging
logger = logging.getLogger(__name__)

def start(args: argparse.Namespace) -> None:
    set_seeds(seed=args.seed)

    exp_path = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")

    app_path = os.path.dirname(__file__)
    exp_path = os.path.join(app_path, 'experiments', exp_path)
    os.makedirs(exp_path, exist_ok=True)

    logging.basicConfig(filename = f'{exp_path}/exp_log.log',
                    filemode = 'a',
                    format = '%(asctime)s - %(levelname)s: %(message)s',
                    datefmt = '%H:%M:%S',
                    level = logging.INFO)
        
    device = get_device(args)

    if args.download_dataset: get_dataset(exp_path, args)
    
    labels, train_df, test_df = check_consistency(args)
    model_dataloader_dict = get_torch_Dataloader(train_df, test_df, args)
    model_dict = get_model_dict(len(labels))
    experiments_results = []
    
    for model in args.models:
        logger.info(f'Starting training for model: {model}')


        cnn = model_dict(model)
        train_dl, val_dl, test_dl = model_dataloader_dict[model]

        optimizer = torch.optim.SGD(params=cnn.parameters(), lr=1e-3, momentum=0.9, weight_decay=5e-5) # Optimizer 
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, factor=0.1, patience=2, verbose=True) # Scheduler

        CNN_arch_single = CNN_Architecture(model = cnn, 
             train_dataloader = train_dl,
             val_dataloader = val_dl,
             optimizer = optimizer,
             loss_fn = nn.CrossEntropyLoss(), #weight=weights
             score_fn = accuracy_score,
             scheduler = scheduler,
             device = device,
             patience = 15,
             save_check = True,
             load_check_train = True,
             load_check_evaluate = True)

        experiments_results.append(train_evaluate(CNN_arch_single, test_dl, exp_path)) # Run train and evaluation




def arg_parser() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Video Classification")
    parser.add_argument('--seed', type=int, default=42, help='Seed for the random number generator')
    parser.add_argument('--download_dataset', required=False, action='store_true', help='Download the dataset if not present')
    parser.add_argument('--models', type=str, nargs='+', required=True, choices=['SingleRes', 'SingleResFovea', 'SingleResContext', 'MultiRes', 'LateFusion', 'EarlyFusion', 'SlowFusion'], help='List of models to use')
    parser.add_argument('--percentage_train_test', type=int, default=100, help='Percentage of training and testing data to use')
    parser.add_argument('--percentage_bag_shots', type=int, default=1, help='Percentage of bag shots to use')
    parser.add_argument('--percentage_to_ignore', type=int, default=10, help='Percentage of data to ignore')

    return parser.parse_args()

if __name__ == "__main__":
    start(arg_parser())