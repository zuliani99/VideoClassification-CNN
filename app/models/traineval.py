
import torch
from tqdm import tqdm
from time import perf_counter as timer
import os

import logging
logger = logging.getLogger(__name__)

# CNN Architecture to perform the Train and Evaluation steps saving the results

class ModelTrainEval():

    def __init__(self, model: torch.nn.Module, train_dataloader: torch.utils.data.DataLoader, 
        val_dataloader: torch.utils.data.DataLoader, optimizer: torch.optim.Optimizer,
        loss_fn: torch.nn.Module, score_fn, scheduler: torch.optim.lr_scheduler.ReduceLROnPlateau, device: torch.device,
        patience: int, check_path: str, save_check = False, load_check_train = False, load_check_evaluate = False):

        self.model = model.to(device)
        self.optimizer = optimizer
        self.train_dataloader = train_dataloader
        self.loss_fn = loss_fn
        self.val_dataloader = val_dataloader
        self.score_fn = score_fn
        self.scheduler = scheduler
        self.device = device
        self.save_check = save_check
        self.load_check_train = load_check_train
        self.load_check_evaluate = load_check_evaluate
        self.patience = patience
        self.check_path = check_path
        
        if self.model.__class__.__name__ == 'SingleResCNN':
            self.model_name = f'{self.model.__class__.__name__}-Stream_Type_{self.model.CNN.stream_type}' # type: ignore
        else: self.model_name = self.model.__class__.__name__

        self.best_checkpoint_filename = f'{self.check_path}/best_checkpoints/{self.model_name}_checkpoint.pth.tar'
        self.last_checkpoint_filename = f'{self.check_path}/last_checkpoints/{self.model_name}_checkpoint.pth.tar'

        for folder in [self.best_checkpoint_filename, self.last_checkpoint_filename]:
            if not os.path.exists(folder): os.makedirs(folder)

        if self.load_check_evaluate: self.__load_best_checkpoint() # If the flag is true I load the best checkpoint



    def __save_checkpoint(self, checkpoint_filename, best_val_loss = None, results = None, actual_patience = None, elapsed_train = None):
        '''
        PORPOUSE:
          Save the a checkpoint model

        TAKES:
          - checkpoint_filename: google drive path where to save the checkpoint 
          - best_val_loss: best validation loss to save
          - results: dictionary containing all the results up that epoch
          - actual_patience: actual patience couter state
          - elapsed_train: time elapsed from the start of training

        RETURNS: None
        '''

        logger.info(f'=> Saving Checkpoint to {checkpoint_filename.split("/")[6]}')
        checkpoint = {'state_dict': self.model.state_dict(), 'optimizer': self.optimizer.state_dict(), 'scheduler': self.scheduler.state_dict(),
                      'patience': self.patience, 'actual_patience': actual_patience, 'results': results, 'best_val_loss': best_val_loss,
                      'elapsed_train': elapsed_train}
        torch.save(checkpoint, checkpoint_filename)
        logger.info(' DONE\n')


    
    def __load_best_checkpoint(self):
        '''
        PORPOUSE: 
          Load the best chekpoint model

        RETURNS:
          None
        '''

        logger.info('=> Loading Best Checkpoint')
        checkpoint = torch.load(self.best_checkpoint_filename, map_location=self.device)
        self.model.load_state_dict(checkpoint['state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer'])
        self.scheduler.load_state_dict(checkpoint['scheduler'])
        logger.info(' DONE\n')



    def __load_last_checkpoint(self):
        '''
        PORPOUSE:
          Load the last chekpoint model

        RETURNS:
          results fit history
        '''

        logger.info('=> Loading Last Checkpoint')
        checkpoint = torch.load(self.last_checkpoint_filename, map_location=self.device)
        self.model.load_state_dict(checkpoint['state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer'])
        self.scheduler.load_state_dict(checkpoint['scheduler'])
        logger.info(' DONE\n')
        return checkpoint['results'], checkpoint['best_val_loss'], checkpoint['patience'], checkpoint['actual_patience'], checkpoint['elapsed_train']

    
    def evaluate(self, val_dataloader: torch.utils.data.DataLoader, epoch = 0, epochs = 1):
        '''
        PORPOUSE:
          Perform the model evaluation / testing

        TAKES:
          - val_dataloader: checkpoint to load
          - epoch / epochs

        RETURNS:
          Dictionary containing the model name, loss and socore
        '''

        val_loss, val_accuracy = 0, 0

        self.model.eval() # Evaluation phase

        pbar = tqdm(enumerate(val_dataloader), total = len(val_dataloader), leave=False) # Initialize the progress bar

        with torch.inference_mode(): # Allow inference mode
            for _, (images, label, _) in pbar:
                images, label = images.to(self.device), label.to(self.device) # Move the images and labels into the device

                output = self.model(images) # Get the model output

                loss = self.loss_fn(output, label) # Get the loss

                accuracy = self.score_fn(output, label)#.item() # Perform the score

                # Increment the statistics
                val_loss += loss.item()
                val_accuracy += accuracy

                # Update the progress bar
                if epoch > 0: pbar.set_description(f'{self.model_name} EVALUATION Epoch [{epoch + 1} / {epochs}]')
                else: pbar.set_description(f'{self.model_name} TESTING')
                pbar.set_postfix(loss = loss.item(), accuracy = accuracy)
              
            val_loss /= len(val_dataloader) # Calculate the final loss
            val_accuracy /= len(val_dataloader) # Calculate the final score


        return { 'model_name': self.model_name,
                 'model_loss': val_loss,
                 'model_accuracy': val_accuracy }


    
    def fit(self, epochs: int):
        '''
        PORPOUSE: 
          Perform the model traing

        TAKES:
          - epochs: number of times that our model will see the data

        RETURNS:
          Dictionary of results containing model name and history of results for each epoch
        '''

        results = { 'train_loss': [], 'train_accuracy': [], 'val_loss': [], 'val_accuracy': [] }
        best_val_loss, best_train_accuracy = float('inf'), float('-inf')
        actual_patience = 0
        elapsed_train = 0

        if self.load_check_train and os.path.exists(self.last_checkpoint_filename):
            # If the flag is true I load the last checkpoint, importing the previous results history whereas the best loss and accuracy score
            results, best_val_loss, self.patience, actual_patience, elapsed_train = self.__load_last_checkpoint() 
        if self.patience is not None and actual_patience < self.patience - 1: # da commentare il -1
            for epoch in range(len(results['train_loss']), epochs):
                train_loss, train_accuracy = 0, 0

                pbar = tqdm(enumerate(self.train_dataloader), total = len(self.train_dataloader), leave=False) # Initialize the progress bar
                
                start_time = timer()

                for _, (images, label, _) in pbar:

                    self.model.train() # Training phase

                    # zero_grad -> backword -> step

                    self.optimizer.zero_grad()
                    images, label = images.to(self.device), label.to(self.device) # Move the images and labels into the device
                    
                    output = self.model(images) # Get the model output

                    loss = self.loss_fn(output, label) # Get the loss

                    loss.backward() # Backword step
                    self.optimizer.step()

                    train_loss += loss.item()

                    accuracy = self.score_fn(output, label) # Perform the score

                    train_accuracy += accuracy


                    # Update the progress bar
                    pbar.set_description(f'{self.model_name} TRAIN Epoch [{epoch + 1} / {epochs}]')
                    pbar.set_postfix(loss = loss.item(), accuracy = accuracy)


                train_loss /= len(self.train_dataloader) # Calculate the final loss
                train_accuracy /= len(self.train_dataloader) # Calculate the final score

                self.scheduler.step(train_loss)

                # Validation phase
                model_name, val_loss, val_accuracy = (self.evaluate(self.val_dataloader, epoch, epochs)).values()

                elapsed_train += (timer() - start_time)

                # Append the results of the current epoch
                results['train_loss'].append(train_loss)
                results['train_accuracy'].append(train_accuracy)
                results['val_loss'].append(val_loss)
                results['val_accuracy'].append(val_accuracy)

                logger.info('Epoch [{}], train_loss: {:.6f}, train_accuracy: {:.6f}, val_loss: {:.6f}, val_accuracy: {:.6f} \n'.format(
                      epoch + 1, train_loss, train_accuracy, val_loss, val_accuracy))
                

                if(self.save_check): 
                    self.__save_checkpoint(self.last_checkpoint_filename, val_loss, results, actual_patience, elapsed_train) # Save last checkpoint 
                    if(val_loss < best_val_loss): # Save best checkpoint if the loss is lower then the previous best loss
                        self.__save_checkpoint(self.best_checkpoint_filename)
                        best_val_loss = val_loss
                        actual_patience = 0 # Reset the actual patience variable
                    else:
                        if self.patience != None:
                            actual_patience += 1
                            if actual_patience >= self.patience: # Process the Early Stopping
                                self.__save_checkpoint(self.last_checkpoint_filename, val_loss, results, actual_patience, elapsed_train) # Save last checkpoint with updated actual_patience
                                logger.info(f'Early stopping, validation loss do not decreased for {self.patience} epochs')
                                pbar.close() # Closing the progress bar before exiting from the train loop
                                break


        if(self.save_check): self.__load_best_checkpoint() # Loading the best checkpoint before early stopping


        return {'model_name': self.model_name, 'results': results}, elapsed_train
