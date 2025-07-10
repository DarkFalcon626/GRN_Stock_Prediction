# -*- coding: utf-8 -*-
"""
Filename: Stock-Predictor-trainer.py
Author: Andrew Francey
Date: 2025-07-04
Description: This script implements a Graph Recurrent Network (GRN) that 
            forecasts the next month's closing prices for a selected group of 
            tech stocks. It uses historical stock data obtained via yfinance,
            processes the data into sequences, constructs a fully connected 
            graph for the stocks, and builds a model that combines a Graph 
            Convolutional Network (GCN) with a GRU to capture both spatial 
            (graph) and temporal (sequential) dependencies.
Version: 1.1.2
License: Proprietary Licesnses
Dependencies: yfinance, pandas, numpy, matplotlib, sklearn, torch
Usage: To run the script, simply execute:
        python Stock_Predictor_train.py

    The script will:
        - Download historical closing prices for a predefined list of tech stocks.
        - Preprocess and scale the data.
        - Create time-series sequences for both the input window and the forecast horizon.
        - Construct a fully connected graph representing inter-stock relationships.
        - Train a Graph Recurrent Network (GRN) that uses a GCN component to extract spatial features and a GRU 
          to model temporal dynamics.
        - Output the forecasted closing prices for the next 30 days.

    Optional command line arguments can be implemented in the future to allow customization of:
        -- stock tickers (default: AAPL, MSFT, GOOGL, AMZN, NVDA)
        -- data start/end dates (default: 2015-01-01 to 2025-01-01)
        -- input sequence length (default: 60 days)
        -- forecast horizon (default: 30 days)
"""

import torch
import time
from datetime import datetime, timedelta
import json, argparse
import Stock_Predictor_Source as src
import matplotlib.pyplot as plt
import torch.nn as nn
import torch.optim as optim
import numpy as np
from torch.utils.data import DataLoader
from multiprocessing import cpu_count


class early_stopping:
    '''
    Stops training if validation loss doesn't improve after a given patience.'
    '''
    
    def __init__(self, patience=3, min_delta=0.5, save_path='checkpoint.pt'):
        
        self.patience = patience
        self.min_delta = min_delta
        self.save_path = save_path
        self.best_loss = np.Inf
        self.counter = 0 
        self.early_stop = False
        
    def __call__(self, val_loss, model):
        
        if val_loss < self.best_loss - self.min_delta:
            self.best_loss = val_loss
            self.counter = 0
            
            torch.save(model.state_dict(), self.save_path)
        
        else:
            self.counter += 1
            print(f"Early Stopping counter: {self.counter} out of {self.patience}")
            
            if self.counter >= self.patience:
                self.early_stop = True
    
def prep(param):
    '''
    Function to set up the training and test datasets as well as the GRN encoder
    and decoder as well as the graph.

    Parameters
    ----------
    param : JSON file.
        JSON file containing the hyperparameters of the dataset and model.

    Returns
    -------
    train_loader : torch.utils.data.DataLoader 
        Dataloader for the training data.
    val_loader : torch.utils.data.DataLoader
        Dataloader for the validation DataLoader.
    model : GRNSeq2Seq
        The GRN decoder and encoder.
    edge_index : Torch.tensor
        The tensor of the edge values of the graph.
    '''
    
    # Load in the hyperparameter for the data and model.
    data_param = param['Data']
    model_param = param['Model']
    
    # Use the current date minus a week as the end date.
    end_date = (datetime.now()).strftime('%Y-%m-%d')
    
    # Download the stock data.
    features_array, target_array, _, scalers, features = src.download_and_preprocess_data(data_param, end_date)
    print("Features shape: ", features_array.shape)
    print("Target shape: ", target_array.shape)
    
    # Create sequences
    X, y = src.create_sequences(features_array, target_array, data_param['Seq_length'], data_param['forecast_steps'])
    print("X shape: ", X.shape, "y shape: ", y.shape)
    
    # Shuffle the sequences
    permutation = np.random.permutation(len(X))
    
    X, y = X[permutation], y[permutation]
    
    # Split into training, validation and test sets.
    num_sample = X.shape[0]
    train_split_idx = int(num_sample * data_param['train_percent'])
    val_split_idx = train_split_idx + int(num_sample * data_param['val_percent'])
    
    
    X_train, y_train = X[:train_split_idx], y[:train_split_idx]
    X_val, y_val = X[train_split_idx:val_split_idx], y[train_split_idx:val_split_idx]
    X_test, y_test = X[val_split_idx:], y[val_split_idx:]
    
    num_cores = cpu_count() # Determine number of cores in cpu for dataloader.
    
    # Set up the datasets and the loaders for both training and validation sets.
    train_dataset = src.StockDataset(X_train, y_train)
    train_loader = DataLoader(train_dataset, batch_size=data_param['batch_size'], shuffle=True, num_workers=num_cores)
    
    # Get the batch size of the valadation set.
    val_size = len(X_val)
    
    val_dataset = src.StockDataset(X_val, y_val)
    val_loader = DataLoader(val_dataset, batch_size=val_size, shuffle=False, num_workers=num_cores)
    
    # Get the test size of the test set.
    test_size = len(X_test)
    
    test_dataset = src.StockDataset(X_test, y_test)
    test_loader = DataLoader(test_dataset, batch_size=test_size, shuffle=False, num_workers=num_cores)

    # Create the graph between the nodes (i.e. stocks)
    num_nodes = len(data_param['tickers'])
    edge_index = src.create_complete_graph(num_nodes).to(device)
    
    encoder_input_dim = len(features)
    
    # Create the model.
    model = src.GRNSeq2Seq(encoder_input_dim, model_param['decoder_input_dim'], model_param['gcn_hidden_dim'],
                       model_param['gru_hidden_dim'], num_nodes, data_param['forecast_steps'])
    
    model = model.to(device) # Send model to the device 
    
    return train_loader, val_loader, test_loader, model, edge_index, scalers


def train_model(model, dataloader, val_loader, test_loader, edge_index, device, train_param):
    '''
    Trains the GRN model and evaluates its performance at the end of each epoch.

    This function performs the training loop for a Graph Recurrent Network (GRN) sequence-to-sequence model,
    updating model weights using batches from the training dataloader. At the end of each epoch, it switches the model to 
    evaluation mode and calculates the average validation loss on the validation dataset. The function supports teacher 
    forcing during training (controlled via teacher_forcing_ratio), while during evaluation no teacher forcing is applied 
    (i.e., teacher_forcing_ratio is set to 0).

    Parameters
    ----------
    model : torch.nn.Module
        The GRN sequence-to-sequence model to be trained.
    dataloader : torch.utils.data.DataLoader
        Dataloader that yields training batches. Each batch should be a tuple (x_batch, y_batch), where:
            - x_batch: Tensor of shape (B, T, D) containing the input sequences (B: batch size, T: number of time steps, D: feature dimension).
            - y_batch: Tensor of shape (B, forecast_steps, N) representing the target values.
    val_loader : torch.utils.data.DataLoader
        Dataloader that yields validation batches, having the same structure as the training dataloader.
    edge_index : torch.Tensor
        Tensor representing the graph connectivity among nodes (e.g., the stocks in the model). 
        Typically of shape (2, E), where E is the number of edges.
    num_epochs : int
        The number of epochs for which the model will be trained.
    device : torch.device
        The device (CPU or GPU) on which the model and data will be allocated.
    train_param : dict
        Dictionary containing training parameters. Must include at least the key 'lr' for the learning rate.
    teacher_forcing_ratio : float, optional
        The probability of applying teacher forcing during training. 
        Defaults to 0.5; during evaluation it is set to 0.

    Notes
    -----
    The initial input for the decoder is created from the training and validation sequences by taking all time steps except 
    the last (using slicing x_batch[:, :-1, :]) and then adding an extra dimension via unsqueeze(-1). This input serves as the 
    starting point for the decoder's iterative forecast generation.



    Returns
    -------
    train_loss : listof Floats
        A list of the training losses per epoch.
    val_loss : listof Floats
        A list of the validation losses per epoch.
    '''
    
    num_epochs = train_param['num_epochs']
    teacher_forcing_ratio = train_param['teacher_forcing_ratio']
    
    # Define the loss function using the Mean Squared Error (MSE) and the Adam optimizer.
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=train_param['lr'], weight_decay=1e-7)
    
    # Using a learning rate scheduling
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=train_param['factor'], 
                                                     threshold=train_param['threshold'], patience=train_param['patience'])
    
    # Initialize lists to store the average training and test loss value per epoch.
    train_loss, cross_loss = [], []
    
    # Create early stopping object
    early_stopper = early_stopping(patience=train_param['stopping_patience'], min_delta=train_param['min_delta'],
                                   save_path='best_model.pt')
    
    # Loop through each epoch.
    for epoch in range(num_epochs):
        model.train() # Set the model to train mode.
        epoch_loss = 0.0 # Variable to accumulate the total training loss for the current epoch.
        
        # Iterate over each batch from the training dataloader.
        for x_batch, y_batch in dataloader:
            x_batch = x_batch.to(device) # Move the inputs and targets to the device.
            y_batch = y_batch.to(device)
            
            # Zero out the gradients so they do not accumulate from previous batches.
            optimizer.zero_grad()
            
            # Create initial decoder input:
            # Here we take all time steps from the input sequence except the last,
            # and then add an extra dimension to match the expected input shape for the decoder.
            decoder_input = x_batch[:, -1, :, 3].unsqueeze(-1)
            
            # Preform the forward pass throught the model with teacher forcing during training.

            output = model(x_batch, decoder_input, edge_index, targets=y_batch, teacher_forcing_ratio=teacher_forcing_ratio)
            
            # Compute the loss between the model's output and the actual target value.
            loss = criterion(output, y_batch)
            loss.backward() # Backpropagate teh loss
            
            optimizer.step() # Update the model parameters
            
            # Accumulate the batch loss
            epoch_loss += loss.item()
        
        # Average training loss for the epoch is the total loss divided by the number of batches
        avg_train_loss = epoch_loss/len(dataloader)
        
        train_loss.append(avg_train_loss)
        
        # Switch the model to evaluation mode for validation (disables dropout, etc.)
        model.eval()
        val_loss = 0.0  # Variable to accumulate the total validation loss for the current epoch.
        
        # Disable gradient computations for efficiency during the evaluation phase.
        with torch.no_grad():
            
            # Iterate over each batch in the validation dataloader.
            for x_val, y_val in val_loader:
                x_val = x_val.to(device) # Move data and targets to device.
                y_val = y_val.to(device)
                
                # Create initial decoder input for validation data.
                decoder_input_val = x_val[:,-1,:,3].unsqueeze(-1)

                # Teacher forcing is disabled during evaluation (teacher_forcing_ratio=0.0).
                output_val = model(x_val, decoder_input_val, edge_index, targets=y_val, teacher_forcing_ratio=0.0)
                
                # Get the model's output for the current validation batch.
                loss_val = criterion(output_val, y_val)
                
                # Accumulate the batch validation loss
                val_loss += loss_val.item() * x_val.size(0)
        
        # Compute the average validation loss for the epoch
        avg_val_loss = val_loss / len(val_loader.dataset)
        
        cross_loss.append(avg_val_loss)
        
        scheduler.step(val_loss)
        
        # Print out the learning rate for monitoring.
        current_lr = optimizer.param_groups[0]['lr']
        
        print(f"Epoch {epoch+1}/{num_epochs} - Training Loss: {avg_train_loss:.5f}, Validation Loss: {avg_val_loss:.5f}, lr: {current_lr}")
        
        early_stopper(val_loss, model)
        if early_stopper.early_stop:
            print("Early stopping triggered. Restoring best model.")
            model.load_state_dict(torch.load('best_model.pt'))
            break
    
    for x_test, y_test in test_loader:
        x_test = x_test.to(device)
        y_test = y_test.to(device)
        
        # Create initial decoder input for test data.
        decoder_input_test = x_test[:,-1,:,3].unsqueeze(-1)
        
        output_test = model(x_test, decoder_input_test, edge_index, targets=y_test, teacher_forcing_ratio=0.0)
        
        test_loss = criterion(output_test, y_test)
        
        print(f"Final test loss is {test_loss:.7f}.")
        
    return train_loss, cross_loss


def plot_loss(train_loss, cross_loss):
    
    num_epochs = len(train_loss)
    
    epochs = np.arange(1,num_epochs+1)
    
    train_loss = np.array(train_loss)
    cross_loss = np.array(cross_loss)
    
    plt.plot(epochs, train_loss, label='Training loss')
    plt.plot(epochs, cross_loss, label='Testing loss')
    plt.title("Loss per epoch (MSELoss)")
    plt.grid()
    plt.legend()
    plt.show()

        
if __name__ == '__main__':
    
    start_time = time.time() # Mark the time the model began training
    
    # Determine if a GPU is availible for use.
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"\tUsing device: {device}")
    
    # Set up commandline arguments.
    parser = argparse.ArgumentParser(description="Stock prediction")
    parser.add_argument('--param', default='param.json', help='Json file for the hyperparameters.')
    parser.add_argument('--model', default='Stock-prediction.pth', help='Name of the file the model will be saved as.')
    parser.add_argument('--fine-tune', default='True',type = bool, help="Bool for if the model's hyperparameters are being fine-tuned")
    
    args = parser.parse_args()
    
    with open(args.param) as paramfile:
        param = json.load(paramfile)
    paramfile.close()
    
    train_loader, val_loader, test_loader, model, edge_index, scalers = prep(param)
    print('Data and model loaded, Begining training....')
    
    train_param = param['Train']

    train_loss, cross_loss = train_model(model, train_loader, val_loader, test_loader, edge_index, device, train_param)
    
    plot_loss(train_loss, cross_loss)
    
    # Save both the model and the edge_index together in a dictionary
    save_dict = {
        'model_state_dict': model.state_dict(),
        'edge_index': edge_index
        }
    
    # Save the model using torch's built in method to the pth file.
    torch.save(save_dict, 'Stock-prediction.pth')
    
    print(f"Model saved as {args.model}.")
    
    train_time = time.time()-start_time
    if (train_time//3600) > 0:
        hours = train_time//3600
        mins = (train_time-hours*3600)//60
        secs = train_time-hours*3600-mins*60
        print('The model trained in {} hours, {} mins and {:.0f} seconds'.format(hours,mins,secs))
    elif (train_time//60) > 0:
        print('The model trained in {} mins and {:.0f} seconds'.format(train_time//60,
                                                               train_time-(train_time//60)*60))
    else:
        print('The model trained in {:.0f} seconds'.format(train_time))
    
    
        
