# -*- coding: utf-8 -*-
"""
Filename: Stock-predictor-source.py
Author: Andrew Francey
Date: 2025-06-20
Description: 
    This script contains the functions and network classes for a Graph Recurrent Network (GRN) based
    approach to predicting stock market prices. The module includes:
        - Functions for data retrieval, preprocessing, and time-series sequence generation.
        - Functions to construct a fully connected graph representing inter-stock relationships.
        - The GRNStockPredictor class, which combines a Graph Convolutional Network (GCN) to extract spatial 
          features from stock data with a Gated Recurrent Unit (GRU) to capture temporal dynamics.
    This modular implementation is designed for forecasting future closing prices of tech stocks and is 
    intended to serve as both an educational tool and a prototype for more advanced trading models.
Version: 1.1.1
License: Proprietary Licesneses
Dependencies: yfinance, pandas, numpy, matplotlib, sklearn, torch
Usage: 
    To use this module import the necessary functions as 
        from Stock-predictor-source import --
"""

import yfinance as yf
import pandas as pd
import talib as ta
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from datetime import datetime, timedelta
import torch
import torch.nn as nn
from torch_geometric.nn import GCNConv
from torch.utils.data import Dataset
from fredapi import Fred
import random
import os


# FRED Api key
fred_api_key = '46cd749d3683d90133cfd5e713355873'


def download_and_preprocess_data(data_param, end_date, start_date=None, fred_api_key=fred_api_key):
    """
    Downloads historical data for the given tickers and preprocesses it.
    For each stock, we extract the following features: Open, High, Low, Close, and Volume.
    Each feature is scaled individually.
    
    Returns:
        features_array: numpy array of shape (num_days, num_stocks, num_features)
                        (features order: [Open, High, Low, Close, Volume])
        target_array: numpy array of shape (num_days, num_stocks) corresponding to Close prices.
        dates: date index for potential plotting.
        scalers: dictionary holding scalers for each ticker and each feature.
    """
    
    tickers = data_param['tickers']
    if start_date == None:
        start_date = data_param['Start_date']
    
    df_list = [] # for features
    target_list = []
    scalers = {ticker: {} for ticker in tickers}
    
    # Initialize the FRED API 
    fred = Fred(api_key=fred_api_key) if fred_api_key else None
    
    for ticker in tickers:
        data = yf.download(ticker, start=start_date, end=end_date, auto_adjust=False)
        if data.empty:
            continue
        df = data.copy()
        
        df_close_values = np.squeeze(df['Close'].to_numpy())

        # Moving Averages
        df['SMA_20'] = ta.SMA(df_close_values, timeperiod=20)
        df['SMA_50'] = ta.SMA(df_close_values, timeperiod=50)
        df['EMA_20'] = ta.EMA(df_close_values, timeperiod=20)
        df['RSI'] = ta.RSI(df_close_values, timeperiod=14)
        
        macd, macd_signal, _ = ta.MACD(df_close_values, fastperiod=12, slowperiod=26, signalperiod=9)
        df['MACD'] = macd
        df['MACD_signal'] = macd_signal
        
        bb_upper, bb_middle, bb_lower = ta.BBANDS(df_close_values, timeperiod=20, nbdevup=2, nbdevdn=2, matype=0)
        df['BB_upper'] = bb_upper
        df['BB_middle'] = bb_middle
        df['BB_lower'] = bb_lower
        
        # Volume Change (% change)
        df['Volume_change'] = df['Volume'].pct_change()
        
        # Price Change (% change)
        df['Price_change'] = df['Close'].pct_change()
        
        df = df.bfill()
        
        # Get fundamental data from yfinance
        ticker_obj = yf.Ticker(ticker)
        info = ticker_obj.info
        
#        df['EPS'] = info.get('trailingEps', np.nan)
        df['PE_ratio'] = info.get('trailingPE', np.nan)
#        df['Debt_to_Equity'] = info.get('debtToEquity', np.nan)
#        df['Revenue'] = info.get('totalRevenue', np.nan)
   
        df[['PE_ratio']] = \
            df[['PE_ratio']].ffill()
        
        # Get the marco economic indicators from FED API
        if fred:
            try:
                df['Interest_Rates'] = fred.get_series('FEDFUNDS', start_date, end_date)
                df['Commodity_Prices'] = fred.get_series('DCOILWTICO', start_date, end_date)
                df['Colatility_Index'] = fred.get_series('VIXCLS', start_date, end_date)
                df['Economic_Index'] = fred.get_series('GDP', start_date, end_date)
                
                df[['Interest_Rates', 'Commodity_Prices', 'Colatility_Index', 'Economic_Index']] = \
                    df[['Interest_Rates', 'Commodity_Prices', 'Colatility_Index', 'Economic_Index']].ffill()
                
                df[['Interest_Rates', 'Commodity_Prices', 'Colatility_Index', 'Economic_Index']] = \
                    df[['Interest_Rates', 'Commodity_Prices', 'Colatility_Index', 'Economic_Index']].bfill()
                    
            except Exception as e:
                print(f"Error fetching FRED data: {e}")
        
        new_columns = [
            (level0, level1 if level1 not in ['', None] else ticker) for level0, level1 in df.columns
            ]
        
        df.columns = pd.MultiIndex.from_tuples(new_columns, names = df.columns.names)
        
        nan = df.isna().sum()
        n_nan = df.isna().sum().sum()
        
        if n_nan != 0:
            print(nan)
            raise ValueError("Column has NaN values")  
        
        features_to_use = [
            'Open', 'High', 'Low', 'Close', 'Volume',
            'SMA_20', 'SMA_50', 'EMA_20', 'RSI', 'MACD', 'MACD_signal',
            'BB_upper', 'BB_middle', 'BB_lower', 'Volume_change', 'Price_change', 'PE_ratio',
            'Interest_Rates', 'Commodity_Prices', 'Colatility_Index','Economic_Index'
        ]
        
        # Scale the features to the domain (0,1)
        for feature in features_to_use:
            scaler = MinMaxScaler(feature_range=(0,1))
            df[feature] = scaler.fit_transform(df[[feature]])
            scalers[ticker][feature.lower()] = scaler # Save the scaler for later use.
        
        df_list.append(df[features_to_use])
        target_list.append(df[['Close']])
    
    # Merge the data on common dates and create a MultiIndex
    df_features = pd.concat([df[features_to_use] for df in df_list], axis=1, join='inner')
    df_features.columns = pd.MultiIndex.from_product([tickers, features_to_use])
    
    df_target = pd.concat([df[['Close']] for df in target_list], axis=1, join='inner')
    df_target.columns = tickers
    
    # Drop all columns that have a null value.
    df_features.dropna(inplace=True)
    df_target.dropna(inplace=True)
    
    # Build features_array: shape (T, num_stocks, num_features)
    
    features = []
    for ticker in tickers:
        features.append(df_features[ticker].values)
    
    features_array = np.stack(features, axis=1)
    target_array = df_target.values
    
    return features_array, target_array, df_features.index, scalers, features_to_use


def create_sequences(features_array, target_array, seq_length, forecast_steps):
   """
    Generates time-series sequences and corresponding multi-step target forecasts.
    
    Input:
        features_array: numpy array of shape (T, num_stocks, num_features)
        target_array: numpy array of shape (T, num_stocks) (closing prices)
    
    Returns:
        X: shape (num_samples, seq_length, num_stocks, num_features)
        y: shape (num_samples, forecast_steps, num_stocks)
    """
    
   X, y = [], []
   T = features_array.shape[0]
   for i in range(T-seq_length-forecast_steps+1):
       seq_x = features_array[i:i+seq_length] #(seq_lenght, num_stocks, num_features)
       
       seq_y = target_array[i+seq_length:i+seq_length+forecast_steps] # (forecast_steps, num_stocks)
       
       X.append(seq_x)
       y.append(seq_y)
       
   return np.array(X), np.array(y)

    
class StockDataset(Dataset):
    def __init__(self, X, y):
        
        self.X = X.astype(np.float32)
        self.y = y.astype(np.float32)
        
    def __len__(self):
        return len(self.X)
    
    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]
    

class GRNEncoder(nn.Module):
    def __init__(self, input_dim, gcn_hidden_dim, encoder_gru_hidden_dim, num_nodes):
        '''
        The sequence encoder, using a graph convolation layer GCN to get a node embedding
        of the stocks features and relationship to each other at each time step. Then
        each time steps embeddings are passed through a GRU cell to finally get a hidden 
        vector.

        Parameters
        ----------
        input_dim : Int
            The number of features per node.
        gcn_hidden_dim : Int
            Output dimension of the GCN layer.
        encoder_gru_hidden_dim : Int
            Output dimension of the GRU layer.
        num_nodes : Int
            Number of nodes (i.e. Stocks) in the graph.
        '''
        super(GRNEncoder, self).__init__()
        self.gcn = GCNConv(input_dim, gcn_hidden_dim)
        self.encoder_gru = nn.GRU(num_nodes * gcn_hidden_dim, encoder_gru_hidden_dim, batch_first=True)
        self.num_nodes = num_nodes
        self.gcn_hidden_dim = gcn_hidden_dim
    
    def forward(self, x, edge_index):
        '''
        The forward pass through the encoder network.

        Parameters
        ----------
        x : Torch tensor
            The input node features with shape (Batch, Time, Nodes, Features).
        edge_index : Torch tensor 
            A graph representing the relationship between the nodes (i.e. Stocks).

        Returns
        -------
        hidden : Torch Tensor
            The hidden encoded vector with shape (B, T, N*gcn_hidden_dim).
        '''
        
        # Extracts the batch size (B), time steps(T), number of nodes (N), and features size (F)
        B, T, N, F = x.shape
        
        # Initialize a list to store the GCN-transformed node embeddings from all the time steps.
        gcn_outputs = []
        
        for t in range(T):          # Loops through each time step in sequence.
            x_t = x[:, t, :, :]     # Extracts each time step t new shape (B,N,F)
            
            temp = []               # Initialize a list to store the GCN outputs per batch.
            for b in range(B):      # Iterates through the batch.
                node_features = x_t[b]  # Extracts node features for the batch B
                
                # Apply the graph convolution to the node features.
                node_emb = self.gcn(node_features, edge_index) 
                node_emb = torch.relu(node_emb)    # Apply activation function.
                temp.append(node_emb)
            
            # Convert all the batches to a tensor of shape (B, N, gcn_hidden_dim)
            out_t = torch.stack(temp, dim=0)
            
            gcn_outputs.append(out_t) # Store processed embeddings for time step t.
        
        # Convert all time steps into tensor with shape (B, T, N, gcn_hidden_dim)
        gcn_outputs = torch.stack(gcn_outputs, dim=1)
        
        # Flatten the 2nd and 3rd dimensions 
        gcn_flat = gcn_outputs.view(B, T, N*self.gcn_hidden_dim)
        _, hidden = self.encoder_gru(gcn_flat) # Pass the flattened sequence through the GRU. 
        # Only take the hidden state.
        return hidden

    
class GRNDecoder(nn.Module):
    def __init__(self, decoder_input_dim, gcn_hidden_dim, decoder_gru_hidden_dim, num_nodes, forecast_steps):
        '''
        The decoder taking in the hidden encoded vector and converting it to the 
        forecasted stock predictions using both the GCN and GRU networks.

        Parameters
        ----------
        decoder_input_dim : Int
            Node feature input dimension.
        gcn_hidden_dim : Int
            The hidden size of the GCN layer.
        decoder_gru_hidden_dim : Int
            The hidden size of the GRU used in decoding..
        num_nodes : Int
            The number of nodes in the graph (i.e number of stocks).
        forecast_steps : Int
            The number of days to predict.
        '''
        
        super(GRNDecoder, self).__init__()
        self.gcn = GCNConv(decoder_input_dim, gcn_hidden_dim)
        self.decoder_gru_cell = nn.GRUCell(num_nodes * gcn_hidden_dim, decoder_gru_hidden_dim)
        self.fc = nn.Linear(decoder_gru_hidden_dim, num_nodes)
        self.num_nodes = num_nodes
        self.gcn_hidden_dim = gcn_hidden_dim
        self.forecast_steps = forecast_steps
        
    def forward(self, decoder_input, hidden, edge_index, targets=None, teacher_forcing_ratio=0.5):
        '''
        Forward pass of the hidden encoder vector through the decoder to get the predicted data.

        Parameters
        ----------
        decoder_input : Torch Tensor
            Initial state of the decoder, Tensor of shape (B, 1, num_nodes * gcn_hidden_dim).
        hidden : Torch Tensor
            The inital hidden state for the GRU (e.g. from the encoder).
            Shape should be (num_layers, B, decoder_gru_hidden_dim)
        edge_index : Torch Tensor
            A graph representing the relationship between the nodes (i.e. Stocks).
        targets : Torch Tensor, optional
            Optional true targets for the teacher forcing. The default is None.
        teacher_forcing_ratio : Float, optional
            Probability of using teacher forcing at each step. The default is 0.5.

        Returns
        -------
        outputs : Torch Tensor
            The predictions of the forcasting period.

        '''
        # Extract the number of batches 
        B = hidden.shape[0]
        
        # Initalize a list to store
        outputs = []
        input_t = decoder_input
        
        for t in range(self.forecast_steps):
            temp = []
            for b in range(B):
                node_features = input_t[b]
                
                node_emb = self.gcn(node_features, edge_index)
                node_emb = torch.relu(node_emb)
                temp.append(node_emb)
            node_emb_batch = torch.stack(temp, dim=0)
            
            node_emb_flat = node_emb_batch.view(B, -1)
            hidden = self.decoder_gru_cell(node_emb_flat, hidden)
            out = self.fc(hidden)
            
            outputs.append(out.unsqueeze(1))
            
            if (targets is not None) and (random.random() < teacher_forcing_ratio):
                next_input = targets[:, t, :].unsqueeze(-1)
            else:
                next_input = out.unsqueeze(-1)
            
            input_t = next_input
        
        outputs = torch.cat(outputs, dim=1)
        
        return outputs


class GRNSeq2Seq(nn.Module):
    def __init__(self, encoder_input_dim, decoder_input_dim, gcn_hidden_dim, 
                 gru_hidden_dim, num_nodes, forecast_steps):
        super(GRNSeq2Seq, self).__init__()
        self.encoder = GRNEncoder(encoder_input_dim, gcn_hidden_dim, gru_hidden_dim, num_nodes)
        self.decoder = GRNDecoder(decoder_input_dim, gcn_hidden_dim, gru_hidden_dim, num_nodes, forecast_steps)
    
    def forward(self, x, decoder_initial_input, edge_index, targets=None, teacher_forcing_ratio=0.5):
        
        hidden = self.encoder(x, edge_index)
        decoder_hidden = hidden.squeeze(0)
        
        outputs = self.decoder(decoder_initial_input, decoder_hidden, edge_index, targets, teacher_forcing_ratio)
        return outputs
    

def create_complete_graph(num_nodes):
    edge_list = []
    for i in range(num_nodes):
        for j in range(num_nodes):
            if i != j:
                edge_list.append([i,j])
    edge_index = torch.tensor(edge_list, dtype=torch.long).t().contiguous()
    return edge_index

def plot_stocks(param, end_date, model, edge_index, scalers, device, fred_api_key=fred_api_key):
    
    # Load in the needed parameters
    data_param = param['Data']
    
    tickers = data_param['tickers']
    Seq_length = data_param['Seq_length']
    
    # Backtrack the start date of the sequence from the end date.
    start_date = datetime .strptime(end_date, '%Y-%m-%d') - timedelta(days=Seq_length)
    start_date = start_date.strftime('%Y-%m-%d')
    
    # Load in the features of the stock data.
    x_input, _, dates, scalers, _ = download_and_preprocess_data(data_param, end_date, fred_api_key=fred_api_key)
    
    # Set the model to eval mode and don't compute the gradients.
    model.eval()
    with torch.no_grad():
        if not isinstance(x_input, torch.Tensor): # Check if the input is a tensor and convert it one if not.
            x_input = torch.tensor(x_input, dtype=torch.float32)
        x_input = x_input.to(device)
        
        # Get the decoder_input.
        decoder_input = x_input[:, :-1, :].unsqueeze(-1)
        
        # Pass the sequence through the model.
        predictions_scaled = model(x_input, decoder_input, edge_index, teacher_forcing_ratio=0.0)
        print('predictions_scaled shape: ', predictions_scaled.shape())
        predictions_scaled = predictions_scaled.cpu().numpy() # Convert to numpy array and move to cpu
        
        predictions_scaled = predictions_scaled.flatten() # Remove the extra dimensions.
    
    # Undo the normalization.
    predictions = scalers.inverse_transform(predictions_scaled.reshape(-1,1))
    
    # switch the temperal axis (axis 0) with the stock axis (axis 1)
    predictions = np.transpose(predictions, (1, 0, 2))
    
    ## -- Get the actual stock data --
    start_date = end_date
    end_date = datetime.strptime(start_date, '%Y-%m-%d') + timedelta(days=7) 
    
    df_list = []
    for ticker in tickers:
        data = yf.download(ticker)
    
    for ticker, i in enumerate(tickers):
        plt.figure()
        plt.plot(dates, predictions[i][:][3], label='Prediction')
        plt.plot()
        plt.title(f"{ticker} Stock price.")
        plt.xlabel("Date")
        plt.ylabel("Closing price")
        plt.legend()
    
    plt.show()
    
    
    
    
