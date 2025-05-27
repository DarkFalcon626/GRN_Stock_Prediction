# -*- coding: utf-8 -*-
"""
Filename: Stock-predictor-source.py
Author: Andrew Francey
Date: 2025-05-20
Description: 
    This script contains the functions and network classes for a Graph Recurrent Network (GRN) based
    approach to predicting stock market prices. The module includes:
        - Functions for data retrieval, preprocessing, and time-series sequence generation.
        - Functions to construct a fully connected graph representing inter-stock relationships.
        - The GRNStockPredictor class, which combines a Graph Convolutional Network (GCN) to extract spatial 
          features from stock data with a Gated Recurrent Unit (GRU) to capture temporal dynamics.
    This modular implementation is designed for forecasting future closing prices of tech stocks and is 
    intended to serve as both an educational tool and a prototype for more advanced trading models.
Version: 1.0.0
License: Proprietary Licesneses
Dependencies: yfinance, pandas, numpy, matplotlib, sklearn, torch
Usage: 
    To use this module import the necessary functions as 
        from Stock-predictor-source import --
"""

import yfinance as yf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
import torch
import torch.nn as nn
from torch_geometric.nn import GCNConv
from torch.utils.data import Dataset
import random


def download_and_preprocess_data(tickers, feature_list, start_date, end_date):
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

    df_list = [] # for features
    
    target_list = []
    
    scalers = {ticker: {} for ticker in tickers}
    
    for ticker in tickers:
        data = yf.download(ticker, start=start_date, end=end_date)
        df = data[feature_list].copy()
        
        # Scale each feature individually
        for feature in feature_list:
            scaler = MinMaxScaler(feature_range=(0,1))
            df[feature] = scaler.fit_transform(df[[feature]])
            # Save the scaler for potental future use.
            scalers[ticker][feature.lower()] = scaler
        
        df_list.append(df)
        target_list.append(df[['Close']])
        
    # Merge the data on common dates and create a MultiIndex
        
    df_features = pd.concat([df[feature_list] for df in df_list], axis=1, join='inner')
    df_features.columns = pd.MultiIndex.from_product([tickers, feature_list])
        
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
    
    return features_array, target_array, df_features.index, scalers


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
        super(GRNEncoder, self).__init__()
        self.gcn = GCNConv(input_dim, gcn_hidden_dim)
        self.encoder_gru = nn.GRU(num_nodes * gcn_hidden_dim, encoder_gru_hidden_dim, batch_first=True)
        self.num_nodes = num_nodes
        self.gcn_hidden_dim = gcn_hidden_dim
    
    def forward(self, x, edge_index):
        
        B, T, N, F = x.shape
        gcn_outputs = []
        
        for t in range(T):
            x_t = x[:, t, :, :]
            
            temp = []
            for b in range(B):
                node_features = x_t[b]
                
                node_emb = self.gcn(node_features, edge_index)
                node_emb = torch.relu(node_emb)
                temp.append(node_emb)
            out_t = torch.stack(temp, dim=0)
            
            gcn_outputs.append(out_t)
        
        gcn_outputs = torch.stack(gcn_outputs, dim=1)
        
        gcn_flat = gcn_outputs.view(B, T, N*self.gcn_hidden_dim)
        _, hidden = self.encoder_gru(gcn_flat)
        
        return hidden

    
class GRNDecoder(nn.Module):
    def __init__(self, decoder_input_dim, gcn_hidden_dim, decoder_gru_hidden_dim, num_nodes, forecast_steps):
        super(GRNDecoder, self).__init__()
        self.gcn = GCNConv(decoder_input_dim, gcn_hidden_dim)
        self.decoder_gru_cell = nn.GRUCell(num_nodes * gcn_hidden_dim, decoder_gru_hidden_dim)
        self.fc = nn.Linear(decoder_gru_hidden_dim, num_nodes)
        self.num_nodes = num_nodes
        self.gcn_hidden_dim = gcn_hidden_dim
        self.forecast_steps = forecast_steps
        
    def forward(self, decoder_input, hidden, edge_index, targets=None, teacher_forcing_ratio=0.5):
        
        B = hidden.shape[0]
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
                 encoder_gru_hidden_dim, decoder_gru_hidden_dim, num_nodes, forecast_steps):
        super(GRNSeq2Seq, self).__init__()
        self.encoder = GRNEncoder(encoder_input_dim, gcn_hidden_dim, encoder_gru_hidden_dim, num_nodes)
        self.decoder = GRNDecoder(decoder_input_dim, gcn_hidden_dim, decoder_gru_hidden_dim, num_nodes, forecast_steps)
    
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


    
    
    
    
    
