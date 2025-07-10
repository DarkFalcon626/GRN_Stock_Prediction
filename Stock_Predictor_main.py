# -*- coding: utf-8 -*-
"""
Filename: Stock_predictor_main.py
Author: Andrew Francey
Date: 2025-07-04
Description:
Version: 1.0.0
License: Proprietary Licesneses
Dependencies: torch, json, argparse, numpy, Stock_Predictor_Source, datetime.
Usage: To run this script, simply execute:
    python Stock_Predictor_main.py
"""

import torch
import json, argparse
import numpy as np
import Stock_Predictor_Source as src
from datetime import datetime, timedelta

fred_api_key = '46cd749d3683d90133cfd5e713355873'

if __name__ == '__main__':
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    parser = argparse.ArgumentParser(description="Stock predictor")
    parser.add_argument('--param', default='param.json', help='Json file for the hyperparameters.')
    parser.add_argument('--model', default='Stock-prediction.pth', help='Name of the file the model that will be used')
    parser.add_argument('--ticker', default='all', help='Tickers of the stocks to plot')
    
    args = parser.parse_args()
    
    with open(args.param) as paramfile:
        param = json.load(paramfile)
    paramfile.close()
    
    model_param = param['Model']
    data_param = param['Data']
    
    model_save_file = torch.load(args.model, map_location='cpu')
    
    end_date = end_date = (datetime.now()).strftime('%Y-%m-%d')
    
    features_array, target_array, dates, scalers, features = src.download_and_preprocess_data(data_param, end_date)
    
    num_nodes = len(data_param['tickers'])
    edge_index = src.create_complete_graph(num_nodes).to(device)
    encoder_input_dim = len(features)
    
    model = src.GRNSeq2Seq(encoder_input_dim, model_param['decoder_input_dim'], 
                           model_param['gcn_hidden_dim'], model_param['gru_hidden_dim'], 
                           num_nodes, data_param['forecast_steps'])
    
    model.load_state_dict(model_save_file['model_state_dict'])
    
    edge_index = model_save_file['edge_index']
    
    model, edge_index = model.to(device), edge_index.to(device)
    
    src.plot_stocks_eval(data_param, model, features_array, target_array, dates, scalers, edge_index, device)
    
    
    
    