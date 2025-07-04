# -*- coding: utf-8 -*-


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
    parser.add_argument('--model', default='Stock-prediction.pkl', help='Name of the file the model that will be used')
    
    args = parser.parse_args()
    
    with open(args.param) as paramfile:
        param = json.load(paramfile)
    paramfile.close()
    
    model_param = param['Model']
    data_param = param['Data']
    
    end_date = (datetime.now() - timedelta(days=5)).strftime('%Y-%m-%d')
    start_date = (datetime.now()-timedelta(days=5)-timedelta(days=data_param['Seq_length'])).strftime('%Y-%m-%d')
    
    features_array, _, scalers, features = src.download_and_preprocess_data(data_param, end_date, start_date)
    print(features_array.shape)
    print(type(features_array))
    
    num_nodes = len(data_param['tickers'])
    edge_index = src.create_complete_graph(num_nodes).to(device)
    encoder_input_dim = len(features)
    
    model = src.GRNSeq2Seq(encoder_input_dim, model_param['decoder_input_dim'], 
                           model_param['gcn_hidden_dim'], model_param['gru_hidden_dim'], 
                           num_nodes, data_param['forecast_steps'])
    
    model = model.to(device)
    
    