import torch
import numpy as np
import torch.nn as nn
import torch.nn.utils as nn_utils

from tqdm import tqdm
from models import ARModel, LSTM, SFM
from utils import get_data, scaled_data
from torch.optim import RMSprop, Adam
from torch.optim.lr_scheduler import StepLR
from torch.utils.data import DataLoader, TensorDataset
from sklearn.metrics import mean_squared_error


def Base_Exp1(steps_ahead):
    data = get_data()
    x_train, y_train, x_val, y_val, x_test, y_test, scaler = scaled_data(data, steps_ahead)
    test_len = x_test.shape[1] - x_val.shape[1]
    pred_y = x_test[:,-test_len:,0].numpy()
    true_y = y_test[:,-test_len:,0].numpy()
    pred_y = scaler.inverse_transform(pred_y.T)
    true_y = scaler.inverse_transform(true_y.T)
    mse = mean_squared_error(pred_y, true_y)
    return mse

def AR_Exp1(steps_ahead):
    lags = 5
    model = ARModel(steps_ahead, lags=lags)
    data = get_data()
    x_train, y_train, x_val, y_val, x_test, y_test, scaler  = scaled_data(data, steps_ahead)
    num_stocks = x_test.shape[0]
    test_len = x_test.shape[1] - x_val.shape[1]
    pred_data = np.zeros((num_stocks, test_len))

    for j in range(num_stocks):
        predictions = []
        history = x_test[j, :, 0].detach().numpy()
        for i in range(test_len):
            end = x_val.shape[1] +1 + i
            X = history[:end]
            pred = model.predict_next(X)
            predictions.append(pred)
        pred_data[j, :] = predictions
        
    y_true = y_test[:, y_val.shape[1]:,0].detach().numpy()
    pred_data_unscaled = scaler.inverse_transform(pred_data.T)
    true_data_unscaled = scaler.inverse_transform(y_true.T)
    mse = mean_squared_error(true_data_unscaled, pred_data_unscaled)
    return mse

def LSTM_Exp1(steps_ahead, hidden_dim, lr, epochs, device, load_model=False):
    input_dim = 1
    output_dim = 1
    num_layers = 1
    batch_size = 50
    
    # Load data
    data = get_data()
    x_train, y_train, x_val, y_val, x_test, y_test, scaler = scaled_data(data, steps_ahead)
    train_loader = DataLoader(TensorDataset(x_train, y_train), batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(TensorDataset(x_val, y_val), batch_size=batch_size, shuffle=False)
    val_len = x_val.shape[1] - x_train.shape[1]
    test_len = x_test.shape[1] - x_val.shape[1]
    
    if load_model:
        model = LSTM(input_dim=input_dim, 
                     hidden_dim=hidden_dim, 
                     output_dim=output_dim, 
                     num_layers=num_layers).to(device)
        model.load_state_dict(torch.load(f"LSTM_weights/Exp1/best_model_step_{steps_ahead}.pth"))

        # Model evaluation
        model.eval()
        with torch.no_grad():
            h0 = torch.zeros(num_layers, batch_size, hidden_dim).to(device)
            c0 = torch.zeros(num_layers, batch_size, hidden_dim).to(device)
            y_pred = model(x_test.to(device), h0, c0)
            y_pred = y_pred[:, -test_len:, :]
            y_test = y_test[:, -test_len:, :]
        y_pred = y_pred.cpu().detach().numpy().reshape(50, -1).T
        y_test = y_test.cpu().detach().numpy().reshape(50, -1).T
        unnorm_y_pred = scaler.inverse_transform(y_pred)
        unnorm_y_test = scaler.inverse_transform(y_test)
        mse = mean_squared_error(unnorm_y_test, unnorm_y_pred)
        return mse
    
    else : 
        model = LSTM(input_dim=input_dim, 
                     hidden_dim=hidden_dim, 
                     output_dim=output_dim, 
                     num_layers=num_layers).to(device)
        optimizer = RMSprop(model.parameters(), lr=lr)
        criterion = nn.MSELoss()
        
        # Best_model selection
        best_val_loss = float('inf')
        best_model_state = None  # To save the best model's state_dict

        # Model training
        with tqdm(total=epochs, desc=f"Step Size: {steps_ahead} Training") as pbar:
            for e in range(epochs):
                model.train()
                train_loss = 0.0
                for X_batch, y_batch in train_loader:
                    X_batch = X_batch.to(device)
                    y_batch = y_batch.to(device)
                    batch_size = X_batch.size(0)
                    h0 = torch.zeros(num_layers, batch_size, hidden_dim).to(device)
                    c0 = torch.zeros(num_layers, batch_size, hidden_dim).to(device)
                    optimizer.zero_grad()
                    y_pred = model(X_batch, h0, c0)
                    loss = criterion(y_pred, y_batch)
                    loss.backward()
                    optimizer.step()
                    train_loss += loss.item()
                train_loss /= len(train_loader)

                # Validation
                model.eval()
                val_loss = 0.0
                with torch.no_grad():
                    for X_val, y_val in val_loader:
                        X_val = X_val.to(device)
                        y_val = y_val.to(device)
                        batch_size = X_val.size(0)
                        h0 = torch.zeros(num_layers, batch_size, hidden_dim).to(device)
                        c0 = torch.zeros(num_layers, batch_size, hidden_dim).to(device)
                        y_pred = model(X_val, h0, c0)
                        y_pred = y_pred[:, -val_len:, :]
                        y_val = y_val[:, -val_len:, :]
                        loss = criterion(y_pred, y_val)
                        val_loss += loss.item()
                val_loss /= len(val_loader)
                if val_loss < best_val_loss:
                    best_val_loss = val_loss
                    best_model_state = model.state_dict()

                # Update progress bar
                pbar.set_description(f"Step: {steps_ahead}, Train Loss: {round(train_loss, 5)}, Val Loss: {round(val_loss, 5)}")
                pbar.update(1)

        # Save the best model
        best_model_path = f"LSTM_weights/Exp1/best_model_step_{steps_ahead}.pth"
        torch.save(best_model_state, best_model_path)
        
        # Load the best model 
        model.load_state_dict(best_model_state)

        # Model evaluation
        model.eval()
        with torch.no_grad():
            h0 = torch.zeros(num_layers, batch_size, hidden_dim).to(device)
            c0 = torch.zeros(num_layers, batch_size, hidden_dim).to(device)
            y_pred = model(x_test.to(device), h0, c0)
            y_pred = y_pred[:, -test_len:, :]
            y_test = y_test[:, -test_len:, :]
        y_pred = y_pred.cpu().detach().numpy().reshape(50, -1).T
        y_test = y_test.cpu().detach().numpy().reshape(50, -1).T
        unnorm_y_pred = scaler.inverse_transform(y_pred)
        unnorm_y_test = scaler.inverse_transform(y_test)

        # Calculate Mean Squared Error
        mse = mean_squared_error(unnorm_y_test, unnorm_y_pred)
        return mse
    
def SFM_Exp1(steps_ahead, freq_dim, hidden_dim, lr, epochs, device, load_model=False):
    batch_size = 50
    input_dim = 1
    output_dim = 1

    # Load data
    data = get_data()
    x_train, y_train, x_val, y_val, x_test, y_test, scaler = scaled_data(data, steps_ahead)
    train_loader = DataLoader(TensorDataset(x_train, y_train), batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(TensorDataset(x_val, y_val), batch_size=batch_size, shuffle=False)
    val_len = x_val.shape[1] - x_train.shape[1]
    test_len = x_test.shape[1] - x_val.shape[1]
    
    if load_model:
        model = SFM(input_dim=input_dim,
                    hidden_dim=hidden_dim,
                    freq_dim=freq_dim,
                    output_dim=output_dim).to(device)
        model.load_state_dict(torch.load(f"SFM_weights/Exp1/best_model_step_{steps_ahead}.pth"))
        
        # Model evaluation
        model.eval()
        with torch.no_grad():
            y_pred = model(x_test.to(device))
            y_pred = y_pred[:, -test_len:, :]
            y_true = y_test[:, -test_len:, :]
        y_pred = y_pred.cpu().detach().numpy().reshape(50, -1).T
        y_true = y_true.cpu().detach().numpy().reshape(50, -1).T
        unnorm_y_pred = scaler.inverse_transform(y_pred)
        unnorm_y_true = scaler.inverse_transform(y_true)
        mse = mean_squared_error(unnorm_y_true, unnorm_y_pred)
        return mse
    
    else:
        model = SFM(input_dim=input_dim,
                    hidden_dim=hidden_dim,
                    freq_dim=freq_dim,
                    output_dim=output_dim).to(device)
        # optimizer = RMSprop(model.parameters(), lr=lr)
        optimizer = Adam(model.parameters(), lr=lr, weight_decay=1e-4)
        criterion = nn.MSELoss()
        
        # best model selection
        best_mse = float('inf')
        best_model_path = f"SFM_weights/Exp1/best_model_step_{steps_ahead}.pth"
        best_model_state = None 
        
        # Model training
        with tqdm(total=epochs, desc=f"Step Size: {steps_ahead} Training") as pbar:
            for e in range(epochs):
                model.train()
                train_loss = 0.0
                for X_batch, y_batch in train_loader:
                    X_batch = X_batch.to(device)
                    y_batch = y_batch.to(device)
                    optimizer.zero_grad()
                    y_pred = model(X_batch)
                    loss = criterion(y_pred, y_batch)
                    loss.backward()
                    optimizer.step()
                    train_loss += loss.item()
                train_loss /= len(train_loader)

                # realtime test evaluation
                model.eval()
                with torch.no_grad():
                    y_pred = model(x_test.to(device))
                    y_pred = y_pred[:, -test_len:, :]
                    y_true = y_test[:, -test_len:, :]
                    y_pred = y_pred.cpu().detach().numpy().reshape(50, -1).T
                    y_true = y_true.cpu().detach().numpy().reshape(50, -1).T
                    unnorm_y_pred = scaler.inverse_transform(y_pred)
                    unnorm_y_true = scaler.inverse_transform(y_true)
                    mse = mean_squared_error(unnorm_y_true, unnorm_y_pred)
                if mse < best_mse:
                    best_mse = mse
                    best_model_state = model.state_dict()
                    torch.save(best_model_state, best_model_path)
                # Update progress bar
                pbar.set_description(f"Step: {steps_ahead}, Train Loss: {train_loss:0.5f}, Test MSE: {mse:0.4f}")
                pbar.update(1)
        return mse

def AR_Exp2(steps_ahead, lags):
    model = ARModel(steps_ahead, lags=lags)
    data = get_data()
    x_train, y_train, x_val, y_val, x_test, y_test, scaler  = scaled_data(data, steps_ahead)
    num_stocks = x_test.shape[0]
    test_len = x_test.shape[1] - x_val.shape[1]
    pred_data = np.zeros((num_stocks, test_len))

    for j in range(num_stocks):
        predictions = []
        history = x_test[j, :, 0].detach().numpy()
        for i in range(test_len):
            end = x_val.shape[1] +1 + i
            X = history[:end]
            pred = model.predict_next(X)
            predictions.append(pred)
        pred_data[j, :] = predictions
        
    y_true = y_test[:, y_val.shape[1]:,0].detach().numpy()
    pred_data_unscaled = scaler.inverse_transform(pred_data.T)
    true_data_unscaled = scaler.inverse_transform(y_true.T)
    mse = mean_squared_error(true_data_unscaled, pred_data_unscaled)
    return mse

def LSTM_Exp3(steps_ahead, hidden_dim, lr, epochs, device, load_model=False):
    input_dim = 1
    output_dim = 1
    num_layers = 1
    batch_size = 50
    
    # Load data
    data = get_data()
    x_train, y_train, x_val, y_val, x_test, y_test, scaler = scaled_data(data, steps_ahead)
    train_loader = DataLoader(TensorDataset(x_train, y_train), batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(TensorDataset(x_val, y_val), batch_size=batch_size, shuffle=False)
    val_len = x_val.shape[1] - x_train.shape[1]
    test_len = x_test.shape[1] - x_val.shape[1]
    
    if load_model:
        model = LSTM(input_dim=input_dim, 
                     hidden_dim=hidden_dim, 
                     output_dim=output_dim, 
                     num_layers=num_layers).to(device)
        model.load_state_dict(torch.load(f"LSTM_weights/Exp3/best_model_step_{steps_ahead}_hd_{hidden_dim}.pth"))

        # Model evaluation
        model.eval()
        with torch.no_grad():
            h0 = torch.zeros(num_layers, batch_size, hidden_dim).to(device)
            c0 = torch.zeros(num_layers, batch_size, hidden_dim).to(device)
            y_pred = model(x_test.to(device), h0, c0)
            y_pred = y_pred[:, -test_len:, :]
            y_test = y_test[:, -test_len:, :]
        y_pred = y_pred.cpu().detach().numpy().reshape(50, -1).T
        y_test = y_test.cpu().detach().numpy().reshape(50, -1).T
        unnorm_y_pred = scaler.inverse_transform(y_pred)
        unnorm_y_test = scaler.inverse_transform(y_test)
        mse = mean_squared_error(unnorm_y_test, unnorm_y_pred)
        return mse
    
    else : 
        model = LSTM(input_dim=input_dim, 
                     hidden_dim=hidden_dim, 
                     output_dim=output_dim, 
                     num_layers=num_layers).to(device)
        # optimizer = RMSprop(model.parameters(), lr=lr)
        optimizer = Adam(model.parameters(), lr=lr)
        criterion = nn.MSELoss()
        
        # Best_model selection
        best_val_loss = float('inf')
        best_model_state = None  # To save the best model's state_dict

        # Model training
        with tqdm(total=epochs, desc=f"Step Size: {steps_ahead} Training") as pbar:
            for e in range(epochs):
                model.train()
                train_loss = 0.0
                for X_batch, y_batch in train_loader:
                    X_batch = X_batch.to(device)
                    y_batch = y_batch.to(device)
                    batch_size = X_batch.size(0)
                    h0 = torch.zeros(num_layers, batch_size, hidden_dim).to(device)
                    c0 = torch.zeros(num_layers, batch_size, hidden_dim).to(device)
                    optimizer.zero_grad()
                    y_pred = model(X_batch, h0, c0)
                    loss = criterion(y_pred, y_batch)
                    loss.backward()
                    optimizer.step()
                    train_loss += loss.item()
                train_loss /= len(train_loader)

                # Validation
                model.eval()
                val_loss = 0.0
                with torch.no_grad():
                    for X_val, y_val in val_loader:
                        X_val = X_val.to(device)
                        y_val = y_val.to(device)
                        batch_size = X_val.size(0)
                        h0 = torch.zeros(num_layers, batch_size, hidden_dim).to(device)
                        c0 = torch.zeros(num_layers, batch_size, hidden_dim).to(device)
                        y_pred = model(X_val, h0, c0)
                        y_pred = y_pred[:, -val_len:, :]
                        y_val = y_val[:, -val_len:, :]
                        loss = criterion(y_pred, y_val)
                        val_loss += loss.item()
                val_loss /= len(val_loader)
                if val_loss < best_val_loss:
                    best_val_loss = val_loss
                    best_model_state = model.state_dict()

                # Update progress bar
                pbar.set_description(f"Step: {steps_ahead}, Train Loss: {round(train_loss, 5)}, Val Loss: {round(val_loss, 5)}")
                pbar.update(1)

        # Save the best model
        best_model_path = f"LSTM_weights/Exp3/best_model_step_{steps_ahead}_hd_{hidden_dim}.pth"
        torch.save(best_model_state, best_model_path)
        
        # Load the best model 
        model.load_state_dict(best_model_state)

        # Model evaluation
        model.eval()
        with torch.no_grad():
            h0 = torch.zeros(num_layers, batch_size, hidden_dim).to(device)
            c0 = torch.zeros(num_layers, batch_size, hidden_dim).to(device)
            y_pred = model(x_test.to(device), h0, c0)
            y_pred = y_pred[:, -test_len:, :]
            y_test = y_test[:, -test_len:, :]
        y_pred = y_pred.cpu().detach().numpy().reshape(50, -1).T
        y_test = y_test.cpu().detach().numpy().reshape(50, -1).T
        unnorm_y_pred = scaler.inverse_transform(y_pred)
        unnorm_y_test = scaler.inverse_transform(y_test)

        # Calculate Mean Squared Error
        mse = mean_squared_error(unnorm_y_test, unnorm_y_pred)
        return mse

def SFM_Exp4(steps_ahead, freq_dim, hidden_dim, lr, epochs, device, load_model=False):
    batch_size = 50
    input_dim = 1
    output_dim = 1

    # Load data
    data = get_data()
    x_train, y_train, x_val, y_val, x_test, y_test, scaler = scaled_data(data, steps_ahead)
    train_loader = DataLoader(TensorDataset(x_train, y_train), batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(TensorDataset(x_val, y_val), batch_size=batch_size, shuffle=False)
    val_len = x_val.shape[1] - x_train.shape[1]
    test_len = x_test.shape[1] - x_val.shape[1]
    
    if load_model:
        model = SFM(input_dim=input_dim,
                    hidden_dim=hidden_dim,
                    freq_dim=freq_dim,
                    output_dim=output_dim).to(device)
        model.load_state_dict(torch.load(f"SFM_weights/Exp4/best_model_step_{steps_ahead}_hd_{hidden_dim}.pth"))
        
        # Model evaluation
        model.eval()
        with torch.no_grad():
            y_pred = model(x_test.to(device))
            y_pred = y_pred[:, -test_len:, :]
            y_true = y_test[:, -test_len:, :]
        y_pred = y_pred.cpu().detach().numpy().reshape(50, -1).T
        y_true = y_true.cpu().detach().numpy().reshape(50, -1).T
        unnorm_y_pred = scaler.inverse_transform(y_pred)
        unnorm_y_true = scaler.inverse_transform(y_true)
        mse = mean_squared_error(unnorm_y_true, unnorm_y_pred)
        return mse
    
    else:
        model = SFM(input_dim=input_dim,
                    hidden_dim=hidden_dim,
                    freq_dim=freq_dim,
                    output_dim=output_dim).to(device)
        optimizer = Adam(model.parameters(), lr=lr)
        criterion = nn.MSELoss()
        
        # best model selection
        best_mse = float('inf')
        best_model_path = f"SFM_weights/Exp4/best_model_step_{steps_ahead}_hd_{hidden_dim}.pth"
        best_model_state = None 
        
        # Model training
        with tqdm(total=epochs, desc=f"Step Size: {steps_ahead} Training") as pbar:
            for e in range(epochs):
                model.train()
                train_loss = 0.0
                for X_batch, y_batch in train_loader:
                    X_batch = X_batch.to(device)
                    y_batch = y_batch.to(device)
                    optimizer.zero_grad()
                    y_pred = model(X_batch)
                    loss = criterion(y_pred, y_batch)
                    loss.backward()

                    nn_utils.clip_grad_norm_(model.parameters(), max_norm=1.0)  # Clip gradients to max norm of 1.0

                    optimizer.step()
                    train_loss += loss.item()
                train_loss /= len(train_loader)

                # realtime test evaluation
                model.eval()
                with torch.no_grad():
                    y_pred = model(x_test.to(device))
                    y_pred = y_pred[:, -test_len:, :]
                    y_true = y_test[:, -test_len:, :]
                    y_pred = y_pred.cpu().detach().numpy().reshape(50, -1).T
                    y_true = y_true.cpu().detach().numpy().reshape(50, -1).T
                    unnorm_y_pred = scaler.inverse_transform(y_pred)
                    unnorm_y_true = scaler.inverse_transform(y_true)
                    mse = mean_squared_error(unnorm_y_true, unnorm_y_pred)
                if mse < best_mse:
                    best_mse = mse
                    best_model_state = model.state_dict()
                    torch.save(best_model_state, best_model_path)
                # Update progress bar
                pbar.set_description(f"Step: {steps_ahead}, Train Loss: {train_loss:0.5f}, Test MSE: {mse:0.4f}")
                pbar.update(1)
        return mse

def SFM_Exp5(steps_ahead, freq_dim, hidden_dim, lr, epochs, device, load_model=False):
    batch_size = 50
    input_dim = 1
    output_dim = 1

    # Load data
    data = get_data()
    x_train, y_train, x_val, y_val, x_test, y_test, scaler = scaled_data(data, steps_ahead)
    train_loader = DataLoader(TensorDataset(x_train, y_train), batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(TensorDataset(x_val, y_val), batch_size=batch_size, shuffle=False)
    val_len = x_val.shape[1] - x_train.shape[1]
    test_len = x_test.shape[1] - x_val.shape[1]
    
    if load_model:
        model = SFM(input_dim=input_dim,
                    hidden_dim=hidden_dim,
                    freq_dim=freq_dim,
                    output_dim=output_dim).to(device)
        model.load_state_dict(torch.load(f"SFM_weights/Exp5/best_model_step_{steps_ahead}_fd_{freq_dim}.pth"))
        
        # Model evaluation
        model.eval()
        with torch.no_grad():
            y_pred = model(x_test.to(device))
            y_pred = y_pred[:, -test_len:, :]
            y_true = y_test[:, -test_len:, :]
        y_pred = y_pred.cpu().detach().numpy().reshape(50, -1).T
        y_true = y_true.cpu().detach().numpy().reshape(50, -1).T
        unnorm_y_pred = scaler.inverse_transform(y_pred)
        unnorm_y_true = scaler.inverse_transform(y_true)
        mse = mean_squared_error(unnorm_y_true, unnorm_y_pred)
        return mse
    
    else:
        model = SFM(input_dim=input_dim,
                    hidden_dim=hidden_dim,
                    freq_dim=freq_dim,
                    output_dim=output_dim).to(device)
        optimizer = Adam(model.parameters(), lr=lr)
        criterion = nn.MSELoss()
        
        # best model selection
        best_mse = float('inf')
        best_model_path = f"SFM_weights/Exp5/best_model_step_{steps_ahead}_fd_{freq_dim}.pth"
        best_model_state = None 
        
        # Model training
        with tqdm(total=epochs, desc=f"Step Size: {steps_ahead} Training") as pbar:
            for e in range(epochs):
                model.train()
                train_loss = 0.0
                for X_batch, y_batch in train_loader:
                    X_batch = X_batch.to(device)
                    y_batch = y_batch.to(device)
                    optimizer.zero_grad()
                    y_pred = model(X_batch)
                    loss = criterion(y_pred, y_batch)
                    loss.backward()

                    nn_utils.clip_grad_norm_(model.parameters(), max_norm=1.0)  # Clip gradients to max norm of 1.0

                    optimizer.step()
                    train_loss += loss.item()
                train_loss /= len(train_loader)

                # realtime test evaluation
                model.eval()
                with torch.no_grad():
                    y_pred = model(x_test.to(device))
                    y_pred = y_pred[:, -test_len:, :]
                    y_true = y_test[:, -test_len:, :]
                    y_pred = y_pred.cpu().detach().numpy().reshape(50, -1).T
                    y_true = y_true.cpu().detach().numpy().reshape(50, -1).T
                    unnorm_y_pred = scaler.inverse_transform(y_pred)
                    unnorm_y_true = scaler.inverse_transform(y_true)
                    mse = mean_squared_error(unnorm_y_true, unnorm_y_pred)
                if mse < best_mse:
                    best_mse = mse
                    best_model_state = model.state_dict()
                    torch.save(best_model_state, best_model_path)
                # Update progress bar
                pbar.set_description(f"Step: {steps_ahead}, Train Loss: {train_loss:0.5f}, Test MSE: {mse:0.4f}")
                pbar.update(1)
        return mse

