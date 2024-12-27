import torch
import numpy as np
import torch.nn as nn
from statsmodels.tsa.ar_model import AutoReg

class ARModel():
    def __init__(self, steps_ahead, lags):
        self.steps_ahead = steps_ahead
        self.lags = lags
    
    def predict_next(self, X):
        model = AutoReg(X, lags=self.lags)
        model_fit = model.fit()
        target_index = len(X) - 1 + self.steps_ahead
        pred = model_fit.predict(start=target_index, end=target_index, dynamic=False)
        return pred[-1]
        
class LSTM(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, num_layers):
        super().__init__()
        self.lstm = nn.LSTM(input_dim, hidden_dim, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_dim, output_dim)
        self.reset_parameters()

    def reset_parameters(self):
        for name, param in self.lstm.named_parameters():
            if 'weight_ih' in name:
                nn.init.xavier_uniform_(param)
            elif 'weight_hh' in name:
                nn.init.orthogonal_(param)
            elif 'bias' in name:
                nn.init.zeros_(param)
        nn.init.xavier_uniform_(self.fc.weight)
        nn.init.zeros_(self.fc.bias)

    def forward(self, x, h0, c0):
        out, _ = self.lstm(x, (h0, c0))
        out = self.fc(out[:, :, :])
        return out
    
class SFMCell(nn.Module):
    def __init__(self, input_dim, hidden_dim, freq_dim, output_dim):
        super(SFMCell, self).__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.freq_dim = freq_dim
        self.output_dim = output_dim
        
        # Weights for gates and inputs
        self.W_i = nn.Linear(input_dim, hidden_dim)
        self.U_i = nn.Linear(hidden_dim, hidden_dim, bias=False)

        self.W_ste = nn.Linear(input_dim, hidden_dim)
        self.U_ste = nn.Linear(hidden_dim, hidden_dim, bias=False)

        self.W_fre = nn.Linear(input_dim, freq_dim)
        self.U_fre = nn.Linear(hidden_dim, freq_dim, bias=False)

        self.W_c = nn.Linear(input_dim, hidden_dim)
        self.U_c = nn.Linear(hidden_dim, hidden_dim, bias=False)

        self.W_o = nn.Linear(input_dim, hidden_dim)
        self.U_o = nn.Linear(hidden_dim, hidden_dim, bias=False)

        self.U_a = nn.Linear(freq_dim, 1, bias=False)
        self.b_a = nn.Parameter(torch.zeros(hidden_dim))

        self.W_p = nn.Linear(hidden_dim, output_dim)
        self.b_p = nn.Parameter(torch.zeros(output_dim))
        
    def forward(self, x, states):
        p_tm1, h_tm1, S_re_tm1, S_im_tm1, time_tm1 = states
        time_t = time_tm1 + 1
        
        # Input gate
        # Equation (15)
        i = torch.sigmoid(self.W_i(x) + self.U_i(h_tm1))
        
        # State-frequency forget gates
        # Equation (12) and (13)
        ste = torch.sigmoid(self.W_ste(x) + self.U_ste(h_tm1))
        fre = torch.sigmoid(self.W_fre(x) + self.U_fre(h_tm1))
        
        # Forget gate combination
        # Equation (14)
        f = ste.unsqueeze(-1) * fre.unsqueeze(-2)
        
        # Candidate cell state
        # Equation (16)
        c = i * torch.tanh(self.W_c(x) + self.U_c(h_tm1))
        
        # Frequency dynamics
        omega = 2 * np.pi * time_t.unsqueeze(-1).unsqueeze(-1)
        omega = omega * torch.arange(self.freq_dim).to(x.device).unsqueeze(0)
        re = torch.cos(omega)
        im = torch.sin(omega)

        # Equation (8) and (9)
        S_re = f * S_re_tm1 + c.unsqueeze(-1) * re
        S_im = f * S_im_tm1 + c.unsqueeze(-1) * im

        # Amplitude calculation
        # Equation (10)
        A = torch.sqrt(S_re ** 2 + S_im ** 2)
        
        # Equation (17)
        A_a = torch.tanh(self.U_a(A).squeeze(-1) + self.b_a)
        
        # Output gate
        # Equation (18)
        o = torch.sigmoid(self.W_o(x) + self.U_o(h_tm1))
        h = o * A_a
        
        # Final output
        # Equation (19)
        p = self.W_p(h) + self.b_p
        
        return p, (p, h, S_re, S_im, time_t)

class SFM(nn.Module):
    def __init__(self, input_dim, hidden_dim, freq_dim, output_dim):
        super(SFM, self).__init__()
        self.cell = SFMCell(input_dim, hidden_dim, freq_dim, output_dim)
    
    def forward(self, x):
        batch_size, seq_len, _ = x.shape
        device = x.device
        
        # Initial states
        p_0 = torch.zeros(batch_size, self.cell.output_dim).to(device)
        h_0 = torch.zeros(batch_size, self.cell.hidden_dim).to(device)
        S_re_0 = torch.zeros(batch_size, self.cell.hidden_dim, self.cell.freq_dim).to(device)
        S_im_0 = torch.zeros(batch_size, self.cell.hidden_dim, self.cell.freq_dim).to(device)
        time_0 = torch.zeros(batch_size).to(device)
        
        states = (p_0, h_0, S_re_0, S_im_0, time_0)
        outputs = []

        for t in range(seq_len):
            xt = x[:, t, :]
            p_t, states = self.cell(xt, states)
            outputs.append(p_t)
        
        # Equation (21)
        outputs = torch.stack(outputs, dim=1)
        return outputs