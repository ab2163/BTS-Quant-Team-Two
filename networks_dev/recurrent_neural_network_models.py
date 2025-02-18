import torch
from torch import nn

### Recurrent Hidden Layer Module Definitions

# Vanilla Recurrent Hidden Layer Module
class VanillaRNN(nn.Module):
    """
    Vanilla RNN Layer.

    Parameters:
        input_size: int, number of input features.
        hidden_size: int, number of hidden units.

    Inputs:
        x: tensor of shape (seq_len, batch_size, input_size), the input sequence.
        hidden: tensor of shape (batch_size, hidden_size), the initial hidden state.
            If None, hidden is initialised to zeros.

    Outputs:
        output: tensor of shape (seq_len, batch_size, hidden_size), hidden states over all timesteps.
        hidden: tensor of shape (batch_size, hidden_size), the final hidden state.
    """
    def __init__(self, input_size, hidden_size):
        super().__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.input2h = nn.Linear(input_size,hidden_size)
        self.h2h = nn.Linear(hidden_size,hidden_size)
        self.activation = nn.Tanh()
    def init_hidden(self, batch_size, device):
        return torch.zeros(batch_size, self.hidden_size).to(device)

    def forward(self, x, hidden=None):

        seq_len, batch_size, _ = x.size()
        if hidden is None:
            hidden = self.init_hidden(batch_size, x.device)

        output = []
        for t in range(seq_len):
            hidden = self.activation(self.input2h(x[t]) + self.h2h(hidden))
            output.append(hidden)

        output = torch.stack(output, dim=0)
        return output, hidden

## Gated Recurrent Unit Hidden Layer Module

class GRULayer(nn.Module):
    """GRU Layer."""
    def __init__(self, input_size, hidden_size):
        super().__init__()
        self.hidden_size = hidden_size
        self.input2z = nn.Linear(input_size, hidden_size)  # Update Gate
        self.h2z = nn.Linear(hidden_size, hidden_size)
        self.input2r = nn.Linear(input_size, hidden_size)  # Reset Gate
        self.h2r = nn.Linear(hidden_size, hidden_size)
        self.input2h = nn.Linear(input_size, hidden_size)  # Candidate Hidden State
        self.h2h = nn.Linear(hidden_size, hidden_size)

    def init_hidden(self, batch_size, device):
        return torch.zeros(batch_size, self.hidden_size).to(device)

    def forward(self, x, hidden=None):
        seq_len, batch_size, _ = x.size()
        if hidden is None:
            hidden = self.init_hidden(batch_size, x.device)

        outputs = []
        for t in range(seq_len):
            z_t = torch.sigmoid(self.input2z(x[t]) + self.h2z(hidden))
            r_t = torch.sigmoid(self.input2r(x[t]) + self.h2r(hidden))
            h_tilde = torch.tanh(self.input2h(x[t]) + self.h2h(r_t * hidden))
            hidden = (1 - z_t) * hidden + z_t * h_tilde
            outputs.append(hidden)

        outputs = torch.stack(outputs, dim=0)
        return outputs, hidden

class LightGRULayer(nn.Module):
    """
    Light GRU Layer.

    This is a simplified version of the GRU (Gated Recurrent Unit) that eliminates
    the reset gate, making it computationally lighter. The recurrence dynamics are
    governed by the update gate and candidate hidden state equations.

    Parameters:
        input_size (int): Dimensionality of the input features.
        hidden_size (int): Number of hidden units in the recurrent layer.

    Inputs:
        x (torch.Tensor): Input tensor of shape (seq_len, batch_size, input_size),
                          where `seq_len` is the sequence length, `batch_size` is
                          the number of samples in the batch, and `input_size` is
                          the dimensionality of the input.
        hidden (torch.Tensor, optional): Tensor of shape (batch_size, hidden_size)
                                         representing the initial hidden state. If
                                         None, it is initialized to zeros.

    Outputs:
        output (torch.Tensor): Tensor of shape (seq_len, batch_size, hidden_size)
                               containing the hidden states for each timestep.
        hidden (torch.Tensor): Tensor of shape (batch_size, hidden_size) containing
                               the final hidden state after processing the entire
                               sequence.

    Recurrence Equations:
        Update Gate:          z_t = sigmoid(W_z x_t + U_z h_{t-1} + b_z)
        Candidate Hidden State:  h_tilde_t = tanh(W_h x_t + U_h h_{t-1} + b_h)
        Hidden State Update:  h_t = (1 - z_t) * h_{t-1} + z_t * h_tilde_t
    """
    def __init__(self, input_size, hidden_size):
        super().__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.input2z = nn.Linear(input_size, hidden_size)  # Update Gate
        self.h2z = nn.Linear(hidden_size, hidden_size)
        self.input2h = nn.Linear(input_size, hidden_size)  # Candidate Hidden State
        self.h2h = nn.Linear(hidden_size, hidden_size)

    def init_hidden(self, batch_size, device):
        return torch.zeros(batch_size, self.hidden_size).to(device)

    def forward(self, x, hidden=None):
        seq_len, batch_size, _ = x.size()
        if hidden is None:
            hidden = self.init_hidden(batch_size, x.device)

        outputs = []
        for t in range(seq_len):
            z_t = torch.sigmoid(self.input2z(x[t]) + self.h2z(hidden))
            h_tilde = torch.tanh(self.input2h(x[t]) + self.h2h(hidden))
            hidden = (1 - z_t) * hidden + z_t * h_tilde
            outputs.append(hidden)

        outputs = torch.stack(outputs, dim=0)
        return outputs, hidden



class LSTMLayer(nn.Module):
    """
    LSTM Layer.

    This implementation of the Long Short-Term Memory (LSTM) layer explicitly
    defines its gating mechanisms. LSTMs are designed to handle long-term dependencies
    in sequential data through the use of a cell state and three gates: forget, input,
    and output.

    Parameters:
        input_size (int): Dimensionality of the input features.
        hidden_size (int): Number of hidden units in the recurrent layer.

    Inputs:
        x (torch.Tensor): Input tensor of shape (seq_len, batch_size, input_size),
                          where `seq_len` is the sequence length, `batch_size` is
                          the number of samples in the batch, and `input_size` is
                          the dimensionality of the input.
        hidden (tuple[torch.Tensor, torch.Tensor], optional): A tuple containing:
            - h (torch.Tensor): Tensor of shape (batch_size, hidden_size), the
                                initial hidden state. Defaults to zeros if not
                                provided.
            - c (torch.Tensor): Tensor of shape (batch_size, hidden_size), the
                                initial cell state. Defaults to zeros if not
                                provided.

    Outputs:
        output (torch.Tensor): Tensor of shape (seq_len, batch_size, hidden_size)
                               containing the hidden states for each timestep.
        hidden (tuple[torch.Tensor, torch.Tensor]): A tuple containing:
            - h (torch.Tensor): Tensor of shape (batch_size, hidden_size), the
                                final hidden state after processing the sequence.
            - c (torch.Tensor): Tensor of shape (batch_size, hidden_size), the
                                final cell state after processing the sequence.

    Recurrence Equations:
        Forget Gate:         f_t = sigmoid(W_f x_t + U_f h_{t-1} + b_f)
        Input Gate:          i_t = sigmoid(W_i x_t + U_i h_{t-1} + b_i)
        Output Gate:         o_t = sigmoid(W_o x_t + U_o h_{t-1} + b_o)
        Candidate Cell State: c_tilde_t = tanh(W_c x_t + U_c h_{t-1} + b_c)
        Cell State Update:   c_t = f_t * c_{t-1} + i_t * c_tilde_t
        Hidden State Update: h_t = o_t * tanh(c_t)
    """

    def __init__(self, input_size, hidden_size):
        super().__init__()
        self.hidden_size = hidden_size
        self.input2f = nn.Linear(input_size, hidden_size)  # Forget Gate
        self.h2f = nn.Linear(hidden_size, hidden_size)
        self.input2i = nn.Linear(input_size, hidden_size)  # Input Gate
        self.h2i = nn.Linear(hidden_size, hidden_size)
        self.input2o = nn.Linear(input_size, hidden_size)  # Output Gate
        self.h2o = nn.Linear(hidden_size, hidden_size)
        self.input2c = nn.Linear(input_size, hidden_size)  # Candidate Cell State
        self.h2c = nn.Linear(hidden_size, hidden_size)

    def init_hidden(self, batch_size, device):
        h = torch.zeros(batch_size, self.hidden_size).to(device)
        c = torch.zeros(batch_size, self.hidden_size).to(device)
        return h, c

    def forward(self, x, hidden=None):
        seq_len, batch_size, _ = x.size()
        if hidden is None:
            hidden = self.init_hidden(batch_size, x.device)

        h, c = hidden
        outputs = []
        for t in range(seq_len):
            f_t = torch.sigmoid(self.input2f(x[t]) + self.h2f(h))
            i_t = torch.sigmoid(self.input2i(x[t]) + self.h2i(h))
            o_t = torch.sigmoid(self.input2o(x[t]) + self.h2o(h))
            c_tilde = torch.tanh(self.input2c(x[t]) + self.h2c(h))
            c = f_t * c + i_t * c_tilde
            h = o_t * torch.tanh(c)
            outputs.append(h)

        outputs = torch.stack(outputs, dim=0)
        return outputs, (h, c)

### Recurrent Neural Network Model Implementations
#### Sets up the full Neural Network with input and output layers using Recurrent Hidden Layers via predefined (^) modules

# Example on how to implement submodules
class RecurrentNetworkModel(nn.Module):
    """
    Flexible recurrent network model with support for different recurrent layer types
    (VanillaRNN, GRU, LightGRU, LSTM) and customizable depth.
    """

    def __init__(self, input_size, hidden_size, output_size, num_layers,
                 recurrent_module='VanillaRNN', hidden_activation_function=None):
        super().__init__()

        # Store network parameters
        self.num_layers = num_layers
        self.hidden_activation_function = hidden_activation_function if hidden_activation_function is not None else nn.Identity()
        self.recurrent_module = recurrent_module

        # Create a list to store the recurrent layers
        self.rnn_layers = nn.ModuleList()

        # Define the input layer
        if recurrent_module == 'VanillaRNN':
            self.rnn_layers.append(VanillaRNN(input_size, hidden_size))
        elif recurrent_module == 'GRULayer':
            self.rnn_layers.append(GRULayer(input_size, hidden_size))
        elif recurrent_module == 'LightGRULayer':
            self.rnn_layers.append(LightGRULayer(input_size, hidden_size))
        elif recurrent_module == 'LSTMLayer':
            self.rnn_layers.append(LSTMLayer(input_size, hidden_size))
        else:
            raise ValueError("Invalid recurrent module specified.")

        # Define the remaining layers with homogeneous architecture
        for _ in range(1, num_layers):
            if recurrent_module == 'VanillaRNN':
                self.rnn_layers.append(VanillaRNN(hidden_size, hidden_size))
            elif recurrent_module == 'GRULayer':
                self.rnn_layers.append(GRULayer(hidden_size, hidden_size))
            elif recurrent_module == 'LightGRULayer':
                self.rnn_layers.append(LightGRULayer(hidden_size, hidden_size))
            elif recurrent_module == 'LSTMLayer':
                self.rnn_layers.append(LSTMLayer(hidden_size, hidden_size))

        # Fully connected output layer
        self.fc = nn.Linear(hidden_size, output_size)

    def forward(self, x, states=None):
        """
        Forward pass through the entire network.

        Inputs:
            x: Tensor of shape (seq_len, batch_size, input_size), input sequence.
            states: List of hidden states for each layer. If None, will initialize.

        Outputs:
            outputs: Tensor of shape (seq_len, batch_size, output_size), output sequence.
            final_states: List of final hidden (and optionally cell) states for each layer.
        """
        # Initialize hidden states if not provided
        if states is None:
            states = [None] * self.num_layers

        layer_output = x  # Initialize input to the first layer
        hidden_layer_activity_pre_act = []
        hidden_layer_activity_post_act = []

        # Forward pass through each layer
        for i, rnn_layer in enumerate(self.rnn_layers):

            layer_output_pre_act, _ = rnn_layer(layer_output, states[i])

            # Apply hidden activation function to the full sequence output
            layer_output = self.hidden_activation_function(layer_output_pre_act)

            hidden_layer_activity_pre_act.append(layer_output_pre_act)

            hidden_layer_activity_post_act.append(layer_output)


        # Stack hidden activities along the first dimension for `num_hidden_layers`
        hidden_activity_pre_act_tensor = torch.stack(hidden_layer_activity_pre_act, dim=0)  # Shape: [num_hidden_layers, seq_len, batch_size, hidden_size]
        hidden_activity_post_act_tensor = torch.stack(hidden_layer_activity_post_act, dim=0)  # Shape: [num_hidden_layers, seq_len, batch_size, hidden_size]

        # Apply the fully connected layer to the sequence output from the last RNN layer
        outputs = self.fc(layer_output)  # Shape: (seq_len, batch_size, output_size)

        return outputs, (hidden_activity_post_act_tensor,hidden_activity_pre_act_tensor)

