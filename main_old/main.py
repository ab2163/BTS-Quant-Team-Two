import ann_model
import rnn_model
import lstm_model
import train_models
import black_sch_data
import torch
from torch import nn
from torch.utils.data import DataLoader
import sys

# Training parameters
LEARN_RATE_ANN = 1e-4
BATCH_SIZE = 64
EPOCHS = 100000
ACCURACY = 0.01

# Setup the models
model_ann = ann_model.ArtificialNeuralNetwork()
model_ann.double()
train_data_ann = black_sch_data.BlackSchDataset(training_set=True)
test_data_ann = black_sch_data.BlackSchDataset(training_set=False)
train_dataloader_ann = DataLoader(train_data_ann, batch_size=BATCH_SIZE)
test_dataloader_ann = DataLoader(test_data_ann, batch_size=BATCH_SIZE)
loss_fn_ann = nn.MSELoss()
optimizer_ann = torch.optim.Adam(model_ann.parameters(), lr=LEARN_RATE_ANN)

# (Similar code here to set up the RNN and LSTM models)

# Train the models
for t in range(EPOCHS):

    # Carry out training
    train_models.train_loop(train_dataloader_ann, model_ann, loss_fn_ann, optimizer_ann)

    # Every 200 epochs print the model accuracy and save model
    if (t+1) % 200 == 0:
        print(f"Epoch {t+1}\n-------------------------------")
        acc_ann = train_models.test_loop(test_dataloader_ann, model_ann, loss_fn_ann, ACCURACY)
        sys.stdout.flush()
        torch.save(model_ann.state_dict(), 'trained_ann.pt')

# Completion message
print("Training Complete")