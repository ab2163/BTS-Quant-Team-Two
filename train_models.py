import torch

# Function which executes one training epoch
def train_loop(dataloader, model, loss_fn, optimizer):
    size = len(dataloader.dataset)

    # Set the model to training mode
    model.train()

    for batch, (X, y) in enumerate(dataloader):
        # Compute prediction and loss
        pred = model(X)
        loss = loss_fn(pred, y)

        # Backpropagation
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()

# Function which evaluates current accuracy of model
def test_loop(dataloader, model, loss_fn, test_acc):
    # Set the model to evaluation mode
    model.eval()
    size = len(dataloader.dataset)
    num_batches = len(dataloader)
    test_loss, correct = 0, 0

    with torch.no_grad():
        for X, y in dataloader:
            pred = model(X)

            # Add up loss function values for each batch
            test_loss += loss_fn(pred, y).item() 

            # Add the number of correct estimations for each batch
            acc_low = 1 - test_acc
            acc_upp = 1 + test_acc
            correct += sum([1 if (acc_low*yval < mval < acc_upp*yval) else 0 for (yval, mval) in zip(y, pred)])

    # Calculate average loss per batch
    test_loss /= num_batches

    # Calculate percentage of correct predictions
    correct /= size

    print(f"Test Error: \nAccuracy: {(100*correct):>0.1f}%, Avg loss: {test_loss:>8f} \n")