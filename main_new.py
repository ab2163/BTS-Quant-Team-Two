'''

Currently a pseudo code outline of how the main file should perform the modelling process
Rough outline not necessarily what the final implementation will be

'''

# DATA PIPELINE STUFF:

### Existing Data Import
# Import options pricing data from csv from WRDS etc
# Import Underlying Asset Price data from api or csv
#   At the same frequency and points in time for our options pricing data


### Data Extraction
# Extract individual time steps for entire population -> vectors of existing data
# Group by unique strike price and expiration date -> groups of vectors representing the " same option contract"
    ## Point of interest : EDA : Number of timesteps for each of these sets. how long are the sequences , how much does this vary

### Feature Engineering and Preprocessing:
## Engineering:
    ## using the Underlying Asset Price Data:
        ## Add UAP data for each time step in sequence (that days market data (price etc))
    ## Greeks
        ## If not already in dataset:
            ## Do we want to calculate the greeks . If so how do we do this
    ## Any other features of interest ...
## Preprocessing (NN suitability specific)
    ## Normalisation and Scaling
        ## Select and Implement a Normalisation//Scaling method for Strike Price, Underlying Asset Price, etc
            ### And any other features which are subject to inter-sequence scaling trends
                ### e.g. when we think about the scale of the Pricing related data for
                       #### a sequence of 30 days of data for an option from 2010 vs 2022 :
        ## Implement any ratio / indexing required on features

## Data Construction
   ## Matrix Construction
    ### Aggregating data and packing into a matrix for each sequence

### MORE INTO TRAINING REGIME

## Preprocessing (NN training regime specific)
    ## Train Eval Test Split Strategy
        ## Stratify by what variables
    ## Tensor Construction for batches
        ## stratify by what variables

## Network Architecture Definition
     ### Make a bunch of different networks (diff base architectures, diff number of layers, diff hidden sizes, diff activation functions)
     ### Diff Output Sizes (decide on predicting (Bid , Ask) ((Output size = 2)) or (Bid+Ask/2) (Output size = 1)

## Training
    ## tuning and optimising the training regime
    ## with a solid training regime:
        ## Train + Evaluate -> Model HyperParameter Tuning
        ## -> Best model from eval
## Performance Evaluation and Testing
    ## Construct error / performance metrics with application (financial markets / trading) context
    ## Execute the tests and obtain results
        ## Includes (if we want: learning and training info) , (accuracy performance, regions (in terms of data / market conditions) of strength and weakness
        ### Extra if time (RNN analysis including hidden layer activity and forming conclusions about how the networks operate to make predictions)


