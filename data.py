import pandas as pd

# Load the CSV file
file_path = "aapl_2021_2023.csv"
df = pd.read_csv(file_path)

# Inspect column names
print(df.columns)

# Define grouping columns
grouping_columns = ["EXPIRE_UNIX", "STRIKE",]

# Group data by (ExpirationDate, StrikePrice)
option_groups = df.groupby(grouping_columns)

# Create a dictionary to store DataFrames
option_dataframes = {group: data.sort_values("QUOTE_UNIXTIME") for group, data in option_groups}

# Display the first few keys (option contract identifiers)
print("Contracts stored:", list(option_dataframes.keys())[:5])

# Print each DataFrame in full
for group_key, dataframe in option_dataframes.items():
    print(f"\n===== Option Data - Expiry: {group_key[0]}, Strike: {group_key[1]} =====")
    print(dataframe.to_string(index=False))  # Print full DataFrame without index

# Print the shape of each DataFrame - some are much bigger than others
for group_key, dataframe in option_dataframes.items():
    print(dataframe.shape)