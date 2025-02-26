import pandas as pd

# Load the CSV file
file_path = "aapl_2021_2023.csv"
df = pd.read_csv(file_path)

# Convert quote dates to numbers starting from 1
quote_dates = pd.to_datetime(df['QUOTE_DATE'])
df['QUOTE_DATE'] = (quote_dates - quote_dates.iloc[-1]).dt.days + 1

# Convert expiry dates to numbers starting from 1
expire_dates = pd.to_datetime(df['EXPIRE_DATE'])
df['EXPIRE_DATE'] = (expire_dates - expire_dates.min()).dt.days + 1

# Convert two-sided open interest of calls to numbers 
opint_calls = df['C_SIZE'].str.extract(r'(?P<C_OPINT_BUY>\d).*(?P<C_OPINT_SELL>\d)')
df = pd.concat([df, opint_calls], axis=1)

# Convert two-sided open interest of puts to numbers 
opint_puts = df['P_SIZE'].str.extract(r'(?P<P_OPINT_BUY>\d).*(?P<P_OPINT_SELL>\d)')
df = pd.concat([df, opint_puts], axis=1)

# Delete the 'C_SIZE' and 'P_SIZE' columns
df = df.drop(columns=['C_SIZE', 'P_SIZE'])

# Replace all entries with nothing or space with zero
df = df.replace(' ', 0)
df = df.replace('', 0)

# Cast all entries to float32 type
df = df.astype('float32')

# Standardize the data
df = (df - df.mean())/df.std()

# Inspect column names
#print(df.columns)

# Define grouping columns
#grouping_columns = ["EXPIRE_UNIX", "STRIKE",]

# Group data by (ExpirationDate, StrikePrice)
#option_groups = df.groupby(grouping_columns)

# Create a dictionary to store DataFrames
#option_dataframes = {group: data.sort_values("QUOTE_UNIXTIME") for group, data in option_groups}

# Display the first few keys (option contract identifiers)
#print("Contracts stored:", list(option_dataframes.keys())[:5])

# Print each DataFrame in full
#for group_key, dataframe in option_dataframes.items():
   #print(f"\n===== Option Data - Expiry: {group_key[0]}, Strike: {group_key[1]} =====")
   #print(dataframe.to_string(index=False))  # Print full DataFrame without index

# Print the shape of each DataFrame - some are much bigger than others
#for group_key, dataframe in option_dataframes.items():
    #print(dataframe.shape)
