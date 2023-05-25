# import required packages
import pandas as pd

# path to the data directory
DATAPATH = "data"


# function to read the data into a pandas dataframe
def read_data(skip=None):
    df = pd.read_parquet(DATAPATH)

    if skip is not None:
        df = df.iloc[0:len(df):skip]

    return df


# test function read_data()
if __name__ == "__main__":
    df = read_data(skip=1000)
    print(df.shape)
