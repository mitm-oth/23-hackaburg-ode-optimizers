from pathlib import Path
from pprint import pprint
from dataclasses import dataclass

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

#def sort_names(path):
#    return int(path.with_suffix('').name)
#root = Path("./measurements")
#paths = sorted(root.rglob("*.parquet"), key=sort_names)

def get_sorted_paths(root, glob):
    paths = list(Path(root).rglob(glob))
    paths.sort()
    paths.sort(key=lambda p: len(str(p)))
    return paths

@dataclass
class Data:
    name: str
    start: pd.Timestamp
    end: pd.Timestamp
    elapsed: pd.Timedelta

def get_intervals(paths):
    xs = []
    for path in paths:
        df = pd.read_parquet(path)
        df["rtctime"] = pd.to_datetime(df["rtctime"], unit="ms")
        #df.set_index("rtctime", inplace=True)
        start = df.iloc[0]["rtctime"]
        end = df.iloc[-1]["rtctime"]
        xs.append(Data(path.name, start, end, end-start))
    return pd.DataFrame(xs)

def plot_intervals(df):
    for i, row in df.iterrows():
        filenum = int(Path(row["name"]).with_suffix("").name)
        plt.plot([row["start"], row["end"]], [filenum-1, filenum])
    plt.title("Data coverage")
    plt.xlabel("Time")
    plt.ylabel("Parquet file number")
    #plt.savefig("coverage.pdf")
    plt.show()

def time_deltas(paths):
    for path in paths:
        df = pd.read_parquet(path)

if __name__ == "__main__":
    paths = get_sorted_paths("./data", "*.parquet")
    #pprint(paths)
    df = get_intervals(paths)
    print(df)
    #print(len(df))
    plot_intervals(df)

