import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
import os

def get_sorted_paths(root="../data", glob="*.parquet"):
    paths = list(Path(root).rglob(glob))
    paths.sort()
    paths.sort(key=lambda p: len(str(p)))
    return paths

# Print min and max difference between consecutive timestamps
def min_max(paths):
    for path in paths:
        #df = pd.read_parquet("./measurements/36.parquet")
        df = pd.read_parquet(path)
        timestamps = np.array(df["rtctime"])
        t = timestamps[1:]
        s = timestamps[:-1]
        deltas = t-s
        print(path, np.min(deltas), "...", np.max(deltas))

# Print all pdf file names, sorted and seperated by spaces, to easily
# use the result with 'pdfunite'
def all_file_names(paths):
    ps = []
    for path in paths:
        p = Path("./plots/") / (path.name + ".pdf")
        ps.append(str(p))
        #print(p)
    return ' '.join(ps)

def plot_one(path, save=False, show=False):
    path = Path(path)
    df = pd.read_parquet(path)
    df["rtctime"] = pd.to_datetime(df["rtctime"], unit="ms")
    print(path)
    df.plot(x="rtctime", y=["target_temperature", "ambient_temp", "feature_ct", "car_speed", "soc", "lat"]) 
    plt.title(path)
    savepath = Path("./plots/") / (path.name + ".pdf")
    if save:
        plt.savefig(savepath)
    if show:
        plt.show()
    plt.close()


def plot_all(paths):
    print("plot all!")
    for path in paths:
        plot_one(path)

def make_unite(replot=True, out_file="./all_parquets.pdf"):
    paths = get_sorted_paths()
    if replot:
        plot_all(paths)
    unite_cmd = "pdfunite " + all_file_names(paths) + ' ' + out_file
    os.system(unite_cmd)

if __name__ == "__main__":
    #min_max(paths, dfs)
    #plot_all(paths, dfs)

    #make_unite(replot=False)
    plot_one("../data/58.parquet", show=True)
