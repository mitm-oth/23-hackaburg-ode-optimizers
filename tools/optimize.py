import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy import integrate, interpolate, optimize
from pathlib import Path
import sys

def get_sorted_paths(root="../data", glob="*.parquet"):
    paths = list(Path(root).rglob(glob))
    paths.sort()
    paths.sort(key=lambda p: len(str(p)))
    return paths

path = Path("../data/58.parquet")
df = pd.read_parquet(path)

time = df["rtctime"].to_numpy()
target = df["target_temperature"].to_numpy()
ambient = df["ambient_temp"].to_numpy()
current = df["feature_c"].to_numpy()



f_target = interpolate.interp1d(time, target)
f_ambient = interpolate.interp1d(time, ambient)
f_current = interpolate.interp1d(time, current)

def heat_transfer(t, y, coeff0, coeff1):
    return coeff0 * (f_ambient(t) - y) + coeff1 * f_current(t)

def err_sq(t, predicted):
    return (f_target(t) - predicted(t))**2

def cost(coeff):
    t_span = [time[0], time[-1]]
    sol = integrate.solve_ivp(heat_transfer, t_span, [target[0]], method="Radau",
            dense_output=True, vectorized=True, args=coeff)
    norm, err = integrate.quad(err_sq, t_span[0], t_span[1], args=sol.sol)
    return norm

def run():
    initial_coeff = np.array([1., 1.])
    print(initial_coeff)
    print(initial_coeff.shape)
    res = optimize.minimize(cost, initial_coeff)
    print(res)


run()


sys.exit()


@np.vectorize
def ambient(t):
    return 20

@np.vectorize
def current(t):
    if t < 10:
        return 0
    if t < 20:
        return 10
    return 0

def heat_prob(t, y, c1, x2):
    return c1 * (ambient(t) - y) + c2 * current(t)

if __name__ == "__main__":
    c1 = 1
    c2 = 1
    sol = integrate.solve_ivp(heat_prob, [0,30], [20], method="Radau", dense_output=True, args=(c1, c2))
    if not sol.success:
        print(sol.message)
    
    t = np.linspace(0, 30, 10000)
    plt.plot(t, ambient(t), label="ambient")
    plt.plot(t, sol.sol(t).T, label="solution")
    plt.plot(t, current(t), label="current")
    plt.title("Sample data")
    plt.legend()
    plt.savefig("simple.pdf")
    plt.show()


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

def plot_all(paths):
    print("plot all!")
    for path in paths:
        df = pd.read_parquet(path)
        df["rtctime"] = pd.to_datetime(df["rtctime"], unit="ms")
        print(path)
        df.plot(x="rtctime", y=["ambient_temp", "target_temperature"]) 
        plt.title(path)
        savepath = Path("./plots/") / (path.name + ".pdf")
        plt.savefig(savepath)
        #plt.show()
        plt.close()

def make_unite(replot=True, out_file="./all_parquets.pdf"):
    paths = get_sorted_paths()
    if replot:
        plot_all(paths)
    unite_cmd = "pdfunite " + all_file_names(paths) + ' ' + out_file
    os.system(unite_cmd)

if __name__ == "__main__":
    #min_max(paths, dfs)
    #plot_all(paths, dfs)
    make_unite(replot=False)
    
