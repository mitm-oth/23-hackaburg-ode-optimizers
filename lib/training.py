# import required packages
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from scipy.integrate import solve_ivp
from scipy.interpolate import interp1d
from scipy import signal
from cutter import Cutter


def convolution(df, w=100):
    conv = {}
    for col in df.columns:
        if col == "rtctime" or col == "target_temperature":
            continue
        conv[col] = signal.convolve(df[col], np.ones(w), 'valid') / w
    
    df = df.iloc[w-1:]
    
    for col in df.columns:
        if col == "rtctime" or col == "target_temperature":
            continue
        df[col] = conv[col]
    
    return df
    

def interpolation(df, columns):
    interpolations = []
    for c in columns:
        if c == "rtctime" or c == "target_temperature":
            continue
        interpolations.append(interp1d(df["rtctime"], df[c], kind="linear"))
    
    return interpolations

# function defining the ode
def odefun(t, target_temp, coefficients, interpolations):
    val = coefficients[0] * (interpolations[0](t) - target_temp) #! ambient temp has to be first coefficient/interpolation
    
    for c, f in zip(coefficients[1:], interpolations[1:]):
            val += c * f(t)
            
    return val


if __name__ == "__main__":
    # get example dataframe
    cutter = Cutter("data", gap_ms=10)
    df = cutter.get_biggest_track()
    print(df.head())
    print(df.shape)
    
    # specify relevant columns
    columns = ["target_temperature", "ambient_temp", "feature_c"]  #! ambient temp has to be first after target coefficient/interpolation
    
    # convolution
    w = 100
    df = convolution(df, w)
    df.plot(x="rtctime", y=columns)
    plt.plot()
    plt.title("features after colvolution")
    plt.show()

    # interpolate
    interpolations = interpolation(df, columns)

    # solve ode and plot prediction
    t_span = [df.iloc[0]["rtctime"], df.iloc[-1]["rtctime"]]
    y0 = [df.iloc[0]["target_temperature"]]
    coefficients = np.array([0.005, 0.0001])
    sol = solve_ivp(fun=odefun, t_span=t_span, y0=y0, args=(coefficients, interpolations))
    
    plt.plot(sol.t, sol.y.T, label="prediction")
    plt.plot(df["rtctime"], df["target_temperature"], label="target")
    plt.legend()
    plt.show()