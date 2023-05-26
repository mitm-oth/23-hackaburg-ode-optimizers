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
    coefficients = {}
    for c in columns:
        if c == "rtctime" or c == "target_temperature":
            continue
        coefficients[c] = interp1d(df["rtctime"], df[c], kind="linear")
    
    return coefficients

# function defining the ode
def odefun(t, target_temp, coefficients, interpolations):
    val = coefficients["ambient_temp"] * (interpolations["ambient_temp"](t) - target_temp)
    
    for col, c in coefficients.items():
        if c != "ambient_temp":
            val += c * interpolations[col](t)
            
    return val


if __name__ == "__main__":
    # get example dataframe
    cutter = Cutter("data", gap_ms=10)
    df = cutter.get_biggest_track()
    print(df.head())
    print(df.shape)
    
    # specify relevant columns
    columns = ["target_temperature", "ambient_temp", "feature_c"]
    
    # convolution
    df = convolution(df, 100)
    df.plot(x="rtctime", y=columns) #y=["target_temperature", "ambient_temp", "car_speed", "feature_c"])
    plt.plot()
    plt.title("features after colvolution")
    plt.show()

    # interpolate
    interpolations = interpolation(df, columns)

    # solve ode and plot prediction
    t_span = [df.iloc[0]["rtctime"], df.iloc[-1]["rtctime"]]
    coefficients = {"ambient_temp": 0.005, "feature_c": 0.0001}
    y0 = [df.iloc[0]["target_temperature"]]
    sol = solve_ivp(fun=odefun, t_span=t_span, y0=y0, args=(coefficients, interpolations))
    plt.plot(sol.t, sol.y.T, label="prediction")
    plt.plot(df["rtctime"], df["target_temperature"], label="target")
    plt.legend()
    plt.show()