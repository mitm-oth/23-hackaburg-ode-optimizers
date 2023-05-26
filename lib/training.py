# import required packages
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from scipy.integrate import solve_ivp
from scipy.interpolate import interp1d
from scipy import signal
from cutter import Cutter

if __name__ == "__main__":
    # get example dataframe
    cutter = Cutter("data")
    gen = cutter.get_gen()
    df = next(gen)
    df.head()
    df = df.iloc[-1000000:]

    # convolution
    w = 10000
    conv_c = signal.convolve(df["feature_c"], np.ones(w), 'valid') / w
    conv_target = signal.convolve(df["target_temperature"], np.ones(w), 'valid') / w
    df = df.iloc[w-1:]
    df["feature_c"] = conv_c
    df["target_temperature"] = conv_target

    # plot after convolution
    df.plot(x="rtctime", y=["target_temperature", "feature_c", "ambient_temp"]) #y=["target_temperature", "ambient_temp", "car_speed", "feature_c"])
    plt.plot()
    plt.show()

    # interpolate
    f_target = interp1d(df["rtctime"], df["target_temperature"], kind="linear")
    f_c = interp1d(df["rtctime"], df["feature_c"], kind="linear")
    f_ambient = interp1d(df["rtctime"], df["ambient_temp"], kind="linear")

    # function defining the ode
    C1, C2 = 0.0002, 0.0005
    def odefun(t, target_temp):
        val = C1 * (f_ambient(t) - target_temp) + C2 * f_c(t)
        return val

    # solve ode and plot prediction
    sol = solve_ivp(fun=odefun, t_span=[df.iloc[0]["rtctime"], df.iloc[-1]["rtctime"]], y0=[df.iloc[0]["target_temperature"]])
    plt.plot(sol.t, sol.y.T)
    plt.plot(df["rtctime"], df["target_temperature"])
    plt.show()