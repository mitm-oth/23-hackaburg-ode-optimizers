# import required packages
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from scipy.integrate import solve_ivp
from scipy.interpolate import interp1d
from scipy import signal
from ..lib.cutter import Cutter
from scipy.optimize import minimize

IVP_METHOD = "Radau"
MIN_METHOD = "BFGS"


def convolution(df, w=20):
    conv = {}
    for col in df.columns:
        if col == "rtctime" or col == "target_temperature":
            continue
        conv[col] = signal.convolve(df[col], np.ones(w), 'valid') / w
    
    conv["rtctime"] = np.array(df["rtctime"])[w-1:]
    conv["target_temperature"] = np.array(df["target_temperature"])[w-1:]
    
    return conv
    

def interpolation(df, columns):
    interpolations = []
    for c in columns:
        if c == "rtctime" or c == "target_temperature":
            continue
        interpolations.append(interp1d(df["rtctime"], df[c], kind="linear"))
    
    return interpolations

# function defining the ode
def odefun(t, target_temp, coefficients, interpolations):
    assert len(coefficients) == len(interpolations)
    
    val = coefficients[0] * (interpolations[0](t) - target_temp) #! ambient temp has to be first coefficient/interpolation
    
    for c, f in zip(coefficients[1:], interpolations[1:]):
            val += c * f(t)
            
    return val


def costfun(coefficients, coeff, interpolations):
    t_span = [coeff["rtctime"][0], coeff["rtctime"][-1]]
    y0 = [df.iloc[0]["target_temperature"]]
    sol = solve_ivp(fun=odefun, t_span=t_span, y0=y0, args=(coefficients, interpolations), dense_output=True, method=IVP_METHOD, vectorized=True)
    
    # integrate squared error #! requires equidistant time steps in dataframe
    ts = np.array(df["rtctime"])
    solvals = sol.sol(ts)
    err = np.linalg.norm(solvals - np.array(df["target_temperature"]))
    
    print(f"called costfun {err}: {coefficients}")
    
    return err


if __name__ == "__main__":
    # get example dataframe
    cutter = Cutter("data", gap_ms=10)
    df = cutter.get_biggest_track()
    #df = df.iloc[-100000:] # cut dataframe to last 5000 rows
    print(df.head())
    print(df.shape)
    
    # specify relevant columns
    columns = ["target_temperature", "ambient_temp", "feature_c", "feature_ct", "car_speed"]  #! ambient temp has to be first after target coefficient/interpolation
    
    # convolution
    w = 20
    conv = convolution(df, w)
    print(len(conv["rtctime"]))
    #df.plot(x="rtctime", y=columns)
    #plt.title("features after colvolution")
    #plt.show()

    # interpolation
    interpolations = interpolation(df, columns)
    
    # starting coefficients
    coefficients = np.array([3.162277660168379, 3.162277660168379, 0.0031622776601683794, 1.0])
    
    # solve ode or initial parameters
    t_span = [conv["rtctime"][0], conv["rtctime"][-1]]
    y0 = [conv["target_temperature"][0]]
    sol0 = solve_ivp(fun=odefun, t_span=t_span, y0=y0, args=(coefficients, interpolations), method=IVP_METHOD)

    # optimize
    res = minimize(costfun, coefficients, args=(conv, interpolations), options={"disp": True}, method=MIN_METHOD, callback=lambda xk: print("\n\n", xk))
    print(res)

    # solve ode and plot prediction for new coefficients
    t_span = [conv["rtctime"][0], conv["rtctime"][-1]]
    y0 = [conv["target_temperature"][0]]
    sol = solve_ivp(fun=odefun, t_span=t_span, y0=y0, args=(res.x, interpolations), method=IVP_METHOD)

    plt.plot(conv["rtctime"], conv["target_temperature"], label="target")
    #plt.plot(df["rtctime"], df["ambient_temp"], label="ambient_temp")
    #plt.plot(df["rtctime"], df["feature_c"], label="feature_c")
    plt.plot(sol0.t, sol0.y.T, label="prediction init")
    plt.plot(sol.t, sol.y.T, label="prediction")
    plt.title(f"{res.fun}")
    plt.legend()
    plt.savefig(f"optimize_coeff.pdf")
    plt.close()

    