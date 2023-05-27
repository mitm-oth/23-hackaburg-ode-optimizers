import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy import integrate, interpolate, optimize, signal
from pathlib import Path
from collections import namedtuple
from datetime import datetime
import sys

Measurement = namedtuple('Measurement', ['time', 'target', 'ambient', 'current', 'speed', 'cooling'])

def read_measurement(path):
    df = pd.read_parquet(path)
    
    time    = df["rtctime"].to_numpy()
    target  = df["target_temperature"].to_numpy()
    ambient = df["ambient_temp"].to_numpy()
    current = df["feature_c"].to_numpy()
    speed   = df["car_speed"].to_numpy()
    cooling = df["feature_ct"].to_numpy()

    # Move timestamps into nicer range
    #time -= time[0]

    return Measurement(time, target, ambient, current, speed, cooling)

def running_mean(raw, window_len):
    return signal.convolve(raw, np.ones(window_len), 'valid') / window_len

def preprocess_measurement(raw, window_lens):
    max_window_len = max(window_lens)
    split_at = -len(raw[0]) + max_window_len
    processed = [running_mean(data, w_len)[split_at:] for data, w_len in zip(raw, window_lens)]
    return Measurement(*processed)

def generate_heat_equation(data):
    f_target = interpolate.interp1d(data.time, data.target)
    f_ambient = interpolate.interp1d(data.time, data.ambient)
    f_current = interpolate.interp1d(data.time, data.current)
    f_speed = interpolate.interp1d(data.time, data.speed)
    f_cooling = interpolate.interp1d(data.time, data.cooling)

    def heat_transfer(t, y, c_ambient, c_current, c_speed, c_cooling):
        return c_ambient * (f_ambient(t) - y) + c_current * f_current(t) + c_speed * f_speed(t) + c_cooling * f_cooling(t)

    return heat_transfer

def solve_heat_equation(heat_transfer, data, coeff):
    #coeff = np.array([2.4e-6, 1e-8, 1.5e-6, 1e-5])
    t_span = [data.time[0], data.time[-1]]
    sol = integrate.solve_ivp(heat_transfer, t_span, [data.ambient[0]], method="Radau", t_eval=data.time,
            dense_output=False, vectorized=True, args=coeff) #, rtol=1e-6, atol=1e-9)

    return sol.y.T

def plot_all(data, predicted, title, *, target=True, prediction=True, sensors=True, show=False, savepath=None):
    plt.figure(figsize=(8, 6))
    timestamps = [datetime.fromtimestamp(x / 1000.0) for x in data.time[::10]]
    if target:
        plt.plot(timestamps, data.target[::10], label="target")
    if prediction:
        plt.plot(timestamps, predicted[::10], label="prediction")
    if sensors:
        plt.plot(timestamps, data.ambient[::10], label="ambient")
        plt.plot(timestamps, data.current[::10], label="current")
        plt.plot(timestamps, data.speed[::10], label="speed")
        plt.plot(timestamps, data.cooling[::10], label="cooling")
    plt.title(title)
    plt.xlabel("Time")
    plt.ylabel("Temperature in degree Celcius")
    plt.legend()
    if savepath:
        plt.savefig(savepath)
    if show:
        plt.show()
    plt.close()

def generate_plots_for(num):
    path = f"../data/{num}.parquet"
    raw = read_measurement(path)
    window_lens = (1, 30000, 30000, 30000, 30000, 30000)
    data = preprocess_measurement(raw, window_lens)
    heat_transfer = generate_heat_equation(data)
    coeff = np.array([2.4e-6, 1e-8, 1.5e-6, 1e-5])
    predicted = solve_heat_equation(heat_transfer, data, coeff)

    plot_all(raw, predicted, f"Raw sensor data ({path})", target=True, prediction=False, sensors=True,
            show=False, savepath=f"../assets/pngs/{num}a_raw_sensor.png")

    plot_all(data, predicted, f"Sensor data after preprocessing ({path})", target=True, prediction=False, sensors=True,
            show=False, savepath=f"../assets/pngs/{num}b_sensors.png")

    plot_all(data, predicted, f"Prediction based on other sensor data ({path})", target=True, prediction=True, sensors=True,
            show=False, savepath=f"../assets/pngs/{num}c_prediction.png")

    plot_all(data, predicted, f"Measured data vs Predicted data ({path})", target=True, prediction=True, sensors=False,
            show=False, savepath=f"../assets/pngs/{num}d_sensor_vs_prediction.png")

if __name__ == "__main__":
    #path = "../data/58.parquet"
    #path = "../data/81.parquet"
    #path = "../data/36.parquet"
    #num = 58
    num = 7
    #num = 3
    generate_plots_for(num)

sys.exit()


###############################################################################
###############################################################################
###############################################################################

#from dataclasses import dataclass

def get_sorted_paths(root="../data", glob="*.parquet"):
    paths = list(Path(root).rglob(glob))
    paths.sort()
    paths.sort(key=lambda p: len(str(p)))
    return paths

#@dataclass
#class Measurements:
#    time: Any
#    target: Any
#    ambient: Any
#    current: Any
#    speed: Any
#    cooling: Any

w = 1
#w_target  = 10000
#w_ambient = 10000
#w_current = 30000
#w_speed   = 7000
#w_cooling = 10000

w_target  = 30000
w_ambient = 30000
w_current = 30000
w_speed   = 30000
w_cooling = 30000


max_window = max(w, w_target, w_ambient, w_current, w_speed, w_cooling)

def running_mean(data, window_len):
    return signal.convolve(data, np.ones(window_len), 'valid') / window_len


split = -n+max_window
time    = running_mean(time, 1)[split:]
target  = running_mean(target, w_target)[split:]
ambient = running_mean(ambient, w_ambient)[split:]
current = running_mean(current, w_current)[split:]
speed   = running_mean(speed, w_speed)[split:]
cooling = running_mean(cooling, w_cooling)[split:]

f_target = interpolate.interp1d(time, target)
f_ambient = interpolate.interp1d(time, ambient)
f_current = interpolate.interp1d(time, current)
f_speed = interpolate.interp1d(time, speed)
f_cooling = interpolate.interp1d(time, cooling)




def heat_transfer(t, y, c_ambient, c_current, c_speed, c_cooling):
    return c_ambient * (f_ambient(t) - y) + c_current * f_current(t) + c_speed * f_speed(t) + c_cooling * f_cooling(t)


#coeff = np.array([1e-6, 0., 9e-7, 1e-7])
coeff = np.array([2.4e-6, 1e-8, 1.5e-6, 1e-5])

t_span = [time[0], time[-1]]
sol = integrate.solve_ivp(heat_transfer, t_span, [target[0]], method="Radau", t_eval=time,
        dense_output=False, vectorized=True, args=coeff) #, rtol=1e-6, atol=1e-9)

plt.plot(time[::10], target[::10], label="target")
plt.plot(time[::10], ambient[::10], label="ambient")
plt.plot(time[::10], current[::10], label="current")
plt.plot(time[::10], speed[::10], label="speed")
plt.plot(time[::10], cooling[::10], label="cooling")
plt.plot(time[::10], sol.y.T[::10], label="prediction")
plt.title(f"Coefficients: {coeff}")
plt.legend(loc="upper right")
plt.savefig("it_works.pdf")
plt.show()

sys.exit()

def err_sq(t, predicted):
    return (f_target(t) - predicted(t))**2

def cost(coeff):
    print(f"Called cost with {coeff}: ", end="")
    t_span = [time[0], time[-1]]
    sol = integrate.solve_ivp(heat_transfer, t_span, [target[0]], method="Radau", t_eval=time,
            dense_output=False, vectorized=True, args=coeff, rtol=1e-6, atol=1e-9)
    #norm, err = integrate.quad(err_sq, t_span[0], t_span[1], args=sol.sol)
    e = target - sol.y.reshape(-1)
    norm = e @ e
    print(norm)
    return norm

def run(initial_coeff):
    #initial_coeff = np.array([1., 1., 1., 1.])
    res = optimize.minimize(cost, initial_coeff, callback=lambda x: print(f"Minimize step: {x}"))
    print(res)


if __name__ == "__main__":
    run(coeff)


