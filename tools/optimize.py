import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy import integrate, interpolate, optimize, signal
from pathlib import Path
import sys

def get_sorted_paths(root="../data", glob="*.parquet"):
    paths = list(Path(root).rglob(glob))
    paths.sort()
    paths.sort(key=lambda p: len(str(p)))
    return paths

path = Path("../data/58.parquet")
df = pd.read_parquet(path)

#step = 1
#n = len(df) // step
#print("n:", n)

#df = df[:n*step]

#there were underscores infront here
time = df["rtctime"].to_numpy()
target = df["target_temperature"].to_numpy()
ambient = df["ambient_temp"].to_numpy()
current = df["feature_c"].to_numpy()
speed = df["car_speed"].to_numpy()
cooling = df["feature_ct"].to_numpy()

n = len(time)
time -= time[0]

if False:
    time = np.zeros(n)
    target = np.zeros(n)
    ambient = np.zeros(n)
    current = np.zeros(n)
    speed = np.zeros(n)
    cooling = np.zeros(n)
    
    for i in range(step):
        time += _time[i::step]
        target += _target[i::step]
        ambient += _ambient[i::step]
        current += _current[i::step]
        speed += _speed[i::step]
        cooling += _cooling[i::step]
    
    time /= step
    target /= step
    ambient /= step
    current /= step
    speed /= step
    cooling /= step

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


