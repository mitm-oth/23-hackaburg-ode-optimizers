import numpy as np
import matplotlib.pyplot as plt
from scipy import integrate

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
    sol = integrate.solve_ivp(heat_prob, [0,30], [20], method="Radau", vectorized=True, dense_output=True, args=(c1, c2))
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

