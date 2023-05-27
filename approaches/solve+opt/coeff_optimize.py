# try to optimize initial parameters

# import required packages
from approaches.training import *
from lib.cutter import Cutter

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
    df = convolution(df, w)
    df = df.iloc[10000::w//2] # skip rows for performance
    print(df.shape)
    #df.plot(x="rtctime", y=columns)
    #plt.title("features after colvolution")
    #plt.show()

    # interpolation
    interpolations = interpolation(df, columns)
    
    # starting coefficients
    coefficients = np.array([3.162277660168379, 3.162277660168379, 0.0031622776601683794, 1.0])
    
    # solve ode or initial parameters
    t_span = [df.iloc[0]["rtctime"], df.iloc[-1]["rtctime"]]
    y0 = [df.iloc[0]["target_temperature"]]
    sol0 = solve_ivp(fun=odefun, t_span=t_span, y0=y0, args=(coefficients, interpolations), method=IVP_METHOD)

    # optimize
    res = minimize(costfun, coefficients, args=(df, interpolations), options={"disp": True}, method=MIN_METHOD, callback=lambda xk: print("\n\n", xk))
    print(res)

    # solve ode and plot prediction for new coefficients
    t_span = [df.iloc[0]["rtctime"], df.iloc[-1]["rtctime"]]
    y0 = [df.iloc[0]["target_temperature"]]
    sol = solve_ivp(fun=odefun, t_span=t_span, y0=y0, args=(res.x, interpolations), method=IVP_METHOD)

    plt.plot(df["rtctime"], df["target_temperature"], label="target")
    #plt.plot(df["rtctime"], df["ambient_temp"], label="ambient_temp")
    #plt.plot(df["rtctime"], df["feature_c"], label="feature_c")
    plt.plot(sol0.t, sol0.y.T, label="prediction init")
    plt.plot(sol.t, sol.y.T, label="prediction")
    plt.title(f"{res.fun}")
    plt.legend()
    plt.savefig(f"optimize_coeff.pdf")