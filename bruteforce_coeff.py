# try to bruteforce coefficients for the ode

# import required packages
from lib.training import *
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

    # interpolate
    interpolations = interpolation(df, columns)

    # try to bruteforce parameters
    for c1 in np.logspace(1, -3, 6):
        for c2 in np.logspace(1, -3, 6):
            for c3 in np.logspace(1, -3, 6):
                for c4 in np.logspace(1, -3, 6):
                    coefficients = np.array([c1, c2,  c3, c4])
                    print(coefficients, end=":   ")
                    try:
                        # solve ode and plot prediction
                        t_span = [df.iloc[0]["rtctime"], df.iloc[-1]["rtctime"]]
                        y0 = [df.iloc[0]["target_temperature"]]
                        sol0 = solve_ivp(fun=odefun, t_span=t_span, y0=y0, args=(coefficients, interpolations), dense_output=True, method=IVP_METHOD)

                        # integrate squared error #! requires equidistant time steps in dataframe
                        ts = np.array(df["rtctime"])
                        solvals = sol0.sol(ts)
                        err = np.linalg.norm(solvals - np.array(df["target_temperature"]))
                        print(err)
    
                        # plot
                        plt.plot(df["rtctime"], df["target_temperature"], label="target")
                        plt.plot(df["rtctime"], df["ambient_temp"], label="ambient_temp")
                        plt.plot(df["rtctime"], df["feature_c"], label="feature_c")
                        plt.plot(sol0.t, sol0.y.T, label="prediction init")
                        plt.title(f"{err}")
                        plt.legend()
                        plt.savefig(f"coeffbruteforce/{c1}_{c2}_{c3}_{c4}_{err}.pdf")
                        plt.show()
                    except Exception as e:
                        print(e)
                        
    
    # try to bruteforce parameters
    for c1 in np.logspace(2, -4, 10):
        for c2 in np.logspace(2, -4, 10):
            for c3 in np.logspace(2, -4, 10):
                for c4 in np.logspace(2, -4, 10):
                    coefficients = np.array([c1, c2,  c3, c4])
                    print(coefficients, end=":   ")
                    try:
                        # solve ode and plot prediction
                        t_span = [df.iloc[0]["rtctime"], df.iloc[-1]["rtctime"]]
                        y0 = [df.iloc[0]["target_temperature"]]
                        sol0 = solve_ivp(fun=odefun, t_span=t_span, y0=y0, args=(coefficients, interpolations), dense_output=True, method=IVP_METHOD)

                        # integrate squared error #! requires equidistant time steps in dataframe
                        ts = np.array(df["rtctime"])
                        solvals = sol0.sol(ts)
                        err = np.linalg.norm(solvals - np.array(df["target_temperature"]))
                        print(err)
    
                        # plot
                        plt.plot(df["rtctime"], df["target_temperature"], label="target")
                        plt.plot(df["rtctime"], df["ambient_temp"], label="ambient_temp")
                        plt.plot(df["rtctime"], df["feature_c"], label="feature_c")
                        plt.plot(sol0.t, sol0.y.T, label="prediction init")
                        plt.title(f"{err}")
                        plt.legend()
                        plt.savefig(f"coeffbruteforce/{c1}_{c2}_{c3}_{c4}_{err}.pdf")
                        plt.close()
                    except Exception as e:
                        print(e)