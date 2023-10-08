from ethan_stinky_model_attempt2 import *

if __name__ == "__main__":

    # t_p, p, t_q, q = load_data()
    # print(len(t_p), len(p), len(t_q), len(q))

    # benchmarking for ODE
    # plot_benchmark()

    # # # ODE model with initial parameter values
    # plot_suitable()

    # # # ODE model with improved parameter values from curve_fit
    # plot_improve()

    # Plot showing that with q = 0, there is very little pressure variation
    # as expected
    plot_scaled_ode()
