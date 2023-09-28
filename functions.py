import pandas as pd


def ode_model(t, P, q, a, b, C, P0, dqdt):
    """ Return the derivative dx/dt at time, t, for given parameters.
        Parameters:
        -----------
        t : float
            Independent variable time.
        P : float
            Dependent variable Pressure
        q : float
            mass injection/ejection rate.
        a : float
            mass injection strength parameter.
        b : float
            recharge strength parameter.
        C : float
            slow drainage strength parameter
        P0 : float
            Ambient value of dependent variable.
        dqdt : float
            change in injection with respect to time
        Returns:
        --------
        dxdt : float
            Derivative of dependent variable with respect to independent variable time.
        Notes:
        ------
        None
    """
    # equation to return the derivative of dependent variable with respect to time

    # TYPE IN YOUR Pressure ODE HERE
    dPdt = a * q - b * (P - P0) - + C * dqdt

    return dPdt


def combine_data():
    """
    combines the q and P data onto the same time axis, join both axis and interpolate the missing values.
    Parameters:
    -----------
    Returns:
    ----------
    """

    df_p = pd.read_csv("P_injection.csv")
    df_q = pd.read_csv("q_injection.csv")
    df_p.columns = df_p.columns.str.lower()
    df_q.columns = df_q.columns.str.lower()
    # Combine the DataFrames
    df_combined = pd.merge(df_p, df_q, on='days', how='outer').sort_values(by='days')

    # Interpolate the missing values
    df_combined.interpolate(method='linear', inplace=True)
    df_combined.ffill(inplace=True)  # Forward fill for any remaining missing values
    df_combined.bfill(inplace=True)  # Backward fill for any remaining missing values

    print(df_combined)

    # Save the combined DataFrame to CSV
    df_combined.to_csv('combined_data.csv', index=False)


combine_data()
