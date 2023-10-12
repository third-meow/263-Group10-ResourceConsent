
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from sklearn.linear_model import LinearRegression
def exponential_fit(x, a, b):
    return a * np.exp(b * x)

# Data points
magnitude = [1, 2, 3]
pressure = [0.07, 0.15, 0.2]

# Fit the data to an exponential curve
params, covariance = np.polyfit(pressure, np.log(magnitude), 1, cov=True)
a = np.exp(params[1])
b = params[0]

# Generate the curve data
pressure_fit = np.linspace(min(pressure), max(pressure), 100)
magnitude_fit = exponential_fit(pressure_fit, a, b)

def pressure_for_magnitude(magnitude, a, b):
    return np.log(magnitude / a) / b

# Extrapolating to find the pressure for magnitude 4.5
pressure_at_4_5 = pressure_for_magnitude(4.5, a, b)
print(pressure_at_4_5)

pressure_at_3 = pressure_for_magnitude(3, a, b)
print(pressure_at_3)

extended_pressure_fit = np.linspace(min(pressure) - 0.1, max(pressure) + 0.1, 100)  # extending the range a bit
extended_magnitude_fit = exponential_fit(extended_pressure_fit, a, b)
# Plotting
plt.figure(figsize=(8, 6))
plt.plot(pressure, magnitude, 'o', label='Data points')
plt.plot(extended_pressure_fit, extended_magnitude_fit, '-', label='Extended Exponential fit')
plt.axvline(pressure_at_4_5, color='r', linestyle='--', label='Pressure at Magnitude 4.5')
plt.axvline(pressure_at_3, color='r', linestyle='--', label='Pressure at Magnitude 3')
plt.axhline(4.5, color='g', linestyle='--', label='Magnitude 4.5')
plt.axhline(3, color='g', linestyle='--', label='Magnitude 3')
plt.xlabel('Pressure')
plt.ylabel('Magnitude')
plt.title('Magnitude vs. Pressure with Exponential Fit and Extrapolated Value')
plt.legend()
plt.grid(True)
plt.show()

injection_data_combined = pd.read_csv("injection_data_combined.csv")


# Assuming df is your dataframe
X = injection_data_combined[['P']]
y = injection_data_combined['q']

model = LinearRegression().fit(X, y)
pressures_to_predict = [0.24705394903669284, 0.19918729628598425]
q_values = model.predict([[p] for p in pressures_to_predict])
print(q_values)


