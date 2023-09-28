import pandas as pd
import numpy as np
import math


res = input("This script will overwrite injection_data_combined.csv. Enter Y to continue")
if res != 'Y':
    quit()

pressure_messy = pd.read_csv("P_injection.csv")
injection_messy = pd.read_csv("q_injection.csv")

TIMESTAMP_INCREMENT = 0.5
START_TIMESTAMP = 0
END_TIMESTAMP = 35

# create timestamps
num_timestamps = math.floor(((END_TIMESTAMP-START_TIMESTAMP)/TIMESTAMP_INCREMENT)+1)
glob_timestamp = np.linspace(START_TIMESTAMP, END_TIMESTAMP, num_timestamps)
assert(num_timestamps == len(glob_timestamp))

# interpelate
pressure_neat = np.interp(glob_timestamp, pressure_messy['DAYS'], pressure_messy['P(Mpa)'])
injection_neat = np.interp(glob_timestamp, injection_messy['days'], injection_messy['t/hr'])

# create differentials
dPdt_neat = np.zeros(num_timestamps)
dqdt_neat = np.zeros(num_timestamps)

# excuse my shitty for-loop
i = 1
while i < 71:
    dPdt_neat[i] = (pressure_neat[i]-pressure_neat[i-1])/(glob_timestamp[i]-glob_timestamp[i-1])
    dqdt_neat[i] = (injection_neat[i]-injection_neat[i-1])/(glob_timestamp[i]-glob_timestamp[i-1])
    i += 1

# create dataframe
tidy_df = pd.DataFrame(data={'P':pressure_neat, 'q':injection_neat, 'dP/dt':dPdt_neat, 'dq/dt':dqdt_neat}, index=glob_timestamp)

# write to file
tidy_df.to_csv("injection_data_combined.csv")



