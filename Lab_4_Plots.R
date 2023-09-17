library(tidyverse)

p_data <- read_csv('P_injection.csv')
q_data <- read_csv('q_injection.csv')

interpolated_P_Mpa <- approx(x = p_data$DAYS, y = p_data$`P(Mpa)`, xout = q_data$days, method = "linear")
q_interp <- data.frame(
  days = q_data$days,
  P_Mpa = interpolated_P_Mpa$y,
  t_hr = q_data$`t/hr`
)

interpolated_t_hr <- approx(x = q_data$days, y = q_data$`t/hr`, xout = p_data$DAYS, method = "linear")
p_interp <- data.frame(
  days = p_data$DAYS,
  P_Mpa = p_data$`P(Mpa)`,
  t_hr = interpolated_t_hr$y
)

data <- rbind(q_interp, p_interp)
data <- data |>
  arrange(days)

ggplot(data = data, aes(x = t_hr, y = P_Mpa)) +
  geom_point() +
  geom_smooth(method = "lm", se = FALSE, color = "blue") +
  labs(x = "t_hr", y = "P(Mpa)", title = "Interpolated Extraction Data Over 35 Days") +
  labs(x = "Water Injection Rate (t/hr)", y = "Average Pressure (MPa)")