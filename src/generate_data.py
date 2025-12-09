import numpy as np
import pandas as pd
import datetime as dt


# -------------------------
# CONFIGURATION
# -------------------------
num_days = 7
num_points = num_days * 24 * 60   # 1 minute interval

start_time = dt.datetime(2024, 1, 1)


def generate_timestamp_series(start, n_points):
    return [start + dt.timedelta(minutes=i) for i in range(n_points)]


timestamps = generate_timestamp_series(start_time, num_points)

np.random.seed(55)


# ================================================================
# HEALTHY PUMP DATA
# ================================================================

# Vibration (mm/s)
vibration = np.random.normal(1.2, 0.2, num_points)

# Suction & discharge pressure (bar)
P_suction = np.random.normal(2.0, 0.1, num_points)
P_discharge = np.random.normal(6.5, 0.15, num_points)

# Flow rate (m³/hr)
flow_rate = np.random.normal(40, 1.5, num_points)

# Motor current (A)
motor_current = np.random.normal(25, 1, num_points)

# Temperature (°C)
temperature = np.random.normal(65, 1.5, num_points)

df_healthy_pump = pd.DataFrame({
    "timestamp": timestamps,
    "vibration": vibration,
    "suction_pressure": P_suction,
    "discharge_pressure": P_discharge,
    "flow_rate": flow_rate,
    "motor_current": motor_current,
    "temperature": temperature
})

df_healthy_pump.to_csv("pump_healthy.csv", index=False)

print("Healthy pump dataset saved as pump_healthy.csv")



# ================================================================
# FAULTY PUMP DATA (bearing wear + cavitation + overload)
# ================================================================

# Vibration increases (bearing damage)
vibration_fault = vibration + np.random.normal(2.0, 0.5, num_points)

# Suction pressure drops (cavitation starts)
P_suction_fault = P_suction - np.random.normal(0.8, 0.2, num_points)

# Discharge pressure drops (inefficiency)
P_discharge_fault = P_discharge - np.random.normal(1.0, 0.3, num_points)

# Flow rate decreases (impeller damage)
flow_rate_fault = flow_rate - np.random.normal(8, 1.5, num_points)

# Motor current increases (overload due to wear)
motor_current_fault = motor_current + np.random.normal(5, 1, num_points)

# Temperature rises (bearing overheating)
temperature_fault = temperature + np.random.normal(10, 2, num_points)

df_faulty_pump = pd.DataFrame({
    "timestamp": timestamps,
    "vibration": vibration_fault,
    "suction_pressure": P_suction_fault,
    "discharge_pressure": P_discharge_fault,
    "flow_rate": flow_rate_fault,
    "motor_current": motor_current_fault,
    "temperature": temperature_fault
})

df_faulty_pump.to_csv("pump_faulty.csv", index=False)

print("Faulty pump dataset saved as pump_faulty.csv")
