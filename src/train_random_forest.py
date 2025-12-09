import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import warnings
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import r2_score
import joblib
import os

warnings.filterwarnings("ignore")

#User Input
HEALTHY_CSV_PATH = "pump_healthy.csv"   # healthy baseline dataset
TEST_CSV_PATH    = "pump_healthy.csv"    # dataset to evaluate
OUTPUT_PREFIX    = "pump_output"

MIN_CONSECUTIVE_FAULTS = 30  # consecutive anomaly points
WARMUP_IGNORE = 60           # ignore first 60 samples for detection

# FUNCTION: load & compute efficiency + features
def prepare_pump_df(csv_path):
    df = pd.read_csv(csv_path, parse_dates=['timestamp'])
    df = df.sort_values('timestamp').reset_index(drop=True)

    required_cols = ['vibration','suction_pressure','discharge_pressure',
                     'flow_rate','motor_current','temperature']
    for c in required_cols:
        if c not in df.columns:
            raise ValueError(f"Missing column {c} in {csv_path}. Required: {required_cols}")

    df[required_cols] = df[required_cols].interpolate(limit=3).fillna(method='bfill').fillna(method='ffill')

    flow_m3s = (df['flow_rate'].astype(float) / 3600.0).replace(0, 1e-6)

    P_suction_bar   = df['suction_pressure'].astype(float)
    P_discharge_bar = df['discharge_pressure'].astype(float)
    P_suction       = P_suction_bar * 1e5
    P_discharge     = P_discharge_bar * 1e5

    P_hydraulic = flow_m3s * (P_discharge - P_suction)

    voltage_line = 415.0
    pf = 0.85
    P_electrical = df['motor_current'].astype(float) * voltage_line * np.sqrt(3) * pf

    eff_est = np.clip(P_hydraulic / (P_electrical + 1e-6), 0.0, 2.0)

    df['P_hydraulic'] = P_hydraulic
    df['P_electrical'] = P_electrical
    df['eff_est'] = eff_est

    df['deltaP_bar'] = P_discharge_bar - P_suction_bar
    df['vib_roll_mean_60']  = df['vibration'].rolling(60, min_periods=1).mean()
    df['vib_roll_std_60']   = df['vibration'].rolling(60, min_periods=1).std().fillna(0)
    df['flow_roll_mean_60'] = df['flow_rate'].rolling(60, min_periods=1).mean()

    feature_cols = ['vibration','suction_pressure','discharge_pressure',
                    'flow_rate','motor_current','temperature',
                    'deltaP_bar','vib_roll_mean_60','vib_roll_std_60','flow_roll_mean_60']
    X = df[feature_cols].fillna(method='ffill').fillna(method='bfill').values
    y = df['eff_est'].values

    return df, X, y, feature_cols

# HEALTHY BASELINE
df_healthy, X_healthy, y_healthy, feature_cols = prepare_pump_df(HEALTHY_CSV_PATH)

rf = RandomForestRegressor(n_estimators=200, random_state=42, n_jobs=-1)
rf.fit(X_healthy, y_healthy)

y_healthy_pred = rf.predict(X_healthy)
resid_healthy  = y_healthy - y_healthy_pred
abs_resid_healthy = np.abs(resid_healthy)

q99 = np.quantile(abs_resid_healthy, 0.99)
threshold = q99 * 1.2 if q99 > 0 else 1e-6

print(f"[PUMP] Healthy |residual| 99th percentile={q99:.4e}, threshold={threshold:.4e}")

# TEST DATA
df_test, X_test, y_test, _ = prepare_pump_df(TEST_CSV_PATH)
y_test_pred = rf.predict(X_test)

r2 = r2_score(y_test, y_test_pred)
print(f"[PUMP] R2 on TEST data: {r2:.4f}")

resid_test = y_test - y_test_pred
abs_resid_test = np.abs(resid_test)
anomaly_flags = abs_resid_test > threshold

times_test = pd.to_datetime(df_test['timestamp'])

# FIRST CONSISTENT FAULT AFTER WARMUP
first_fault_idx = None
count = 0
for i in range(WARMUP_IGNORE, len(anomaly_flags)):
    if anomaly_flags[i]:
        count += 1
        if count >= MIN_CONSECUTIVE_FAULTS:
            first_fault_idx = i - MIN_CONSECUTIVE_FAULTS + 1
            break
    else:
        count = 0

if first_fault_idx is not None:
    first_fault_time = times_test.iloc[first_fault_idx]
    print(f"[PUMP] First consistent fault detected at index {first_fault_idx}, time {first_fault_time}")
else:
    print("[PUMP] No consistent fault detected (after warmup).")

# Consider anomaly fraction only after warmup
valid_flags = anomaly_flags.copy()
valid_flags[:WARMUP_IGNORE] = False
anom_fraction = valid_flags.mean()

if first_fault_idx is None and anom_fraction < 0.05:
    summary_msg = "HEALTHY: Pump operates normally; no immediate maintenance needed."
else:
    summary_msg = ("FAULTY: Pump shows consistent deviation from healthy behavior; "
                   "maintenance/inspection recommended.")
print("[PUMP] FINAL STATUS:", summary_msg)

# Save model
os.makedirs('models', exist_ok=True)
joblib.dump(rf, os.path.join('models', OUTPUT_PREFIX + '_rf.pkl'))

#Plots
os.makedirs('plots', exist_ok=True)

plt.figure(figsize=(14,12))

ax1 = plt.subplot(4,1,1)
ax1.plot(times_test, df_test['vibration'], label='Vibration (mm/s)')
ax1.plot(times_test, df_test['flow_rate'], label='Flow rate (m3/hr)')
ax1.set_ylabel('Vibration / Flow')
ax1.legend(loc='upper right')

ax2 = plt.subplot(4,1,2, sharex=ax1)
ax2.plot(times_test, df_test['suction_pressure'], label='Suction Pressure (bar)')
ax2.plot(times_test, df_test['discharge_pressure'], label='Discharge Pressure (bar)')
ax2.set_ylabel('Pressure (bar)')
ax2.legend(loc='upper right')

ax3 = plt.subplot(4,1,3, sharex=ax1)
ax3.plot(times_test, df_test['motor_current'], label='Motor Current (A)')
ax3.plot(times_test, df_test['temperature'], label='Temperature (°C)')
ax3.set_ylabel('Current / Temp')
ax3.legend(loc='upper right')

ax4 = plt.subplot(4,1,4, sharex=ax1)
ax4.plot(times_test, y_test, label='Efficiency True (proxy)', linewidth=1)
ax4.plot(times_test, y_test_pred, label='Efficiency Predicted (RF)', linestyle='--', linewidth=1)
ax4.scatter(times_test[valid_flags], y_test[valid_flags], color='red', s=8, label='Anomaly (after warmup)')
ax4.set_ylabel('Efficiency')
ax4.set_xlabel('Timestamp')
ax4.legend(loc='upper right')

if first_fault_idx is not None:
    for ax in [ax1, ax2, ax3, ax4]:
        ax.axvline(first_fault_time, color='black', linestyle='--', label='First Fault')
    handles, labels = ax4.get_legend_handles_labels()
    unique = dict(zip(labels, handles))
    ax4.legend(unique.values(), unique.keys(), loc='upper right')

plt.suptitle('Pump: Test Sensor Time-Series & Efficiency Prediction')
plt.tight_layout(rect=[0,0,1,0.96])
plt.savefig(os.path.join('plots', OUTPUT_PREFIX + "_test_sensors_efficiency.png"), dpi=200)

# Residuals + threshold
plt.figure(figsize=(12,4))
plt.plot(times_test, resid_test, label='Residual (true - pred)')
plt.axhline(threshold,  color='red', linestyle='--', label='+Threshold')
plt.axhline(-threshold, color='red', linestyle='--')
plt.axhline(0, color='k', linewidth=0.5)
if first_fault_idx is not None:
    plt.axvline(first_fault_time, color='black', linestyle='--', label='First Fault')
plt.ylabel('Residual')
plt.xlabel('Time')
plt.title('Pump: Residuals & Threshold')
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.savefig(os.path.join('plots', OUTPUT_PREFIX + "_residuals_threshold.png"), dpi=200)

# Feature importance
importances = rf.feature_importances_
idx = np.argsort(importances)[::-1]
plt.figure(figsize=(9,4))
plt.bar([feature_cols[i] for i in idx], importances[idx])
plt.xticks(rotation=45, ha='right')
plt.title('Pump RF Feature Importances (healthy training)')
plt.tight_layout()
plt.savefig(os.path.join('plots', OUTPUT_PREFIX + "_feature_importances.png"), dpi=200)

# Save predictions CSV
df_out = df_test.copy()
df_out['eff_true'] = y_test
df_out['eff_pred'] = y_test_pred
df_out['residual'] = resid_test
df_out['abs_residual'] = abs_resid_test
df_out['anomaly_flag'] = valid_flags.astype(int)
if first_fault_idx is not None:
    df_out['first_fault_time'] = first_fault_time
else:
    df_out['first_fault_time'] = pd.NaT

df_out.to_csv(OUTPUT_PREFIX + "_test_predictions.csv", index=False)

print("Result:", summary_msg)
