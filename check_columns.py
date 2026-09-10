from src.inference.predict_pipeline import read_raw_sensor_csv
import numpy as np

filepath = r"C:\Users\Admin\OneDrive\Desktop\rehab_planner\data\sample_upload\Lateral_raises_gyro.csv"

result = read_raw_sensor_csv(filepath)

for i, arr in enumerate(result):
    print(f"\n--- Array {i} ---")
    print("Type:", type(arr))
    print("Shape:", arr.shape)
    print("dtype:", arr.dtype)
    print("First 3 rows:\n", arr[:3])