import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os

# Load the CSV file generated from the C++ benchmark
csv_file = "stream_timing_log.csv"

# Check if the file exists
if not os.path.exists(csv_file):
    raise FileNotFoundError(f"'{csv_file}' not found. Please ensure it exists in the working directory.")

# Read data
df = pd.read_csv(csv_file)

# Convert timing from microseconds if needed
df["total_time_us"] = df["copy_time_us"] + df["kernel_time_us"]

# Plot

# Plot
plt.figure(figsize=(12, 6))
sns.lineplot(data=df, x="batch_id", y="total_time_us", label="Total time (us)")
sns.lineplot(data=df, x="batch_id", y="copy_time_us", label="Copy time (us)")
sns.lineplot(data=df, x="batch_id", y="kernel_time_us", label="Kernel time (us)")
plt.title("CUDA Pointwise Multiply: Batch Timing Profile")
plt.xlabel("Batch ID")
plt.ylabel("Time (microseconds)")
plt.grid(True)
plt.legend()
plt.tight_layout()

# Save the figure as PNG with high resolution
plt.savefig("cuda_timing_profile.png", dpi=300)
