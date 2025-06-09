import pandas as pd
import matplotlib.pyplot as plt

# Load the CSV file
df = pd.read_csv("timing_report.csv")

# Filter out the wall-clock row
df_clean = df[df['buffer_id'].astype(str).str.lower() != 'wall']
df_clean['buffer_id'] = df_clean['buffer_id'].astype(int)

# Create performance plot
fig, ax1 = plt.subplots(figsize=(10, 6))

color = 'tab:blue'
ax1.set_xlabel('Buffer ID')
ax1.set_ylabel('GFlops/s', color=color)
ax1.plot(df_clean['buffer_id'], df_clean['GFlops/s'], marker='o', color=color, label='GFlops/s')
ax1.tick_params(axis='y', labelcolor=color)
ax1.grid(True)

# Add second y-axis for bandwidth
ax2 = ax1.twinx()
color = 'tab:red'
ax2.set_ylabel('Bandwidth (GB/s)', color=color)
ax2.plot(df_clean['buffer_id'], df_clean['BW_GBps'], marker='s', linestyle='--', color=color, label='Bandwidth (GB/s)')
ax2.tick_params(axis='y', labelcolor=color)

plt.title("GPU Performance per Buffer: GFLOPs and Bandwidth")
fig.tight_layout()
plt.show()
