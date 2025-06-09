import pandas as pd
import matplotlib.pyplot as plt

# Reload CSV after kernel reset
df = pd.read_csv("timing_report.csv")

# Filter out wall timing row
df_clean = df[df['buffer_id'].astype(str).str.lower() != 'wall']

# Convert buffer_id to integer
df_clean['buffer_id'] = df_clean['buffer_id'].astype(int)

# Plot timing components
plt.figure(figsize=(10, 6))
plt.plot(df_clean['buffer_id'], df_clean['H2D_ms'], label='H2D (ms)', marker='o')
plt.plot(df_clean['buffer_id'], df_clean['kernel_ms'], label='Kernel (ms)', marker='s')
plt.plot(df_clean['buffer_id'], df_clean['D2H_ms'], label='D2H (ms)', marker='^')

plt.title('CUDA Timing per Buffer')
plt.xlabel('Buffer ID')
plt.ylabel('Time (ms)')
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()
