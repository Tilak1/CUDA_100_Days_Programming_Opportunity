import pandas as pd
import matplotlib.pyplot as plt
import matplotlib
import numpy as np
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import cm

# Use non-interactive backend
matplotlib.use('Agg')

# Read CSV
df = pd.read_csv('cuda_graph_sweep_results.csv')

# Filter and prepare data
kernel_sweep = df[df['Total_Rays'] == 4000].copy()
kernel_sweep['Symbols_Per_Kernel'] = (kernel_sweep['Total_Rays'] * 20) / kernel_sweep['Kernels']
kernel_sweep_sorted = kernel_sweep.sort_values('Symbols_Per_Kernel').reset_index(drop=True)

# Create figure
fig = plt.figure(figsize=(14, 10))
ax = fig.add_subplot(111, projection='3d')

# Create 3D bars for launch overhead
xpos = np.arange(len(kernel_sweep_sorted))
ypos = np.zeros(len(kernel_sweep_sorted))
zpos = np.zeros(len(kernel_sweep_sorted))

dx = np.ones(len(kernel_sweep_sorted)) * 0.7
dy = np.ones(len(kernel_sweep_sorted)) * 0.7
dz = kernel_sweep_sorted['Launch_Overhead_Percent'].values

# Create colormap based on launch overhead percentage
# Use reverse RdYlBu so red = high overhead, blue = low overhead
norm = plt.Normalize(vmin=dz.min(), vmax=dz.max())
colors_map = cm.RdYlBu_r(norm(dz))

# Create 3D bars
bars = ax.bar3d(xpos, ypos, zpos, dx, dy, dz, 
                 color=colors_map, alpha=0.9, 
                 edgecolor='black', linewidth=0.5, 
                 zsort='average')

# Customize x-axis labels
# Show symbols/kernel and kernel count
x_labels = []
for idx in range(len(kernel_sweep_sorted)):
    sym_per_kern = kernel_sweep_sorted.iloc[idx]['Symbols_Per_Kernel']
    kernels = kernel_sweep_sorted.iloc[idx]['Kernels']
    
    if kernels >= 1000:
        kernel_str = f"{kernels/1000:.0f}k"
    else:
        kernel_str = f"{int(kernels)}"
    
    if sym_per_kern >= 1:
        x_labels.append(f"{sym_per_kern:.0f}\n({kernel_str})")
    else:
        x_labels.append(f"{sym_per_kern:.1f}\n({kernel_str})")

# Set x-axis
ax.set_xticks(xpos)
ax.set_xticklabels(x_labels, rotation=0, ha='center', fontsize=9)

# Hide y-axis since it's not used
ax.set_yticks([])
ax.set_ylim(-0.5, 0.5)

# Labels
ax.set_xlabel('Symbols/Kernel\n(Number of Kernels)', fontsize=14, labelpad=10)
ax.set_ylabel('')  # Empty since we're not using Y axis
ax.set_zlabel('Launch Overhead %', fontsize=14, labelpad=10)

# Title
ax.set_title('Launch Overhead by Configuration\nFading Rate Impact on Kernel Launch Cost', 
             fontsize=16, fontweight='bold', pad=20)

# Add grid for better readability
ax.grid(True, alpha=0.3, linestyle='--')

# Set viewing angle for best visibility
ax.view_init(elev=25, azim=-45)

# Add text annotations for highest overhead bars
for idx in range(len(kernel_sweep_sorted)):
    if kernel_sweep_sorted.iloc[idx]['Launch_Overhead_Percent'] > 5:
        ax.text(xpos[idx], 0, dz[idx] + 1, 
                f"{dz[idx]:.1f}%", 
                ha='center', va='bottom', fontsize=10, fontweight='bold')

# Create a colorbar
sm = plt.cm.ScalarMappable(cmap=cm.RdYlBu_r, norm=norm)
sm.set_array([])
cbar = fig.colorbar(sm, ax=ax, pad=0.1, shrink=0.6, aspect=10)
cbar.set_label('Launch Overhead Percentage', fontsize=12)
cbar.ax.tick_params(labelsize=10)

# Add informative text
info_text = (
    "Red bars: High launch overhead (graphs very beneficial)\n"
    "Blue bars: Low launch overhead (minimal graph benefit)\n"
    "Height: Percentage of total time spent in kernel launches"
)
ax.text2D(0.02, 0.98, info_text, transform=ax.transAxes, 
          fontsize=11, verticalalignment='top',
          bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))

# Adjust layout
plt.tight_layout()

# Save figure
plt.savefig('launch_overhead_3d_bars.png', dpi=300, bbox_inches='tight', facecolor='white')
print("Saved launch overhead 3D bar plot as 'launch_overhead_3d_bars.png'")

# Also create a cleaner version with fewer labels
fig2 = plt.figure(figsize=(12, 9))
ax2 = fig2.add_subplot(111, projection='3d')

# Same data but show fewer labels for clarity
bars2 = ax2.bar3d(xpos, ypos, zpos, dx, dy, dz, 
                  color=colors_map, alpha=0.9, 
                  edgecolor='black', linewidth=0.5)

# Show only every 3rd label
x_ticks_sparse = xpos[::3]
x_labels_sparse = [x_labels[i] for i in range(0, len(x_labels), 3)]

ax2.set_xticks(x_ticks_sparse)
ax2.set_xticklabels(x_labels_sparse, rotation=0, ha='center', fontsize=10)

# Hide y-axis
ax2.set_yticks([])
ax2.set_ylim(-0.5, 0.5)

# Labels
ax2.set_xlabel('Symbols per Kernel (Number of Kernels)', fontsize=13, labelpad=8)
ax2.set_zlabel('Launch Overhead %', fontsize=13, labelpad=8)
ax2.set_title('CUDA Kernel Launch Overhead Analysis', fontsize=15, fontweight='bold')

# Grid
ax2.grid(True, alpha=0.3)

# Viewing angle
ax2.view_init(elev=20, azim=-40)

# Add zone annotations
ax2.text(1, 0.5, 40, "Fast Fading\nZone", fontsize=11, ha='center', 
         color='darkred', fontweight='bold',
         bbox=dict(boxstyle='round', facecolor='white', alpha=0.7))

ax2.text(len(xpos)-2, 0.5, 5, "Slow Fading\nZone", fontsize=11, ha='center', 
         color='darkblue', fontweight='bold',
         bbox=dict(boxstyle='round', facecolor='white', alpha=0.7))

# Colorbar
cbar2 = fig2.colorbar(sm, ax=ax2, pad=0.05, shrink=0.6, aspect=10)
cbar2.set_label('Launch Overhead %', fontsize=12)

plt.tight_layout()
plt.savefig('launch_overhead_3d_bars_clean.png', dpi=300, bbox_inches='tight', facecolor='white')
print("Saved clean version as 'launch_overhead_3d_bars_clean.png'")

# Print summary
print("\nLaunch Overhead Summary:")
print(f"Maximum overhead: {dz.max():.1f}% ({int(kernel_sweep_sorted.loc[dz.argmax(), 'Kernels'])} kernels)")
print(f"Minimum overhead: {dz.min():.1f}% ({int(kernel_sweep_sorted.loc[dz.argmin(), 'Kernels'])} kernels)")
print(f"Overhead > 5%: {len(dz[dz > 5])} configurations")
print(f"Overhead > 1%: {len(dz[dz > 1])} configurations")