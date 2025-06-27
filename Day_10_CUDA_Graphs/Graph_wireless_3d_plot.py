import pandas as pd
import matplotlib.pyplot as plt
import matplotlib
import numpy as np
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import cm
# Use non-interactive backend for headless setup
matplotlib.use('Agg')

# Read CSV
df = pd.read_csv('cuda_graph_sweep_results.csv')

# Filter for 4000 ray configurations
kernel_sweep = df[df['Total_Rays'] == 4000].copy()
kernel_sweep['Symbols_Per_Kernel'] = (kernel_sweep['Total_Rays'] * 20) / kernel_sweep['Kernels']
kernel_sweep_sorted = kernel_sweep.sort_values('Symbols_Per_Kernel').reset_index(drop=True)  # Reset index!

# Define fading categories
def categorize_fading(symbols_per_kernel):
    if symbols_per_kernel <= 1:
        return 'Fast Fading (≤1 symbol)'
    elif symbols_per_kernel <= 20:
        return 'Moderate Fading (1-20 symbols)'
    elif symbols_per_kernel <= 100:
        return 'Slow Fading (20-100 symbols)'
    elif symbols_per_kernel <= 1000:
        return 'Block Fading (100-1000 symbols)'
    else:
        return 'Quasi-Static (>1000 symbols)'

kernel_sweep_sorted['Fading_Category'] = kernel_sweep_sorted['Symbols_Per_Kernel'].apply(categorize_fading)

# Create figure with 3D subplot
fig = plt.figure(figsize=(20, 16))

# Define color scheme
colors = {'Fast Fading (≤1 symbol)': 'red', 
          'Moderate Fading (1-20 symbols)': 'orange',
          'Slow Fading (20-100 symbols)': 'gold',
          'Block Fading (100-1000 symbols)': 'lightgreen',
          'Quasi-Static (>1000 symbols)': 'lightblue'}

# 1. 3D Scatter Plot: Symbols per Kernel vs Kernels vs Launch Speedup
ax1 = fig.add_subplot(2, 2, 1, projection='3d')

# Prepare data for 3D plot
x = np.log10(kernel_sweep_sorted['Symbols_Per_Kernel'].values)
y = np.log10(kernel_sweep_sorted['Kernels'].values)
z = np.log10(kernel_sweep_sorted['Launch_Speedup'].values)

# Create color array based on fading category
color_array = [colors[cat] for cat in kernel_sweep_sorted['Fading_Category']]

# Create 3D scatter plot
scatter = ax1.scatter(x, y, z, c=color_array, s=200, alpha=0.8, edgecolors='black', linewidth=1)

# Add text labels for each point
for idx in range(len(kernel_sweep_sorted)):
    row = kernel_sweep_sorted.iloc[idx]
    ax1.text(np.log10(row['Symbols_Per_Kernel']), 
             np.log10(row['Kernels']), 
             np.log10(row['Launch_Speedup']) + 0.05,
             f"{int(row['Kernels'])}", 
             fontsize=8, ha='center')

# Connect points with lines
for i in range(len(x)-1):
    ax1.plot([x[i], x[i+1]], [y[i], y[i+1]], [z[i], z[i+1]], 
             'k-', alpha=0.3, linewidth=1)

# Set labels
ax1.set_xlabel('Log10(Symbols per Kernel)', fontsize=12, labelpad=10)
ax1.set_ylabel('Log10(Number of Kernels)', fontsize=12, labelpad=10)
ax1.set_zlabel('Log10(Launch Speedup)', fontsize=12, labelpad=10)
ax1.set_title('3D View: Fading Rate vs Kernel Count vs Speedup', fontsize=14, fontweight='bold')

# Add grid
ax1.grid(True, alpha=0.3)

# Rotate for better view
ax1.view_init(elev=20, azim=45)

# 2. 3D Bar Plot instead of surface (to avoid interpolation issues)
ax2 = fig.add_subplot(2, 2, 2, projection='3d')

# Create 3D bars for launch overhead
xpos = np.arange(len(kernel_sweep_sorted))
ypos = np.zeros(len(kernel_sweep_sorted))
zpos = np.zeros(len(kernel_sweep_sorted))

dx = np.ones(len(kernel_sweep_sorted)) * 0.8
dy = np.ones(len(kernel_sweep_sorted)) * 0.8
dz = kernel_sweep_sorted['Launch_Overhead_Percent'].values

# Color bars by overhead percentage
colors_overhead = plt.cm.RdYlBu_r(dz / dz.max())

bars = ax2.bar3d(xpos, ypos, zpos, dx, dy, dz, color=colors_overhead, alpha=0.8, 
                  edgecolor='black', linewidth=0.5)

# Custom x-axis labels
ax2.set_xticks(xpos[::2])  # Show every other label to avoid crowding
ax2.set_xticklabels([f"{s:.1f}\n({int(k)}k)" if k >= 1000 else f"{s:.1f}\n({int(k)})" 
                     for s, k in zip(kernel_sweep_sorted['Symbols_Per_Kernel'][::2], 
                                     kernel_sweep_sorted['Kernels'][::2])], 
                    rotation=0, ha='center', fontsize=8)

ax2.set_xlabel('Symbols/Kernel (Kernels)', fontsize=12)
ax2.set_ylabel('', fontsize=12)
ax2.set_zlabel('Launch Overhead %', fontsize=12)
ax2.set_title('Launch Overhead by Configuration', fontsize=14, fontweight='bold')
ax2.set_ylim(-1, 1)

# 3. Enhanced 2D plot with 3D information
ax3 = fig.add_subplot(2, 2, 3)

# Add fading zone backgrounds
ax3.axvspan(0.01, 1, alpha=0.15, color='red')
ax3.axvspan(1, 20, alpha=0.15, color='orange')
ax3.axvspan(20, 100, alpha=0.15, color='gold')
ax3.axvspan(100, 1000, alpha=0.15, color='lightgreen')
ax3.axvspan(1000, 100000, alpha=0.15, color='lightblue')

# Create bubble plot where size represents number of kernels
# Use log scale for bubble sizes for better visualization
bubble_sizes = (np.log10(kernel_sweep_sorted['Kernels'].values) + 1) * 100

# Use enumerate to ensure proper indexing
for idx, (_, row) in enumerate(kernel_sweep_sorted.iterrows()):
    color = colors[row['Fading_Category']]
    ax3.scatter(row['Symbols_Per_Kernel'], row['Launch_Speedup'], 
               s=bubble_sizes[idx], c=color, alpha=0.7, edgecolors='black', linewidth=1)
    # Add kernel count annotation
    ax3.annotate(f"{int(row['Kernels'])}", 
                xy=(row['Symbols_Per_Kernel'], row['Launch_Speedup']),
                xytext=(0, 0), textcoords='offset points',
                fontsize=9, ha='center', va='center', weight='bold')

# Connect points with lines
ax3.loglog(kernel_sweep_sorted['Symbols_Per_Kernel'], 
           kernel_sweep_sorted['Launch_Speedup'], 
           'k--', alpha=0.3, linewidth=1)

ax3.set_xscale('log')
ax3.set_yscale('log')
ax3.set_xlabel('Symbols per Kernel (Fading Rate)', fontsize=12)
ax3.set_ylabel('Launch Speedup', fontsize=12)
ax3.set_title('Launch Speedup vs Fading Rate\n(bubble size ∝ log(kernels))', 
              fontsize=14, fontweight='bold')
ax3.grid(True, alpha=0.3)
ax3.set_xlim(0.01, 100000)

# Add legend for bubble sizes
legend_kernels = [1, 10, 100, 1000, 10000, 100000]
legend_elements = []
for k in legend_kernels:
    if k <= kernel_sweep_sorted['Kernels'].max():  # Only show relevant sizes
        size = (np.log10(k) + 1) * 100
        legend_elements.append(plt.scatter([], [], s=size, c='gray', alpha=0.5, 
                                          edgecolors='black', label=f'{k:,} kernels'))
ax3.legend(handles=legend_elements, loc='lower left', title='Number of Kernels', 
           frameon=True, fancybox=True)

# 4. 3D Trajectory Plot
ax4 = fig.add_subplot(2, 2, 4, projection='3d')

# Create a parametric plot showing the path through parameter space
t = np.arange(len(kernel_sweep_sorted))
x_traj = np.log10(kernel_sweep_sorted['Symbols_Per_Kernel'].values)
y_traj = np.log10(kernel_sweep_sorted['Kernels'].values)
z_traj = np.log10(kernel_sweep_sorted['Launch_Speedup'].values)

# Plot the trajectory
ax4.plot(x_traj, y_traj, z_traj, 'k-', linewidth=2, alpha=0.6)

# Add colored points along the trajectory
scatter = ax4.scatter(x_traj, y_traj, z_traj, c=t, cmap='viridis', 
                     s=100, edgecolors='black', linewidth=1)

# Add start and end markers
ax4.scatter(x_traj[0], y_traj[0], z_traj[0], color='green', s=300, 
           marker='^', edgecolors='black', linewidth=2, label='Start (few large kernels)')
ax4.scatter(x_traj[-1], y_traj[-1], z_traj[-1], color='red', s=300, 
           marker='v', edgecolors='black', linewidth=2, label='End (many small kernels)')

# Add projections to the walls
# XY plane projection
ax4.plot(x_traj, y_traj, np.ones_like(z_traj) * z_traj.min(), 
         'gray', alpha=0.3, linewidth=1)
# XZ plane projection  
ax4.plot(x_traj, np.ones_like(y_traj) * y_traj.max(), z_traj, 
         'gray', alpha=0.3, linewidth=1)
# YZ plane projection
ax4.plot(np.ones_like(x_traj) * x_traj.min(), y_traj, z_traj, 
         'gray', alpha=0.3, linewidth=1)

ax4.set_xlabel('Log10(Symbols/Kernel)', fontsize=12)
ax4.set_ylabel('Log10(Kernels)', fontsize=12)
ax4.set_zlabel('Log10(Speedup)', fontsize=12)
ax4.set_title('Parameter Space Trajectory\nFrom Coarse to Fine Granularity', 
              fontsize=14, fontweight='bold')
ax4.legend(loc='upper left')

# Add colorbar
cbar = plt.colorbar(scatter, ax=ax4, pad=0.1, shrink=0.5)
cbar.set_label('Configuration Index', fontsize=10)

# Set viewing angle
ax4.view_init(elev=20, azim=-60)

plt.suptitle('CUDA Graph Performance: 3D Analysis with Kernel Count Dimension', 
             fontsize=16, fontweight='bold')
plt.tight_layout()
plt.savefig('cuda_fading_3d_analysis_fixed.png', dpi=300, bbox_inches='tight')
print("Saved 3D analysis as 'cuda_fading_3d_analysis_fixed.png'")

# Create a focused single 3D plot
fig2 = plt.figure(figsize=(14, 10))
ax = fig2.add_subplot(111, projection='3d')

# Main 3D scatter with annotations
x = np.log10(kernel_sweep_sorted['Symbols_Per_Kernel'].values)
y = np.log10(kernel_sweep_sorted['Kernels'].values)
z = kernel_sweep_sorted['Launch_Speedup'].values  # Don't log transform speedup for clarity

# Create scatter plot with gradient coloring
scatter = ax.scatter(x, y, z, c=z, cmap='plasma', s=200, 
                    edgecolors='black', linewidth=1, alpha=0.9)

# Add vertical lines from points to base
for i in range(len(x)):
    ax.plot([x[i], x[i]], [y[i], y[i]], [0, z[i]], 
            'gray', alpha=0.3, linewidth=0.5)

# Annotate key points
annotations = [
    (0, "Quasi-Static\n1 kernel"),
    (len(x)//4, "Block Fading"),
    (len(x)//2, "Slow Fading"),
    (3*len(x)//4, "Fast Fading"),
    (len(x)-1, f"Ultra-Fast\n{int(kernel_sweep_sorted['Kernels'].iloc[-1]):,} kernels")
]

for idx, label in annotations:
    if idx < len(x):
        ax.text(x[idx], y[idx], z[idx] + z.max()*0.05,
               label, fontsize=10, ha='center',
               bbox=dict(boxstyle="round,pad=0.3", facecolor="white", alpha=0.8))

ax.set_xlabel('Log10(Symbols per Kernel)\n← More Granular | Less Granular →', fontsize=12)
ax.set_ylabel('Log10(Number of Kernels)\n← Fewer | More →', fontsize=12)
ax.set_zlabel('Launch Speedup\n← Lower | Higher →', fontsize=12)
ax.set_title('3D Visualization: How Kernel Granularity Affects CUDA Graph Performance\n' +
             'Higher points = Better graph performance', 
             fontsize=14, fontweight='bold')

# Add colorbar
cbar = plt.colorbar(scatter, ax=ax, pad=0.1, shrink=0.7)
cbar.set_label('Launch Speedup Factor', fontsize=12)

# Set optimal viewing angle
ax.view_init(elev=25, azim=135)
ax.grid(True, alpha=0.3)

# Set z-axis to start from 0
ax.set_zlim(bottom=0)

plt.tight_layout()
plt.savefig('cuda_fading_3d_main_fixed.png', dpi=300, bbox_inches='tight')
print("Saved main 3D plot as 'cuda_fading_3d_main_fixed.png'")

# Print summary statistics
print("\n✅ 3D plots generated successfully!")
print(f"\nData summary:")
print(f"- Total configurations analyzed: {len(kernel_sweep_sorted)}")
print(f"- Kernel range: {kernel_sweep_sorted['Kernels'].min()} to {kernel_sweep_sorted['Kernels'].max():,}")
print(f"- Symbols/kernel range: {kernel_sweep_sorted['Symbols_Per_Kernel'].min():.1f} to {kernel_sweep_sorted['Symbols_Per_Kernel'].max():.1f}")
print(f"- Max launch speedup: {kernel_sweep_sorted['Launch_Speedup'].max():.1f}x")
print("\nKey insights from 3D visualization:")
print("- X-axis (Symbols/Kernel): Represents fading rate")
print("- Y-axis (Kernels): Shows granularity of work")
print("- Z-axis (Speedup): Shows CUDA graph benefit")
print("- The trajectory shows optimal path from coarse to fine granularity")