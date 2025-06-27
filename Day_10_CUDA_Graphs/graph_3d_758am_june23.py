import pandas as pd
import matplotlib.pyplot as plt
import matplotlib
import numpy as np
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import cm
from matplotlib.patches import Patch
# Use non-interactive backend for headless setup
matplotlib.use('Agg')

# Read CSV
df = pd.read_csv('cuda_graph_sweep_results.csv')

# Filter for 4000 ray configurations
kernel_sweep = df[df['Total_Rays'] == 4000].copy()
kernel_sweep['Symbols_Per_Kernel'] = (kernel_sweep['Total_Rays'] * 20) / kernel_sweep['Kernels']
kernel_sweep_sorted = kernel_sweep.sort_values('Symbols_Per_Kernel').reset_index(drop=True)

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

# 1. 3D Scatter Plot with Color Legend
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

# Add color legend for fading categories
legend_elements = [Patch(facecolor=color, edgecolor='black', label=category)
                  for category, color in colors.items()
                  if category in kernel_sweep_sorted['Fading_Category'].unique()]
ax1.legend(handles=legend_elements, loc='upper left', title='Fading Categories', 
          bbox_to_anchor=(0.0, 0.95), fontsize=10)

# 2. 3D Bar Plot with color scale
ax2 = fig.add_subplot(2, 2, 2, projection='3d')

# Create 3D bars for launch overhead
xpos = np.arange(len(kernel_sweep_sorted))
ypos = np.zeros(len(kernel_sweep_sorted))
zpos = np.zeros(len(kernel_sweep_sorted))

dx = np.ones(len(kernel_sweep_sorted)) * 0.8
dy = np.ones(len(kernel_sweep_sorted)) * 0.8
dz = kernel_sweep_sorted['Launch_Overhead_Percent'].values

# Color bars by overhead percentage
norm = plt.Normalize(vmin=dz.min(), vmax=dz.max())
colors_overhead = plt.cm.RdYlBu_r(norm(dz))

bars = ax2.bar3d(xpos, ypos, zpos, dx, dy, dz, color=colors_overhead, alpha=0.8, 
                  edgecolor='black', linewidth=0.5)

# Custom x-axis labels
ax2.set_xticks(xpos[::2])
ax2.set_xticklabels([f"{s:.1f}\n({int(k)}k)" if k >= 1000 else f"{s:.1f}\n({int(k)})" 
                     for s, k in zip(kernel_sweep_sorted['Symbols_Per_Kernel'][::2], 
                                     kernel_sweep_sorted['Kernels'][::2])], 
                    rotation=0, ha='center', fontsize=8)

ax2.set_xlabel('Symbols/Kernel (Kernels)', fontsize=12)
ax2.set_ylabel('', fontsize=12)
ax2.set_zlabel('Launch Overhead %', fontsize=12)
ax2.set_title('Launch Overhead by Configuration', fontsize=14, fontweight='bold')
ax2.set_ylim(-1, 1)

# Add colorbar for overhead percentage
sm = plt.cm.ScalarMappable(cmap=plt.cm.RdYlBu_r, norm=norm)
sm.set_array([])
cbar = plt.colorbar(sm, ax=ax2, pad=0.1, shrink=0.5, aspect=10)
cbar.set_label('Launch Overhead %', fontsize=10)
cbar.ax.tick_params(labelsize=8)

# 3. Enhanced 2D plot with color zones and legend
ax3 = fig.add_subplot(2, 2, 3)

# Add fading zone backgrounds with labels
zone_alpha = 0.15
ax3.axvspan(0.01, 1, alpha=zone_alpha, color='red', label='Fast Fading Zone')
ax3.axvspan(1, 20, alpha=zone_alpha, color='orange', label='Moderate Fading Zone')
ax3.axvspan(20, 100, alpha=zone_alpha, color='gold', label='Slow Fading Zone')
ax3.axvspan(100, 1000, alpha=zone_alpha, color='lightgreen', label='Block Fading Zone')
ax3.axvspan(1000, 100000, alpha=zone_alpha, color='lightblue', label='Quasi-Static Zone')

# Create bubble plot
bubble_sizes = (np.log10(kernel_sweep_sorted['Kernels'].values) + 1) * 100

# Plot points by category for legend
plotted_categories = set()
for idx, (_, row) in enumerate(kernel_sweep_sorted.iterrows()):
    color = colors[row['Fading_Category']]
    label = row['Fading_Category'] if row['Fading_Category'] not in plotted_categories else ""
    if label:
        plotted_categories.add(row['Fading_Category'])
    
    ax3.scatter(row['Symbols_Per_Kernel'], row['Launch_Speedup'], 
               s=bubble_sizes[idx], c=color, alpha=0.7, edgecolors='black', 
               linewidth=1, label=label)
    
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
ax3.set_title('Launch Speedup vs Fading Rate\n(bubble size ∝ log(kernels), numbers show kernel count)', 
              fontsize=14, fontweight='bold')
ax3.grid(True, alpha=0.3)
ax3.set_xlim(0.01, 100000)

# Create two separate legends
# First legend: Fading zones (already added via axvspan)
zone_legend = ax3.legend(loc='upper right', title='Fading Zones', 
                        frameon=True, fancybox=True, fontsize=9)

# Second legend: Bubble sizes
ax3_2 = ax3.add_artist(zone_legend)  # Keep the first legend
legend_kernels = [1, 10, 100, 1000, 10000, 100000]
legend_elements = []
for k in legend_kernels:
    if k <= kernel_sweep_sorted['Kernels'].max():
        size = (np.log10(k) + 1) * 100
        legend_elements.append(plt.scatter([], [], s=size, c='gray', alpha=0.5, 
                                          edgecolors='black', label=f'{k:,} kernels'))

bubble_legend = ax3.legend(handles=legend_elements, loc='lower left', 
                          title='Number of Kernels', frameon=True, fancybox=True, fontsize=9)

# 4. 3D Trajectory Plot with color gradient
ax4 = fig.add_subplot(2, 2, 4, projection='3d')

# Create parametric plot
t = np.arange(len(kernel_sweep_sorted))
x_traj = np.log10(kernel_sweep_sorted['Symbols_Per_Kernel'].values)
y_traj = np.log10(kernel_sweep_sorted['Kernels'].values)
z_traj = np.log10(kernel_sweep_sorted['Launch_Speedup'].values)

# Plot trajectory
ax4.plot(x_traj, y_traj, z_traj, 'k-', linewidth=2, alpha=0.6, label='Optimization Path')

# Add colored points with fading categories
for i, cat in enumerate(kernel_sweep_sorted['Fading_Category']):
    ax4.scatter(x_traj[i], y_traj[i], z_traj[i], 
               c=colors[cat], s=100, edgecolors='black', linewidth=1)

# Add start and end markers
ax4.scatter(x_traj[0], y_traj[0], z_traj[0], color='green', s=300, 
           marker='^', edgecolors='black', linewidth=2, label='Start: Few Large Kernels')
ax4.scatter(x_traj[-1], y_traj[-1], z_traj[-1], color='red', s=300, 
           marker='v', edgecolors='black', linewidth=2, label='End: Many Small Kernels')

# Add projections
ax4.plot(x_traj, y_traj, np.ones_like(z_traj) * z_traj.min(), 
         'gray', alpha=0.3, linewidth=1)
ax4.plot(x_traj, np.ones_like(y_traj) * y_traj.max(), z_traj, 
         'gray', alpha=0.3, linewidth=1)
ax4.plot(np.ones_like(x_traj) * x_traj.min(), y_traj, z_traj, 
         'gray', alpha=0.3, linewidth=1)

ax4.set_xlabel('Log10(Symbols/Kernel)', fontsize=12)
ax4.set_ylabel('Log10(Kernels)', fontsize=12)
ax4.set_zlabel('Log10(Speedup)', fontsize=12)
ax4.set_title('Parameter Space Trajectory\nFrom Coarse to Fine Granularity', 
              fontsize=14, fontweight='bold')

# Add legend
ax4.legend(loc='upper left', fontsize=10)

# Add fading category color legend
legend_elements_3d = [Patch(facecolor=color, edgecolor='black', label=category)
                     for category, color in colors.items()
                     if category in kernel_sweep_sorted['Fading_Category'].unique()]
ax4.legend(handles=ax4.get_legend_handles_labels()[0] + legend_elements_3d, 
          loc='upper left', fontsize=9, title='Legend')

ax4.view_init(elev=20, azim=-60)

plt.suptitle('CUDA Graph Performance: 3D Analysis with Kernel Count Dimension\n' +
             'Color Coding: Red=Fast Fading, Orange=Moderate, Gold=Slow, Green=Block, Blue=Quasi-Static', 
             fontsize=16, fontweight='bold')
plt.tight_layout()
plt.savefig('cuda_fading_3d_analysis_with_legends.png', dpi=300, bbox_inches='tight')
print("Saved 3D analysis with color legends as 'cuda_fading_3d_analysis_with_legends.png'")

# Create a reference card for color coding
fig_ref = plt.figure(figsize=(10, 6))
ax_ref = fig_ref.add_subplot(111)
ax_ref.axis('off')

# Title
ax_ref.text(0.5, 0.95, 'CUDA Graph Analysis - Color Coding Reference', 
            ha='center', va='top', fontsize=18, fontweight='bold')

# Fading categories color legend
y_pos = 0.8
ax_ref.text(0.1, y_pos, 'Fading Categories:', fontsize=14, fontweight='bold')
y_pos -= 0.08

for category, color in colors.items():
    rect = plt.Rectangle((0.1, y_pos), 0.05, 0.05, facecolor=color, 
                        edgecolor='black', linewidth=1)
    ax_ref.add_patch(rect)
    ax_ref.text(0.17, y_pos + 0.025, category, va='center', fontsize=12)
    y_pos -= 0.08

# Launch overhead color scale
y_pos = 0.8
ax_ref.text(0.6, y_pos, 'Launch Overhead Scale:', fontsize=14, fontweight='bold')
y_pos -= 0.08

# Create mini colorbar
gradient = np.linspace(0, 100, 256).reshape(256, 1)
gradient = np.hstack((gradient, gradient))

ax_mini = fig_ref.add_axes([0.6, y_pos - 0.2, 0.3, 0.15])
ax_mini.imshow(gradient.T, aspect='auto', cmap=plt.cm.RdYlBu_r, extent=[0, 100, 0, 1])
ax_mini.set_xlabel('Launch Overhead %', fontsize=12)
ax_mini.set_yticks([])
ax_mini.set_xticks([0, 25, 50, 75, 100])

# Add interpretation
y_pos -= 0.3
ax_ref.text(0.1, y_pos, 'Interpretation:', fontsize=14, fontweight='bold')
y_pos -= 0.08
ax_ref.text(0.1, y_pos, '• Fast Fading (Red): H changes every symbol - Maximum graph benefit', 
            fontsize=11)
y_pos -= 0.06
ax_ref.text(0.1, y_pos, '• Moderate Fading (Orange): H changes every few symbols', 
            fontsize=11)
y_pos -= 0.06
ax_ref.text(0.1, y_pos, '• Slow Fading (Gold): H changes every 20-100 symbols', 
            fontsize=11)
y_pos -= 0.06
ax_ref.text(0.1, y_pos, '• Block Fading (Green): H constant for blocks of symbols', 
            fontsize=11)
y_pos -= 0.06
ax_ref.text(0.1, y_pos, '• Quasi-Static (Blue): H rarely changes - Minimal graph benefit', 
            fontsize=11)

plt.savefig('cuda_color_coding_reference.png', dpi=300, bbox_inches='tight', facecolor='white')
print("Saved color coding reference as 'cuda_color_coding_reference.png'")

print("\n✅ All plots generated with proper color coding!")
print("\nColor scheme summary:")
print("- Fading categories use distinct colors (red → blue)")
print("- Launch overhead uses gradient (blue=low, red=high)")
print("- All legends and color bars are properly labeled")