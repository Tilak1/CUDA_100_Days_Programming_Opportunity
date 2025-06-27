import pandas as pd
import matplotlib.pyplot as plt
import matplotlib
import numpy as np
from matplotlib.patches import Rectangle
# Use non-interactive backend for headless setup
matplotlib.use('Agg')

# Read CSV
df = pd.read_csv('cuda_graph_sweep_results.csv')

# Filter for 4000 ray configurations and calculate symbols per kernel
kernel_sweep = df[df['Total_Rays'] == 4000].copy()
kernel_sweep['Symbols_Per_Kernel'] = (kernel_sweep['Total_Rays'] * 20) / kernel_sweep['Kernels']

# Sort by symbols per kernel (ascending - from fast fading to slow fading)
kernel_sweep_sorted = kernel_sweep.sort_values('Symbols_Per_Kernel')

# Define fading categories based on symbols per kernel
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

# Create comprehensive fading-oriented plots
fig = plt.figure(figsize=(20, 16))

# Define color scheme
colors = {'Fast Fading (≤1 symbol)': 'red', 
          'Moderate Fading (1-20 symbols)': 'orange',
          'Slow Fading (20-100 symbols)': 'gold',
          'Block Fading (100-1000 symbols)': 'lightgreen',
          'Quasi-Static (>1000 symbols)': 'lightblue'}

# 1. Launch Speedup vs Symbols per Kernel with kernel counts
ax1 = plt.subplot(3, 2, 1)

# Add fading zone backgrounds
ax1.axvspan(0.01, 1, alpha=0.15, color='red', label='Fast Fading Zone')
ax1.axvspan(1, 20, alpha=0.15, color='orange', label='Moderate Fading Zone')
ax1.axvspan(20, 100, alpha=0.15, color='gold', label='Slow Fading Zone')
ax1.axvspan(100, 1000, alpha=0.15, color='lightgreen', label='Block Fading Zone')
ax1.axvspan(1000, 100000, alpha=0.15, color='lightblue', label='Quasi-Static Zone')

# Plot data points with kernel count annotations
for _, row in kernel_sweep_sorted.iterrows():
    color = colors[row['Fading_Category']]
    ax1.loglog(row['Symbols_Per_Kernel'], row['Launch_Speedup'], 
               'o', color=color, markersize=10)
    # Add kernel count annotation
    ax1.annotate(f"{int(row['Kernels'])}", 
                xy=(row['Symbols_Per_Kernel'], row['Launch_Speedup']),
                xytext=(5, 5), textcoords='offset points',
                fontsize=8, ha='left')

# Connect points with lines
ax1.loglog(kernel_sweep_sorted['Symbols_Per_Kernel'], 
           kernel_sweep_sorted['Launch_Speedup'], 
           'k-', alpha=0.3, linewidth=1)

ax1.set_xlabel('Symbols per Kernel (Fading Rate)', fontsize=12)
ax1.set_ylabel('Launch Speedup', fontsize=12)
ax1.set_title('CUDA Graph Benefits vs Fading Scenarios\n(numbers indicate kernels launched)', 
              fontsize=14, fontweight='bold')
ax1.grid(True, alpha=0.3)
ax1.set_xlim(0.01, 100000)

# 2. Launch Overhead Percentage with fading zones
ax2 = plt.subplot(3, 2, 2)

# Add fading zone backgrounds
ax2.axvspan(0.01, 1, alpha=0.15, color='red')
ax2.axvspan(1, 20, alpha=0.15, color='orange')
ax2.axvspan(20, 100, alpha=0.15, color='gold')
ax2.axvspan(100, 1000, alpha=0.15, color='lightgreen')
ax2.axvspan(1000, 100000, alpha=0.15, color='lightblue')

# Plot with kernel count labels
for _, row in kernel_sweep_sorted.iterrows():
    ax2.semilogx(row['Symbols_Per_Kernel'], row['Launch_Overhead_Percent'], 
                 'ko', markersize=8)
    if row['Launch_Overhead_Percent'] > 1:  # Only label significant overhead
        ax2.annotate(f"{int(row['Kernels'])}k" if row['Kernels'] >= 1000 else f"{int(row['Kernels'])}", 
                    xy=(row['Symbols_Per_Kernel'], row['Launch_Overhead_Percent']),
                    xytext=(0, 5), textcoords='offset points',
                    fontsize=8, ha='center')

ax2.semilogx(kernel_sweep_sorted['Symbols_Per_Kernel'], 
             kernel_sweep_sorted['Launch_Overhead_Percent'], 
             'k-', linewidth=2)

ax2.set_xlabel('Symbols per Kernel (Fading Rate)', fontsize=12)
ax2.set_ylabel('Launch Overhead %', fontsize=12)
ax2.set_title('Launch Overhead Impact Across Fading Scenarios', fontsize=14, fontweight='bold')
ax2.grid(True, alpha=0.3)
ax2.axhline(y=5, color='red', linestyle='--', alpha=0.7, linewidth=2, label='5% threshold')
ax2.axhline(y=1, color='orange', linestyle='--', alpha=0.7, linewidth=2, label='1% threshold')
ax2.set_xlim(0.01, 100000)
ax2.legend(loc='upper right')

# 3. Detailed Bar Chart with Kernel Counts
ax3 = plt.subplot(3, 2, 3)
x_pos = np.arange(len(kernel_sweep_sorted))
width = 0.35

bars1 = ax3.bar(x_pos - width/2, kernel_sweep_sorted['Standard_Total_ms'], 
                 width, label='Standard Launch', color='darkblue', alpha=0.7)
bars2 = ax3.bar(x_pos + width/2, kernel_sweep_sorted['Graph_Total_ms'], 
                 width, label='Graph Launch', color='darkgreen', alpha=0.7)

# Create custom x-axis labels with symbols/kernel and kernel count
custom_labels = []
for _, row in kernel_sweep_sorted.iterrows():
    if row['Kernels'] >= 1000:
        kernel_str = f"{row['Kernels']/1000:.0f}k"
    else:
        kernel_str = f"{int(row['Kernels'])}"
    custom_labels.append(f"{row['Symbols_Per_Kernel']:.1f} sym\n({kernel_str} kern)")

ax3.set_xlabel('Configuration\n(symbols/kernel and total kernels)', fontsize=12)
ax3.set_ylabel('Total Time (ms)', fontsize=12)
ax3.set_title('Total Execution Time: Standard vs Graph', fontsize=14, fontweight='bold')
ax3.set_xticks(x_pos)
ax3.set_xticklabels(custom_labels, rotation=45, ha='right', fontsize=9)
ax3.legend()
ax3.grid(True, axis='y', alpha=0.3)

# Add speedup text above bars
for i, (s1, s2) in enumerate(zip(bars1, bars2)):
    speedup = s1.get_height() / s2.get_height()
    ax3.text(i, max(s1.get_height(), s2.get_height()) + 10, 
             f'{speedup:.1f}x', ha='center', fontsize=8)

# 4. Kernel Count vs Speedup Relationship
ax4 = plt.subplot(3, 2, 4)

# Create scatter plot with fading categories
for category in colors:
    data = kernel_sweep_sorted[kernel_sweep_sorted['Fading_Category'] == category]
    if not data.empty:
        ax4.scatter(data['Kernels'], data['Total_Speedup'], 
                   color=colors[category], s=100, alpha=0.7, 
                   edgecolors='black', label=category)

ax4.set_xscale('log')
ax4.set_xlabel('Number of Kernels Launched', fontsize=12)
ax4.set_ylabel('Total Speedup', fontsize=12)
ax4.set_title('Performance Speedup vs Kernel Count', fontsize=14, fontweight='bold')
ax4.grid(True, alpha=0.3)
ax4.legend(loc='best', fontsize=10)

# Add reference lines
ax4.axhline(y=1, color='red', linestyle='--', alpha=0.5)
ax4.axvline(x=40, color='gray', linestyle=':', alpha=0.5, label='Typical threshold')

# 5. Fading Scenario Summary Table
ax5 = plt.subplot(3, 2, 5)
ax5.axis('off')

# Create summary table data
summary_data = []
for category in ['Fast Fading (≤1 symbol)', 'Moderate Fading (1-20 symbols)', 
                 'Slow Fading (20-100 symbols)', 'Block Fading (100-1000 symbols)', 
                 'Quasi-Static (>1000 symbols)']:
    cat_data = kernel_sweep_sorted[kernel_sweep_sorted['Fading_Category'] == category]
    if not cat_data.empty:
        # Find configuration with max speedup
        optimal = cat_data.loc[cat_data['Total_Speedup'].idxmax()]
        # Find range of kernels tested
        kernel_range = f"{cat_data['Kernels'].min()}-{cat_data['Kernels'].max()}"
        if cat_data['Kernels'].min() == cat_data['Kernels'].max():
            kernel_range = str(cat_data['Kernels'].min())
        
        summary_data.append([
            category.split(' (')[0],  # Just the category name
            kernel_range,
            f"{optimal['Kernels']}",
            f"{optimal['Launch_Speedup']:.1f}x",
            f"{optimal['Total_Speedup']:.2f}x"
        ])

if summary_data:
    table = ax5.table(cellText=summary_data,
                     colLabels=['Fading Type', 'Kernels Tested', 'Optimal Kernels', 
                               'Max Launch\nSpeedup', 'Max Total\nSpeedup'],
                     cellLoc='center',
                     loc='center',
                     colWidths=[0.25, 0.2, 0.15, 0.2, 0.2])
    
    table.auto_set_font_size(False)
    table.set_fontsize(11)
    table.scale(1.2, 2.0)
    
    # Color code rows
    for i, cat in enumerate(['Fast Fading', 'Moderate Fading', 'Slow Fading', 
                            'Block Fading', 'Quasi-Static']):
        color = list(colors.values())[i]
        for j in range(5):
            table[(i+1, j)].set_facecolor(color)
            table[(i+1, j)].set_alpha(0.3)

ax5.set_title('Fading Scenario Performance Summary', fontsize=14, fontweight='bold')

# 6. Launch Overhead Time vs Kernel Count
ax6 = plt.subplot(3, 2, 6)

# Plot both standard and graph launch times
ax6.loglog(kernel_sweep_sorted['Kernels'], 
           kernel_sweep_sorted['Standard_Launch_ms'], 
           'ro-', label='Standard Launch Time', markersize=8, linewidth=2)
ax6.loglog(kernel_sweep_sorted['Kernels'], 
           kernel_sweep_sorted['Graph_Launch_ms'], 
           'go-', label='Graph Launch Time', markersize=8, linewidth=2)

# Add annotations for key points
for _, row in kernel_sweep_sorted.iterrows():
    if row['Kernels'] in [1, 4, 40, 400, 4000, 40000]:
        ax6.annotate(f"{row['Standard_Launch_ms']:.1f}ms", 
                    xy=(row['Kernels'], row['Standard_Launch_ms']),
                    xytext=(5, 5), textcoords='offset points',
                    fontsize=8, color='red')

ax6.set_xlabel('Number of Kernels', fontsize=12)
ax6.set_ylabel('Launch Time (ms)', fontsize=12)
ax6.set_title('Launch Time Scaling with Kernel Count', fontsize=14, fontweight='bold')
ax6.grid(True, alpha=0.3)
ax6.legend()

# Add ideal scaling line
kernels = np.array([1, 100000])
ideal_scaling = kernels * 0.0025  # Assuming 2.5us per kernel
ax6.loglog(kernels, ideal_scaling, 'k--', alpha=0.5, label='Ideal linear scaling')

plt.suptitle('CUDA Graph Performance Analysis for Different Fading Scenarios', 
             fontsize=16, fontweight='bold')
plt.tight_layout()
plt.savefig('cuda_fading_analysis_enhanced.png', dpi=300, bbox_inches='tight')
print("Saved enhanced plot as 'cuda_fading_analysis_enhanced.png'")

# Generate detailed text report with kernel counts
with open('cuda_fading_kernel_analysis.txt', 'w') as f:
    f.write("CUDA GRAPH PERFORMANCE - FADING SCENARIOS WITH KERNEL COUNTS\n")
    f.write("=" * 80 + "\n\n")
    
    for category in ['Fast Fading (≤1 symbol)', 'Moderate Fading (1-20 symbols)', 
                     'Slow Fading (20-100 symbols)', 'Block Fading (100-1000 symbols)', 
                     'Quasi-Static (>1000 symbols)']:
        cat_data = kernel_sweep_sorted[kernel_sweep_sorted['Fading_Category'] == category]
        if not cat_data.empty:
            f.write(f"\n{category}:\n")
            f.write("-" * len(category) + "\n")
            for _, row in cat_data.iterrows():
                f.write(f"  {int(row['Kernels']):6d} kernels ({row['Symbols_Per_Kernel']:6.1f} sym/kernel): "
                       f"Launch speedup = {row['Launch_Speedup']:6.1f}x, "
                       f"Total speedup = {row['Total_Speedup']:5.2f}x, "
                       f"Launch OH = {row['Launch_Overhead_Percent']:5.1f}%\n")

print("\nGenerated files:")
print("  - cuda_fading_analysis_enhanced.png (with kernel counts and fading zones)")
print("  - cuda_fading_kernel_analysis.txt (detailed breakdown)")