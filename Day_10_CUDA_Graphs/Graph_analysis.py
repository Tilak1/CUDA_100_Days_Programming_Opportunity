import pandas as pd
import matplotlib.pyplot as plt
import matplotlib
# Use non-interactive backend for headless setup
matplotlib.use('Agg')

# Read CSV
df = pd.read_csv('cuda_graph_sweep_results.csv')

# Create multiple plots for comprehensive analysis
fig, axes = plt.subplots(2, 2, figsize=(15, 12))
kernel_sweep = df[df['Total_Rays'] == 4000]

# 1. Launch Speedup vs Kernel Count
ax1 = axes[0, 0]
ax1.loglog(kernel_sweep['Kernels'], kernel_sweep['Launch_Speedup'], 'bo-', linewidth=2, markersize=8)
ax1.set_xlabel('Number of Kernels')
ax1.set_ylabel('Launch Speedup')
ax1.set_title('CUDA Graph Launch Speedup vs Kernel Count')
ax1.grid(True, alpha=0.3)
ax1.axhline(y=1, color='r', linestyle='--', alpha=0.5, label='No benefit line')

# 2. Launch Overhead Percentage vs Kernel Count
ax2 = axes[0, 1]
ax2.semilogx(kernel_sweep['Kernels'], kernel_sweep['Launch_Overhead_Percent'], 'ro-', linewidth=2, markersize=8)
ax2.set_xlabel('Number of Kernels')
ax2.set_ylabel('Launch Overhead %')
ax2.set_title('Launch Overhead as % of Total Time')
ax2.grid(True, alpha=0.3)
ax2.axhline(y=5, color='g', linestyle='--', alpha=0.5, label='5% threshold')

# 3. Total Speedup vs Kernel Count
ax3 = axes[1, 0]
ax3.semilogx(kernel_sweep['Kernels'], kernel_sweep['Total_Speedup'], 'go-', linewidth=2, markersize=8)
ax3.set_xlabel('Number of Kernels')
ax3.set_ylabel('Total Speedup')
ax3.set_title('Overall Performance Speedup with Graphs')
ax3.grid(True, alpha=0.3)
ax3.axhline(y=1, color='r', linestyle='--', alpha=0.5, label='No benefit line')

# 4. Time Breakdown
ax4 = axes[1, 1]
# Stack plot showing time distribution
kernel_counts = kernel_sweep['Kernels'].values
h2d_time = kernel_sweep['Standard_H2D_ms'].values
launch_time = kernel_sweep['Standard_Launch_ms'].values
exec_time = kernel_sweep['Standard_Exec_ms'].values
d2h_time = kernel_sweep['Standard_D2H_ms'].values

ax4.bar(range(len(kernel_counts)), h2d_time, label='H2D Transfer')
ax4.bar(range(len(kernel_counts)), launch_time, bottom=h2d_time, label='Launch Overhead')
ax4.bar(range(len(kernel_counts)), exec_time, bottom=h2d_time+launch_time, label='Kernel Execution')
ax4.bar(range(len(kernel_counts)), d2h_time, bottom=h2d_time+launch_time+exec_time, label='D2H Transfer')

ax4.set_xticks(range(len(kernel_counts)))
ax4.set_xticklabels([str(k) for k in kernel_counts], rotation=45, ha='right')
ax4.set_xlabel('Number of Kernels')
ax4.set_ylabel('Time (ms)')
ax4.set_title('Time Breakdown by Phase')
ax4.legend()

plt.tight_layout()
plt.savefig('cuda_graph_analysis.png', dpi=300, bbox_inches='tight')
print("Saved plot as 'cuda_graph_analysis.png'")

# Create additional detailed plots
plt.figure(figsize=(12, 8))

# Plot showing when graphs become beneficial
plt.subplot(2, 1, 1)
benefit_threshold = kernel_sweep[kernel_sweep['Launch_Overhead_Percent'] > 1.0]
if not benefit_threshold.empty:
    plt.axvline(x=benefit_threshold.iloc[0]['Kernels'], color='r', linestyle='--', 
                label=f"Benefit threshold: {benefit_threshold.iloc[0]['Kernels']} kernels")

plt.loglog(kernel_sweep['Kernels'], kernel_sweep['Standard_Launch_ms'], 'b^-', 
           label='Standard Launch Overhead', linewidth=2, markersize=8)
plt.loglog(kernel_sweep['Kernels'], kernel_sweep['Graph_Launch_ms'], 'gs-', 
           label='Graph Launch Overhead', linewidth=2, markersize=8)
plt.xlabel('Number of Kernels')
plt.ylabel('Launch Overhead (ms)')
plt.title('Launch Overhead Comparison: Standard vs Graph')
plt.legend()
plt.grid(True, alpha=0.3)

# Efficiency plot
plt.subplot(2, 1, 2)
kernel_sweep_sorted = kernel_sweep.sort_values('Kernels')
efficiency = (kernel_sweep_sorted['Standard_Exec_ms'] / kernel_sweep_sorted['Standard_Total_ms']) * 100
plt.semilogx(kernel_sweep_sorted['Kernels'], efficiency, 'mo-', linewidth=2, markersize=8)
plt.xlabel('Number of Kernels')
plt.ylabel('Kernel Execution Efficiency (%)')
plt.title('GPU Utilization Efficiency vs Kernel Granularity')
plt.grid(True, alpha=0.3)
plt.axhline(y=50, color='r', linestyle='--', alpha=0.5, label='50% efficiency')

plt.tight_layout()
plt.savefig('cuda_graph_detailed_analysis.png', dpi=300, bbox_inches='tight')
print("Saved plot as 'cuda_graph_detailed_analysis.png'")

# Generate summary statistics
print("\n=== Summary Statistics ===")
print(f"Kernel count range tested: {kernel_sweep['Kernels'].min()} - {kernel_sweep['Kernels'].max()}")
print(f"Maximum launch speedup: {kernel_sweep['Launch_Speedup'].max():.1f}x")
print(f"Maximum total speedup: {kernel_sweep['Total_Speedup'].max():.2f}x")

# Find optimal configuration
optimal = kernel_sweep.loc[kernel_sweep['Total_Speedup'].idxmax()]
print(f"\nOptimal configuration:")
print(f"  Kernels: {optimal['Kernels']}")
print(f"  Total speedup: {optimal['Total_Speedup']:.2f}x")
print(f"  Launch overhead: {optimal['Launch_Overhead_Percent']:.1f}%")

# Create a text report
with open('cuda_graph_analysis_report.txt', 'w') as f:
    f.write("CUDA Graph Performance Analysis Report\n")
    f.write("=" * 50 + "\n\n")
    
    f.write("Configuration Summary:\n")
    f.write(f"Total test configurations: {len(df)}\n")
    f.write(f"Fixed workload tests: {len(kernel_sweep)}\n\n")
    
    f.write("Key Findings:\n")
    f.write(f"1. Maximum launch speedup: {kernel_sweep['Launch_Speedup'].max():.1f}x\n")
    f.write(f"2. Launch overhead becomes significant (>5%) at {benefit_threshold.iloc[0]['Kernels'] if not benefit_threshold.empty else 'N/A'} kernels\n")
    f.write(f"3. Optimal kernel count for total performance: {optimal['Kernels']}\n")
    
    f.write("\nDetailed Results:\n")
    f.write(kernel_sweep.to_string())

print("\nGenerated files:")
print("  - cuda_graph_analysis.png")
print("  - cuda_graph_detailed_analysis.png")
print("  - cuda_graph_analysis_report.txt")