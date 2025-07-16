import numpy as np
import matplotlib.pyplot as plt
from scipy import stats
import seaborn as sns

# Set style for better visuals
plt.style.use('seaborn-v0_8-whitegrid')
sns.set_palette("husl")

def plot_cdf_with_logit(data, title="Cumulative Distribution Function", 
                       xlabel="Number of Pulses", save_path=None):
    """
    Plot CDF with logit scale for y-axis and improved visuals
    """
    # Remove zeros for log scale on x-axis
    data_filtered = data[data > 0]
    
    # Sort data for CDF calculation
    sorted_data = np.sort(data_filtered)
    n = len(sorted_data)
    
    # Calculate cumulative probabilities
    cumulative_prob = np.arange(1, n + 1) / n
    
    # Create figure with better styling
    fig, ax = plt.subplots(figsize=(12, 8), facecolor='white')
    
    # Plot CDF
    ax.plot(sorted_data, cumulative_prob, linewidth=2.5, color='#2E86AB', 
            alpha=0.8, label='Empirical CDF')
    
    # Set scales
    ax.set_xscale('log')
    ax.set_yscale('logit')
    
    # Calculate and mark key percentiles
    percentiles = [50, 90, 95, 99, 99.9]
    colors = ['#A23B72', '#F18F01', '#C73E1D', '#8B0000', '#4B0000']
    
    for i, perc in enumerate(percentiles):
        value = np.percentile(data_filtered, perc)
        prob = perc / 100
        
        # Vertical line
        ax.axvline(x=value, color=colors[i], linestyle='--', linewidth=2, 
                  alpha=0.7, label=f'{perc}th percentile')
        
        # Horizontal line
        ax.axhline(y=prob, color=colors[i], linestyle=':', linewidth=1.5, alpha=0.5)
        
        # Add text annotation
        ax.annotate(f'{perc}%\n({value:.0f})', 
                   xy=(value, prob), xytext=(10, 10), 
                   textcoords='offset points', fontsize=10,
                   bbox=dict(boxstyle='round,pad=0.3', facecolor=colors[i], alpha=0.3),
                   ha='left', va='bottom')
    
    # Add reference line for 1024
    ax.axvline(x=1024, color='#2D5016', linestyle='-', linewidth=2, 
              alpha=0.8, label='Reference: 1024')
    
    # Styling improvements
    ax.set_title(title, fontsize=16, fontweight='bold', pad=20)
    ax.set_xlabel(xlabel, fontsize=14, fontweight='medium')
    ax.set_ylabel('Cumulative Probability (logit scale)', fontsize=14, fontweight='medium')
    
    # Grid improvements
    ax.grid(True, which="major", alpha=0.3, linewidth=0.8)
    ax.grid(True, which="minor", alpha=0.1, linewidth=0.5)
    
    # Legend improvements
    ax.legend(loc='lower right', fontsize=11, frameon=True, 
             fancybox=True, shadow=True, framealpha=0.9)
    
    # Tick improvements
    ax.tick_params(axis='both', which='major', labelsize=12, width=1.2)
    ax.tick_params(axis='both', which='minor', width=0.8)
    
    # Set reasonable y-axis limits for logit scale
    ax.set_ylim(0.001, 0.999)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight', 
                   facecolor='white', edgecolor='none')
    
    return fig, ax

def print_statistics(data, name="Pulses"):
    """
    Print comprehensive statistics about the data
    """
    print(f"\n{'='*50}")
    print(f"STATISTICAL SUMMARY FOR {name.upper()}")
    print(f"{'='*50}")
    
    # Basic statistics
    print(f"Total number of events: {len(data):,}")
    print(f"Mean: {np.mean(data):.2f}")
    print(f"Median: {np.median(data):.2f}")
    print(f"Standard deviation: {np.std(data):.2f}")
    print(f"Min: {np.min(data)}")
    print(f"Max: {np.max(data):,}")
    
    print(f"\n{'-'*30}")
    print("PERCENTILES")
    print(f"{'-'*30}")
    
    # Key percentiles
    percentiles = [1, 5, 10, 25, 50, 75, 90, 95, 99, 99.9]
    for p in percentiles:
        value = np.percentile(data, p)
        print(f"{p:5.1f}%: {value:8.1f}")
    
    print(f"\n{'-'*30}")
    print("THRESHOLD ANALYSIS")
    print(f"{'-'*30}")
    
    # Threshold analysis
    thresholds = [512, 1024, 2048]
    for threshold in thresholds:
        count = np.sum(data > threshold)
        fraction = count / len(data)
        print(f"Events > {threshold:4d}: {count:6,} ({fraction:6.2%})")

if __name__ == '__main__':
    print("Loading pulses from numpy array...")
    
    # Load the data
    data_path = '/storage/home/hcoda1/8/cfilho3/r-itaboada3-0/graphnet/carlos_tests/data_viz/pulses.npy'
    pulses = np.load(data_path)
    
    print("Pulses loaded successfully")
    
    # Print comprehensive statistics
    print_statistics(pulses, "Pulses per Event")
    
    # Create the CDF plot
    print("\nGenerating CDF plot...")
    
    save_path = '/storage/home/hcoda1/8/cfilho3/r-itaboada3-0/graphnet/carlos_tests/data_viz/plots/pulses_cdf_logit.png'
    
    fig, ax = plot_cdf_with_logit(
        pulses, 
        title="Cumulative Distribution of Pulses per Event",
        xlabel="Number of Pulses (log scale)",
        save_path=save_path
    )
    
    print(f"Plot saved to: {save_path}")
    plt.show()
    
    print("\nAnalysis complete!")