import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import numpy as np
import os
from matplotlib.gridspec import GridSpec

def load_and_process_data(log_file='nvidia_dmon.log'):
    # 1. Load the log, skipping comment lines and using whitespace as delimiter
    df = pd.read_csv(
        log_file,
        comment='#',
        sep=r'\s+',
        header=None,
        names=[
            'Date', 'Time', 'GPU', 'Power', 'GTemp', 'MTemp',
            'SM', 'Mem', 'Enc', 'Dec', 'JPG', 'OFA',
            'MCLK', 'PCLK', 'PViol', 'TViol',
            'FB', 'BAR1', 'CCPM', 'SBECC', 'DBECC',
            'PCI', 'Rx', 'Tx'
        ]
    )

    # Print available columns for debugging
    print(f"Available columns in log file: {df.columns.tolist()}")

    # 2. Parse a datetime index
    df['Timestamp'] = pd.to_datetime(df['Date'].astype(str) + ' ' + df['Time'])
    df.set_index('Timestamp', inplace=True)
    
    return df

def create_visualization(df, gpu_id=0, output_dir=None, resample_rate='1S'):
    # Filter for specific GPU
    gpu_data = df[df['GPU'] == gpu_id]
    
    # Resample and average only numeric columns
    gpu_num = gpu_data.select_dtypes(include='number')
    gpu_resampled = gpu_num.resample(resample_rate).mean()
    
    # Calculate time elapsed in minutes for more intuitive x-axis
    start_time = gpu_resampled.index[0]
    gpu_resampled['minutes'] = [(t - start_time).total_seconds() / 60 for t in gpu_resampled.index]
    
    # Create figure with GridSpec for better control of subplot layout
    plt.style.use('ggplot')
    fig = plt.figure(figsize=(12, 10))
    gs = GridSpec(3, 2, figure=fig, height_ratios=[1, 1, 0.8])
    
    # 1. Utilization Plot (SM and Memory)
    ax1 = fig.add_subplot(gs[0, :])
    ax1.plot(gpu_resampled['minutes'], gpu_resampled['SM'], 'b-', linewidth=2, label='SM Util (%)')
    ax1.plot(gpu_resampled['minutes'], gpu_resampled['Mem'], 'r-', linewidth=2, label='Mem Util (%)')
    ax1.set_ylabel('Utilization (%)')
    ax1.set_title(f'GPU {gpu_id} Utilization Over Time')
    ax1.legend(loc='upper right')
    ax1.grid(True, linestyle='--', alpha=0.7)
    
    # Add annotations for peak values
    sm_peak = gpu_resampled['SM'].max()
    sm_peak_idx = gpu_resampled['SM'].idxmax()
    sm_peak_minute = gpu_resampled.loc[sm_peak_idx, 'minutes']
    ax1.annotate(f'Peak: {sm_peak:.1f}%', 
                xy=(sm_peak_minute, sm_peak),
                xytext=(10, 20),
                textcoords='offset points',
                arrowprops=dict(arrowstyle='->', lw=1.5),
                bbox=dict(boxstyle='round,pad=0.3', fc='yellow', alpha=0.7))
    
    # 2. Temperature Plot - Check if MTemp exists
    ax2 = fig.add_subplot(gs[1, 0])
    ax2.plot(gpu_resampled['minutes'], gpu_resampled['GTemp'], 'g-', linewidth=2, label='GPU Temp (°C)')
    
    # Only plot MTemp if it exists in the dataframe
    if 'MTemp' in gpu_resampled.columns:
        ax2.plot(gpu_resampled['minutes'], gpu_resampled['MTemp'], 'y-', linewidth=2, label='Mem Temp (°C)')
    
    ax2.set_ylabel('Temperature (°C)')
    ax2.set_title(f'GPU {gpu_id} Temperature')
    ax2.legend(loc='upper right')
    ax2.grid(True, linestyle='--', alpha=0.7)
    
    # 3. Power Plot
    ax3 = fig.add_subplot(gs[1, 1])
    power_line = ax3.plot(gpu_resampled['minutes'], gpu_resampled['Power'], 'r-', linewidth=2, label='Power (W)')
    ax3.set_ylabel('Power (W)')
    ax3.set_title(f'GPU {gpu_id} Power Consumption')
    ax3.legend(loc='upper right')
    ax3.grid(True, linestyle='--', alpha=0.7)
    
    # Fill under the power curve
    ax3.fill_between(gpu_resampled['minutes'], 0, gpu_resampled['Power'], alpha=0.3, color='r')
    
    # 4. Clock Speeds
    ax4 = fig.add_subplot(gs[2, :])
    ax4.plot(gpu_resampled['minutes'], gpu_resampled['MCLK'], 'b-', linewidth=2, label='Memory Clock')
    ax4.plot(gpu_resampled['minutes'], gpu_resampled['PCLK'], 'g-', linewidth=2, label='Processor Clock')
    ax4.set_ylabel('Clock Speed')
    ax4.set_xlabel('Time Elapsed (minutes)')
    ax4.set_title(f'GPU {gpu_id} Clock Speeds')
    ax4.legend(loc='upper right')
    ax4.grid(True, linestyle='--', alpha=0.7)
    
    # Set common x-axis label and adjust layout
    for ax in [ax1, ax2, ax3]:
        ax.set_xlabel('Time Elapsed (minutes)')
    
    # Add some summary statistics in a text box
    stats_text = (
        f"Summary Statistics:\n"
        f"Avg SM Util: {gpu_resampled['SM'].mean():.1f}%\n"
        f"Max SM Util: {gpu_resampled['SM'].max():.1f}%\n"
        f"Avg Power: {gpu_resampled['Power'].mean():.1f}W\n"
        f"Max Power: {gpu_resampled['Power'].max():.1f}W\n"
        f"Max GPU Temp: {gpu_resampled['GTemp'].max():.1f}°C"
    )
    fig.text(0.01, 0.01, stats_text, fontsize=10,
             bbox=dict(facecolor='white', alpha=0.8, boxstyle='round,pad=0.5'))
    
    plt.tight_layout()
    fig.subplots_adjust(bottom=0.12)  # Make room for the stats text
    
    # Save to file if output directory is specified
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
        timestamp = pd.Timestamp.now().strftime('%Y%m%d_%H%M%S')
        filename = os.path.join(output_dir, f'gpu{gpu_id}_viz_{timestamp}.png')
        plt.savefig(filename, dpi=150, bbox_inches='tight')
        print(f"Visualization saved to {filename}")
    
    return fig

def create_comparative_plot(df, output_dir=None, window_minutes=10):
    """
    Create a comparative plot showing all GPUs together:
    1. SM usage over time for all GPUs
    2. Memory usage over time for all GPUs
    3. Memory Clock (MCLK) over time for all GPUs
    4. Processor Clock (PCLK) over time for all GPUs
    
    Args:
        df: The full dataframe with all GPU data
        output_dir: Directory to save the output image
        window_minutes: Size of the moving average window in minutes
    """
    # Get unique GPU IDs
    gpu_ids = sorted(df['GPU'].unique())
    
    # Create figure with four subplots, aiming for a 16:9 aspect ratio for the figure
    # For example, width=16, height=9.
    fig, (ax1, ax2, ax3, ax4) = plt.subplots(4, 1, figsize=(16, 9), sharex=True)
    
    # Color map for different GPUs
    colors = plt.cm.tab10(np.linspace(0, 1, len(gpu_ids)))
    # Define a list of line styles
    line_styles = ['-', '--', '-.', ':']
    
    # Calculate window size in data points
    # Assuming data is sampled at 1 second intervals
    window_size = window_minutes * 60
    
    # For each GPU, plot the SM, Memory, MCLK, and PCLK usage
    for i, gpu_id in enumerate(gpu_ids):
        # Filter data for this GPU
        gpu_data = df[df['GPU'] == gpu_id].copy()
        
        # Create a datetime index for consistent resampling
        gpu_data = gpu_data.sort_index()
        
        # Calculate time elapsed in minutes from the start of the data
        start_time = df.index[0]
        gpu_data['minutes'] = [(t - start_time).total_seconds() / 60 for t in gpu_data.index]
        
        # Apply moving average to smooth the data
        gpu_data['SM_smooth'] = gpu_data['SM'].rolling(window=window_size, min_periods=1).mean()
        gpu_data['Mem_smooth'] = gpu_data['Mem'].rolling(window=window_size, min_periods=1).mean()
        if 'MCLK' in gpu_data.columns:
            gpu_data['MCLK_smooth'] = gpu_data['MCLK'].rolling(window=window_size, min_periods=1).mean()
        if 'PCLK' in gpu_data.columns:
            gpu_data['PCLK_smooth'] = gpu_data['PCLK'].rolling(window=window_size, min_periods=1).mean()
        
        current_linestyle = line_styles[i % len(line_styles)] # Cycle through line styles
        current_alpha = 0.7 if len(gpu_ids) > 4 else 0.9 # Adjust alpha based on number of GPUs

        # Plot SM usage
        ax1.plot(gpu_data['minutes'], gpu_data['SM_smooth'], 
                 color=colors[i], linewidth=1.5, label=f'GPU {gpu_id}', 
                 linestyle=current_linestyle, alpha=current_alpha)
        
        # Plot Memory usage
        ax2.plot(gpu_data['minutes'], gpu_data['Mem_smooth'], 
                 color=colors[i], linewidth=1.5, label=f'GPU {gpu_id}',
                 linestyle=current_linestyle, alpha=current_alpha)

        # Plot Memory Clock (MCLK)
        if 'MCLK_smooth' in gpu_data.columns:
            ax3.plot(gpu_data['minutes'], gpu_data['MCLK_smooth'],
                     color=colors[i], linewidth=1.5, label=f'GPU {gpu_id}',
                     linestyle=current_linestyle, alpha=current_alpha)

        # Plot Processor Clock (PCLK)
        if 'PCLK_smooth' in gpu_data.columns:
            ax4.plot(gpu_data['minutes'], gpu_data['PCLK_smooth'],
                     color=colors[i], linewidth=1.5, label=f'GPU {gpu_id}',
                     linestyle=current_linestyle, alpha=current_alpha)
    
    # Configure SM usage subplot
    ax1.set_title('SM Utilization Across All GPUs (10-min Moving Average)', fontsize=14)
    ax1.set_ylabel('SM Utilization (%)', fontsize=12)
    ax1.grid(True, linestyle='--', alpha=0.7)
    ax1.legend(loc='upper right', ncol=min(4, len(gpu_ids)))
    
    # Configure Memory usage subplot
    ax2.set_title('Memory Utilization Across All GPUs (10-min Moving Average)', fontsize=14)
    ax2.set_ylabel('Memory Utilization (%)', fontsize=12)
    ax2.grid(True, linestyle='--', alpha=0.7)
    ax2.legend(loc='upper right', ncol=min(4, len(gpu_ids)))

    # Configure Memory Clock (MCLK) subplot
    ax3.set_title('Memory Clock (MCLK) Across All GPUs (10-min Moving Average)', fontsize=14)
    ax3.set_ylabel('MCLK (MHz)', fontsize=12) # Assuming MHz, adjust if different
    ax3.grid(True, linestyle='--', alpha=0.7)
    ax3.legend(loc='upper right', ncol=min(4, len(gpu_ids)))

    # Configure Processor Clock (PCLK) subplot
    ax4.set_title('Processor Clock (PCLK) Across All GPUs (10-min Moving Average)', fontsize=14)
    ax4.set_ylabel('PCLK (MHz)', fontsize=12) # Assuming MHz, adjust if different
    ax4.set_xlabel('Time Elapsed (minutes)', fontsize=12)
    ax4.grid(True, linestyle='--', alpha=0.7)
    ax4.legend(loc='upper right', ncol=min(4, len(gpu_ids)))
    
    plt.tight_layout()
    
    # Save to file if output directory is specified
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
        timestamp = pd.Timestamp.now().strftime('%Y%m%d_%H%M%S')
        filename = os.path.join(output_dir, f'comparative_gpu_viz_{timestamp}.png')
        plt.savefig(filename, dpi=150, bbox_inches='tight')
        print(f"Comparative visualization saved to {filename}")
    
    return fig

def check_required_columns(df):
    """Verify that required columns exist in the dataframe."""
    required_columns = ['GPU', 'Power', 'GTemp', 'SM', 'Mem', 'MCLK', 'PCLK']
    missing_columns = [col for col in required_columns if col not in df.columns]
    
    if missing_columns:
        print(f"Warning: The following required columns are missing: {missing_columns}")
        return False
    return True

if __name__ == "__main__":
    try:
        # Load data
        df = load_and_process_data()
        
        # Check if required columns exist
        if not check_required_columns(df):
            print("Some essential columns are missing. The visualization may be incomplete.")
        
        # Get unique GPU IDs
        gpu_ids = df['GPU'].unique()
        
        # Create visualization for each GPU
        for gpu_id in gpu_ids:
            fig = create_visualization(df, gpu_id=gpu_id, output_dir='plots')
            plt.show()
            
        # Create and show the comparative plot with all GPUs
        print("Creating comparative visualization of all GPUs...")
        comp_fig = create_comparative_plot(df, output_dir='plots')
        plt.show()
            
    except FileNotFoundError:
        print("Error: Log file 'nvidia_dmon.log' not found in the current directory.")
    except Exception as e:
        import traceback
        print(f"Error during visualization: {e}")
        print("Detailed error information:")
        traceback.print_exc()