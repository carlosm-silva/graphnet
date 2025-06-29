import numpy as np
import matplotlib.pyplot as plt
plt.style.use('seaborn-v0_8-darkgrid')


if __name__ == '__main__':
    # Load pulses from the numpy array
    print('Loading pulses from numpy array...')
    pulses = np.load('/storage/home/hcoda1/8/cfilho3/p-itaboada3-0/graphnet/carlos_tests/data_viz/pulses.npy')
    print('Pulses loaded successfully')
    # Print the max and min of the pulses
    print(f'Max pulses: {np.max(pulses)}')
    print(f'Min pulses: {np.min(pulses)}')
    # Print the 1% and 99% of the pulses
    print(f'1% of the pulses: {np.percentile(pulses, 1)}')
    print(f'99% of the pulses: {np.percentile(pulses, 99)}')
    # Print how many pulses are greater than 1024
    print(f'Number of pulses greater than 1024: {np.sum(pulses > 1024)}')
    # Print total number of events
    print(f'Total number of events: {pulses.shape[0]}')
    # Print the mean of the pulses
    print(f'Mean pulses: {np.mean(pulses)}')
    # Print the median of the pulses
    print(f'Median pulses: {np.median(pulses)}')
    # Print the standard deviation of the pulses
    print(f'Standard deviation of the pulses: {np.std(pulses)}')
    # Number and fraction of events with more than 512 pulses
    print(f'Number of events with more than 512 pulses: {np.sum(pulses > 512)}')
    print(f'Fraction of events with more than 512 pulses: {np.sum(pulses > 512) / pulses.shape[0]}')
    # get 99.9% percentile of the pulses
    perc_999 = np.percentile(pulses, 99.9)
    print(f'99.9% percentile of the pulses: {perc_999}')

    # Start plotting
    # Plot the histogram of the pulses with custom binning
    plt.figure(figsize=(15, 5), facecolor='#f7f7fa')
    min_pulse = np.min(pulses[pulses > 0])
    max_pulse = np.max(pulses)
    # First bin: 1 to 100, then log-spaced bins from 100 to max
    bins = np.concatenate((np.arange(1, 100, 1), np.logspace(np.log10(100), np.log10(max_pulse), 50)))
    n, bins, patches = plt.hist(pulses, bins=bins, color='royalblue', alpha=0.65, density=True, edgecolor='black', linewidth=0.5)
    plt.title('Number of Pulses per Event (Custom Log Binning)', fontsize=18, pad=15)
    plt.xlabel('Number of Pulses (log scale)', fontsize=15, labelpad=10)
    plt.xscale('log')
    plt.yscale('log')
    plt.ylabel('PDF', fontsize=15, labelpad=10)
    # Draw a vertical line at the 99.9% percentile
    perc_999 = np.percentile(pulses, 99.9)
    plt.axvline(x=perc_999, color='crimson', linestyle='--', linewidth=2, label='99.9% percentile')
    plt.text(perc_999*1.05, max(n)*0.7, '99.9%', color='crimson', fontsize=12, rotation=90, va='center', ha='left', backgroundcolor='#f7f7fa')
    # Draw a vertical line at 1024
    plt.axvline(x=1024, color='forestgreen', linestyle='--', linewidth=2, label='1024')
    plt.text(1024*1.05, max(n)*0.5, '1024', color='forestgreen', fontsize=12, rotation=90, va='center', ha='left', backgroundcolor='#f7f7fa')
    # Legend
    plt.legend(loc='upper left', fontsize=13, frameon=True, facecolor='white', edgecolor='gray')
    plt.grid(True, which="major", ls="--", lw=0.7, alpha=0.7)  # Only major gridlines
    # Add minor ticks (but not gridlines)
    plt.tick_params(axis='both', which='minor', length=5, width=1, direction='in', top=True, right=True)
    plt.minorticks_on()
    plt.tight_layout()
    plt.savefig('/storage/home/hcoda1/8/cfilho3/p-itaboada3-0/graphnet/carlos_tests/data_viz/plots/pulses_per_event_histogram2.png', dpi=300, bbox_inches='tight', facecolor='#f7f7fa')
    plt.show()
