"""Run the historical dynamic-split utility on an author-specific database."""

import sys
sys.path.append('.')
from src.utils import get_dynamic_splits
import time

def main():
    """Run the abandoned split probe against its historical PACE path."""
    data_paths = ['/storage/home/hcoda1/4/jliao74/r-itaboada3-0/jliao74/my_numu_database_part_1 (1).db']
    start = time.time()
    train, val, test = get_dynamic_splits(data_paths, 42, [0.8, 0.1, 0.1])
    print(f'Train: {len(train[0])}, Val: {len(val[0])}, Test: {len(test[0])}')
    print(f'Time taken: {time.time()-start:.2f}s')


if __name__ == "__main__":
    main()
