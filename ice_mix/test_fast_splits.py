"""Benchmark a historical SQLite event-splitting implementation."""

import sys
import sqlite3
import pandas as pd
import random
import time

def optimized_split(db_path, seed=42, split_ratio=[0.8, 0.1, 0.1]):
    """Split ``db_path`` event IDs using ``seed`` and ``split_ratio``.

        Returns deterministic ``(train, validation, test)`` integer lists and
        prints database-read/split timings. This is an author-confirmed abandoned
        exploratory benchmark, not the production splitter.
        """
    print(f"Reading {db_path}...")
    start = time.time()
    
    # Using pandas is usually much faster for huge SQLite tables and memory efficient
    with sqlite3.connect(db_path) as conn:
        df = pd.read_sql_query("SELECT event_no FROM truth", conn)
    
    events = df['event_no'].tolist()
    
    print(f"Found {len(events)} events in {time.time()-start:.2f}s")
    
    start = time.time()
    rnd = random.Random(seed)
    rnd.shuffle(events)
    
    n_events = len(events)
    n_train = int(n_events * split_ratio[0])
    n_val = int(n_events * split_ratio[1])
    
    train_events = events[:n_train]
    val_events = events[n_train:n_train+n_val]
    test_events = events[n_train+n_val:]
    
    print(f"Splits generated in {time.time()-start:.2f}s")
    return train_events, val_events, test_events

data_paths = ['/storage/home/hcoda1/4/jliao74/r-itaboada3-0/jliao74/my_numu_database_part_1 (1).db']
train, val, test = optimized_split(data_paths[0])
print(f'Train: {len(train)}, Val: {len(val)}, Test: {len(test)}')
