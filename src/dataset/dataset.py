import os
import json
import random
import pandas as pd
from tqdm import tqdm

from src.utils import get_max_ts, train_test_split

def train_val_split(train_data, output_path, val_days = 7, trim=True ):
    train_file = output_path/"sessions.json"
    val_file = output_path/"val_sessions.json"

    max_ts = get_max_ts(train_file)
    print(f"Using {days} before {max_ts} as validation set")

    session_chunks = pd.read_json(train_file, lines=True, chunksize=100000)

    if train_file.exists():
        os.remove(train_file)
    if val_file.exists():
        os.remove(val_file)
    
    train_test_split(session_chunks, train_file, val_file, max_ts, val_days, trim=trim)
    