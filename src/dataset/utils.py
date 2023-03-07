import json
import random
import argparse
import pandas as pd

from tqdm import tqdm
from typing import List
from pathlib import Path
from copy import deepcopy
from beartype import beartype

from src.labels import ground_truth

class setEncoder(json.JSONEncoder):
    def default(self, obj):
        return list(obj)

def get_max_ts(sessions_file: Path) -> int:
    max_ts = float("-inf")
    with open(sessions_file) as f:
        for line in tqdm(f, desc="Finding max timestamp"):
            session = json.loads(line)
            max_ts = max(max_ts, session["events"][-1]["ts"])
    return max_ts

def trim_session(session: dict, max_ts: int) -> dict:
    session["events"] = [event for event in session["events"] if event["ts"] < max_ts]
    return session

def train_test_split(session_chunks, train_file, test_file, max_ts, test_days, trim=True):
    assert (test_file is not None) or (train_file is not None), "Nothing to save"

    split_millis = test_days * 24 * 60 * 60 * 1000
    split_ts = max_ts - split_millis

    if train_file is not None:
        Path(train_file).parent.mkdir(parents=True, exist_ok=True)
        train_file = open(train_file, "w")
        print(f"- Saving train sessions to {train_file}")

    if test_file is not None:
        Path(test_file).parent.mkdir(parents=True, exist_ok=True)
        test_file = open(test_file, "w")
        print(f"- Saving test sessions to {test_file}")

    for chunk in tqdm(session_chunks, desc="Splitting sessions"):
        for _, session in chunk.iterrows():
            session = session.to_dict()
            if session["events"][0]["ts"] > split_ts:  # After split -> test
                if test_file is not None:
                    test_file.write(json.dumps(session, cls=setEncoder) + "\n")
            elif train_file is not None:  # Train
                if trim:
                    session = trim_session(session, split_ts)
                train_file.write(json.dumps(session, cls=setEncoder) + "\n")

    if train_file is not None:
        train_file.close()
    if test_file is not None:
        test_file.close()