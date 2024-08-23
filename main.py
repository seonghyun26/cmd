import os
import torch
import logging
import datetime
import argparse

from src import *

import numpy as np
import matplotlib.pyplot as plt

if __name__ == "__main__":
    # Load configs
    print(f"Loading configs...")
    config = load_config()
    log_dir = set_logging(config)
    
    
    