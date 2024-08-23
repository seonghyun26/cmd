import os
import pytz
import json
import argparse

from datetime import datetime

from openmm import *
from openmm.app import *
from openmm.unit import *

def load_config():
    # Parser
    parser = argparse.ArgumentParser(description="Simulation script")
    parser.add_argument("--config", type=str, help="Path to the config file", default="config/alanine/debug.json")
    args = parser.parse_args()
    
    # Config file
    config_file = args.config
    with open(config_file, "r") as f:
        config = json.load(f)
    
    return config