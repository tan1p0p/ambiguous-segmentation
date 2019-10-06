import datetime
import json
import os

from modules.trainer import Trainer
from utils.argument_handler import argment_handler

def test_main():
    config = json.load(open('config.json', 'r'))
    config["output_path"] += "{:%Y-%m-%d_%H:%M}/".format(datetime.datetime.now())
    config['is_file_saved'] = False

    t = Trainer(**config)
    t.optimize()
    assert 1 == 1
