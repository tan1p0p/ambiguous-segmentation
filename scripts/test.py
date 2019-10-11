import datetime
import json
import os

from modules.trainer import Trainer
from utils.argument_handler import argment_handler

def test_main():
    config = json.load(open('config.json', 'r'))
    config["output_path"] += "{:%Y-%m-%d_%H:%M}/".format(datetime.datetime.now())
    config['is_file_saved'] = False
    config['portrait_dir'] = "./data/person_image_dataset/96x64_one/"
    config['batch_size'] = 1
    config['train_data_num'] = 1
    config['test_data_num'] = 1

    t = Trainer(**config)
    t.optimize()
    assert 1 == 1
