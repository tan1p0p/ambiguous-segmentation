import datetime
import json
import os

from modules.trainer import Trainer
from utils.argument_handler import argment_handler

if __name__ == "__main__":
    args = argment_handler()

    config = json.load(open(args.config_file, 'r'))
    config["output_path"] += "{:%Y-%m-%d_%H:%M}/".format(datetime.datetime.now())
    config['is_file_saved'] = not args.no_write

    if config['is_file_saved']:
        os.mkdir(config["output_path"])
        json.dump(config, open(config["output_path"] + 'config.json', 'w'), indent=4)

    t = Trainer(**config)
    t.optimize()
