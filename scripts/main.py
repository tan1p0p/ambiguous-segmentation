import datetime
import json
import os

from modules.trainer import Trainer

if __name__ == "__main__":
    config = json.load(open('config.json', 'r'))
    config["output_path"] += "{:%Y-%m-%d_%H:%M}/".format(datetime.datetime.now())

    os.mkdir(config["output_path"])
    json.dump(config, open(config["output_path"] + 'config.json', 'w'), indent=4)
    
    t = Trainer(**config)
    t.optimize()
