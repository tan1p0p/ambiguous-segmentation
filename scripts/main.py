import json
from modules.trainer import Trainer

if __name__ == "__main__":
    with open('config.json', 'r') as f:
        config = json.load(f)
    t = Trainer(**config)
    t.optimize()
