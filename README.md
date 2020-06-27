# ambiguous segmentation

## About
Code for ambiguous segmentation experiment.

## Setup
- Copy `config.json.example` to `config.json`.
- Download `https://github.com/huochaitiantang/pytorch-deep-image-matting/releases/download/v1.3/stage1_sad_57.1.pth` to `models/` and write path to `config.json`.
- Download datasets from `https://www.kaggle.com/laurentmih/aisegmentcom-matting-human-datasets`, `https://cocodataset.org/#download` and write path to `config.json`.

## Run
```bash
make build
make run
```

``` bash
# in docker workspace
make train
```

To see evaluation results, please use `notebooks/evaluation.ipynb`.
