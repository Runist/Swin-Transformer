# Swin-Transformer

![sw.png](https://s2.loli.net/2024/08/19/8kViRG6uxM4yFmP.png)

Network for Swin-Transformer. The pytorch version.

If this works for you, please give me a star, this is very important to me.😊

1. Clone this repository.

```shell
git clone https://github.com/Runist/Swin-Transformer
```

2. Install code from source.

```shell
cd Swin-Transformer
pip install -r requirements.txt
```

3. Download the **[flower dataset](https://github.com/Runist/Swin-Transformer/releases/download/dataset/flower_dataset.zip)**.
4. Download pretrain weights, the url in [model.py](https://github.com/Runist/Swin-Transformer/blob/master/model.py).
5. Start train your model.

```shell
python train.py --train-data-dir "train-data-path" --val-data-dir "val-data-path --device cuda:0
```

6. Get prediction of model.

```shell
python predict.py
```

7. Evaluate model.

```shell
python create_confusion_matrix.py --weights './weights/model-9.pth' --val-data-dir "val-data-path --device cuda:0
```

## Train your dataset

You need to store your data set like this:

```shell
├── train
│   ├── daisy
│   ├── dandelion
│   ├── roses
│   ├── sunflowers
│   └── tulips
└── validation
    ├── daisy
    ├── dandelion
    ├── roses
    ├── sunflowers
    └── tulips
```

## Reference

Appreciate the work from the following repositories:

- [microsoft](https://github.com/microsoft)/[Swin-Transformer](https://github.com/microsoft/Swin-Transformer)
- [WZMIAOMIAO](https://github.com/WZMIAOMIAO)/[swin-transformer](https://github.com/microsoft/Swin-Transformer)

## License

Code and datasets are released for non-commercial and research purposes **only**. For commercial purposes, please contact the authors.
