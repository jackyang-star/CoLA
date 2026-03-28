# ACGRL

This is the official PyTorch implementation for the [paper]():

> 

## Requirements

The project's requirements have already been defined in the `environment.yml` file; simply follow the configuration in that file.

## Quick Start

### Train and evaluate

Train the main model. 

For PW dataset:

```shell
python /main/pw/train_test_pw.py
```

For HAG dataset:

```
python /main/hw/train_test_pw.py
```

Training for model variants can be switched to the corresponding file.

### Customized Datasets

The HGA dataset refers to our previous work:[ https://github.com/528Lab/CAData](https://github.com/528Lab/CAData)

The PWA dataset refers to: https://github.com/kkfletch/API-Dataset