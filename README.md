# Exploring Light-Weight Object Recognition for Real-Time Document Detection

This repository contains the code for all of the experiments done in the paper "Exploring Light-Weight Object Recognition for Real-Time Document Detection". We outsource the IWPOD-Net implementation from its original repository from which this is a fork. It is available [here](https://github.com/claudiojung/iwpod-net). NBID, the dataset we used in our experiments, is available [here](https://github.com/BOVIFOCR/NBID-Dataset-Towards-Robust-Information-Extraction-in-Official-Documents).

## Training the model

```
python train_iwpodnet_tf2.py [-md RESULTS_DIR] [-n NAME] [-tr TRAINING_FILE] [-bs BATCH_SIZE] [-lr LEARNING_RATE] [-e EPOCHS] [-se SAVE_EPOCHS] [-p PATIENCE] [-v]
```

Our default parameters are 100.000 epochs, 6.000 patience, 0.001 learning rate, 6.000 save epochs and 64 batch size. The `-v` flag, if present, makes it so the model uses the validation set (which must be present).

## Testing the model

```
python test_iwpodnet_tf2.py [-m RESULTS_DIR] [-d DATASET_FILE] [-bs BATCH_SIZE] [-si]
```

For testing, we use a batch size of 1. The `-si` flag makes it so the images with predictions are saved (in the same directory as the source model, found in `-m`).

## Further reproduction

We also train and evaluate RTMDet and YOLO on NBID. The configuration we used for these models is available in `experiments/other_models`.

Likewise, our implementation of the OCR score and scripts for querying the LLM model used for OCR are available in `experiments/llm-ocr-analysis`.

## Single testing

The basic usage is

```
python example_plate_detection.py --image [image name] --vtype [vehicle type] --lp_threshold [detection threshold]
```

You can run a simple test based on the provided images.

```
python example_plate_detection.py --image images\example_aolp_fullimage.jpg --vtype fullimage
python example_plate_detection.py --image images\example_bike.jpg --vtype bike
```

