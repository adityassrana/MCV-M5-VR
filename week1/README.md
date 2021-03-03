# Week1 - Image Classification Using PyTorch

The goal of this week is to adapt the Keras model we trained for M3 project to PyTorch. 

Training Script Usage

````
$ python train.py -h
usage: train.py [-h] [--exp_name EXP_NAME] [--data_path DATA_PATH] [--max_epochs MAX_EPOCHS] [--lr LR] [--image_size IMAGE_SIZE] [--batch_size BATCH_SIZE] [--num_workers NUM_WORKERS] [--save_model]
                [--tb]

A simple script for training an image classifier

optional arguments:
  -h, --help            show this help message and exit
  --exp_name EXP_NAME   name of experiment
  --data_path DATA_PATH
                        path to MITSplit Dataset
  --max_epochs MAX_EPOCHS
                        number of epochs to train our models for
  --lr LR               base learning rate
  --image_size IMAGE_SIZE
                        image size for training
  --batch_size BATCH_SIZE
                        batch size for training
  --num_workers NUM_WORKERS
                        number of workers for loading data
  --save_model          to save the model at the end of each epoch
  --tb                  to write to tensorboard
````