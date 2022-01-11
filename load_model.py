import os
import shutil

import tensorflow as tf
import tensorflow.keras as keras
from tensorflow.keras.applications import resnet50
from tensorflow.keras.applications import mobilenet
from tensorflow.keras.applications import inception_v3



models_detail = {
    'resnet50':resnet50.ResNet50(weights='imagenet'),
    'inception_v3':inception_v3.InceptionV3(weights='imagenet',include_top=False),
    'mobilenet':mobilenet.MobileNet(weights='imagenet')
}

import argparse

results = None
parser = argparse.ArgumentParser()
parser.add_argument('--model',default='resnet50' , type=str)
args = parser.parse_args()
load_model = args.model

saved_model_dir = f'{load_model}_saved_model'

def load_save_model(load_model,saved_model_dir):
    print(models_detail[load_model])
    model = models_detail[load_model]
    shutil.rmtree(saved_model_dir, ignore_errors=True)
    try:
        model.save(saved_model_dir, include_optimizer=False, save_format='tf')
        print(saved_model_dir," : complete load ")
    except:
        print("NOT saved")

if load_model : 
    load_save_model(load_model,saved_model_dir)
