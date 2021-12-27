import numpy as np
import pickle
import time
import json
import boto3
import os
from os.path import join, dirname

import tvm
from tvm import relay
from tvm.relay import testing
from tvm.contrib import graph_runtime
from tvm.contrib import graph_executor
from tvm.contrib import utils


bucket_name = 'subin-tvm'
folder_name = 'compiled_model/target-cpu'

s3 = boto3.client('s3')
ctx = tvm.cpu()

def load_model(model,batch_size,arch):
    filename = f'./model/{model}_{batch_size}_{arch}_deploy_lib.tar'
    key = f'{folder_name}/{model}_deploy_lib.tar'

    s3.download_file(bucket_name,key,filename)
    # 다운받은 모델 호출 => 경로 수정 필요해보임 
    loaded_lib = tvm.runtime.load_module(f'./model/{model}_{batch_size}_{arch}_deploy_lib.tar')

    module = graph_executor.GraphModule(loaded_lib["default"](ctx))
    return module

def get_tvm_model(name, batch_size,image_shape,dtype="float32"):
    output_shape = (batch_size, 1000)
    mod=None

    if name.startswith("resnet"):
        n_layer = int(name.split("t")[1])
        mod, params = relay.testing.resnet.get_workload(
            num_layers=n_layer,
            batch_size=batch_size,
            dtype=dtype,
            image_shape=image_shape,
        )
    elif name == "mobilenet":
        mod, params = relay.testing.mobilenet.get_workload(
            batch_size=batch_size, dtype=dtype, image_shape=image_shape)
    elif name == "inception_v3":
        # input_shape = (batch_size, 3, 299, 299) if layout == "NCHW" else (batch_size, 299, 299, 3)
        mod, params = relay.testing.inception_v3.get_workload(batch_size=batch_size, dtype=dtype)

    return mod, params

def build_model(model,batch_size,image_shape):
    target = 'llvm'

    mod,params = get_tvm_model(model,batch_size,image_shape)

    # build 
    opt_level = 3
    with relay.build_config(opt_level=opt_level):
        graph, lib, params = relay.build_module.build(
            mod, target, params=params)

    # graph create 
    module = graph_runtime.create(graph, lib, ctx)

    #export model 
    lib.export_library(f"./upload_model/{model}_{batch_size}_{target}_deploy_lib.tar")

    # upload to s3 
    filename = f"./upload_model/{model}_{batch_size}_{target}_deploy_lib.tar"
    key = f'{folder_name}/{model}_{batch_size}_{target}_deploy_lib.tar'
    res = s3.upload_file(filename,bucket_name,key)
    
    return module

def make_random_data(batch_size,size):
    image_shape = (3, size, size)
    data_shape = (batch_size,) + image_shape

    data = np.random.uniform(-1, 1, size=data_shape).astype("float32")

    return data,image_shape

def inference(module,input_data):
    pred_start = time.time()
    module.run(data=input_data)
    pred_time = time.time() - pred_start
    
    out_deploy = module.get_output(0).numpy()

    data_tvm = tvm.nd.array(input_data.astype('float32'))
    e = module.module.time_evaluator("run", ctx, number=10, repeat=1)
    t = e(data_tvm).results
    t = np.array(t) * 1000
    print('inference time : {} ms'.format( t.mean()))

    return out_deploy,pred_time


def lambda_handler(event, context):
    model = event['model']
    batch_size = event['batch_size']
    arch = event['arch']

    if model == 'inception_v3':
        input_data,image_shape = make_random_data(batch_size,299)
    else:
        input_data,image_shape = make_random_data(batch_size,224)

    total_start = time.time()
    #1. s3에서 model load 하는 경우, 즉 build 는 다른 인스턴스에서 진행하고 추론만 하고 싶을 때
    #module = load_model(model,batch_size,arch)

    #2. lambda에서 build와 inference 모두 다 진행하는 경우 
    module = build_model(model,batch_size,image_shape)

    result, pred_time = inference(module,input_data)
    total_time = time.time() - total_start

    return {
        'model_name': model,
        'batch_size': batch_size,
        'arch_info': arch,
        'total_time': total_time,
        'pred_time': pred_time,
    }

  

if __name__ =="__main__":
  context =""
  event= {
  "model": "mobilenet",
  "batch_size": 16,
  "arch": "llvm"
  }
  print(lambda_handler(event,context))
