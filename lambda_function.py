import time
import_start_time = time.time()
from tvm import relay
import numpy as np 
import tvm 
from tvm.contrib import graph_executor
from tvm.contrib import graph_runtime
import onnx
import boto3
print('import time: ', time.time() - import_start_time)

def get_model(model_name, bucket_name, get_path):
    s3_client = boto3.client('s3')
    if get_path == 'tvm/':
        print(get_path + model_name +'/model.tar')
        s3_client.download_file(bucket_name, get_path + model_name +'/model.tar', '/tmp/model.tar')
        s3_client.download_file(bucket_name, get_path + model_name +'/model.params', '/tmp/model.params')
        s3_client.download_file(bucket_name, get_path + model_name +'/model.json', '/tmp/model.json')
        return '/tmp/model.json', '/tmp/model.tar', '/tmp/model.params'
    else:
        s3_client.download_file(bucket_name, get_path + model_name, '/tmp/'+ model_name)
        return '/tmp/' + model_name
def make_dataset(batch_size,size):
    image_shape = (size, size, 3)
    data_shape = (batch_size,) + image_shape

    data = np.random.uniform(-1, 1, size=data_shape).astype("float32")

    return data,image_shape

load_model = time.time()


def lambda_handler(event, context):
    bucket_name = event['bucket_name']
    batch_size = event['batch_size']
    onnx_name = event['model_name'] + '.onnx'
    arch_type = event['arch_type']
    model_name = event['model_name'] +'_'+str(batch_size)+'_' + arch_type
    is_build = event['is_build']
    count = event['count']
    size = 224
    
    if arch_type == 'arm':
        target = tvm.target.arm_cpu()
    else:
        target = arch_type
    ctx = tvm.cpu()
    
    data, image_shape = make_dataset(batch_size,size)
    
    if is_build == 'true':
        print("ONNX model imported to relay frontend.")
        
        onnx_model = onnx.load(get_model(onnx_name, bucket_name, 'onnx/'))
        shape_dict = {"input_1": data.shape}
        mod, params = relay.frontend.from_onnx(onnx_model, shape=shape_dict)
        build_time = time.time()
        with tvm.transform.PassContext(opt_level=3):
            mod = relay.transform.InferType()(mod)
            graph, lib, params = relay.build_module.build(mod, target=target, params=params)
        print('build time:', time.time() - build_time)
        module = graph_runtime.create(graph, lib, ctx)
        module.set_input("input_1", data)
        module.set_input(**params)
    else:
        graph_fn, mod_fn, params_fn = get_model(model_name, bucket_name, 'tvm/')
        graph = open(graph_fn).read()
        lib = tvm.runtime.load_module(mod_fn)
        params = open(params_fn, "rb").read()
        module = graph_runtime.create(graph, lib, ctx)
        module.set_input("input_1", data)
        module.load_params(params)

    time_list = []
    for i in range(count):
        start_time = time.time()
        module.run(data=data)
        running_time = time.time() - start_time
        print(f"VM {model_name}-{batch_size} inference latency : ",(running_time)*1000,"ms")
        time_list.append(running_time)
    time_medium = np.median(np.array(time_list))
    return time_medium
