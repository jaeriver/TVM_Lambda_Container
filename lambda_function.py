import time
import itertools
import tensorflow as tf
import_start_time = time.time()
from tvm import relay
import numpy as np 
import tvm 
from tvm.contrib import graph_executor
import tvm.relay.testing.tf as tf_testing

import tvm.testing 
from tvm.runtime.vm import VirtualMachine
print('import time: ', time.time() - import_start_time)

model_name = 'resnet50'
arch_type = 'intel'


def make_dataset(batch_size,size):
    image_shape = (size, size, 3)
    data_shape = (batch_size,) + image_shape

    data = np.random.uniform(-1, 1, size=data_shape).astype("float32")

    return data,image_shape

def wrap_frozen_graph(graph_def, inputs, outputs, print_graph=False):
    def _imports_graph_def():
        tf.compat.v1.import_graph_def(graph_def, name="")

    wrapped_import = tf.compat.v1.wrap_function(_imports_graph_def, [])
    import_graph = wrapped_import.graph
    
    return wrapped_import.prune(
        tf.nest.map_structure(import_graph.as_graph_element, inputs),
        tf.nest.map_structure(import_graph.as_graph_element, outputs))

load_model = time.time()
with tf.io.gfile.GFile(f"./frozen_models/frozen_{model_name}.pb", "rb") as f:
        graph_def = tf.compat.v1.GraphDef()
        graph_def.ParseFromString(f.read())
        graph = tf.import_graph_def(graph_def, name="")
        # Call the utility to import the graph definition into default graph.
        graph_def = tf_testing.ProcessGraphDefParam(graph_def)
        try : 
            with tf.compat.v1.Session() as sess:
                graph_def = tf_testing.AddShapesToGraphDef(sess, "softmax")
        except:
            pass

frozen_func = wrap_frozen_graph(graph_def=graph_def,
                                    inputs=["input_1:0"],
                                    outputs=["Identity:0"],
                                    print_graph=True)




def lambda_handler(event, context):
    count = event['count']
    batch_size = event['batch_size']
    size=224
    data, image_shape = make_dataset(batch_size,size)
    
    # test dataset 생성 
    data, image_shape = make_dataset(batch_size,size)

    print(data.shape)
    shape_dict = {"DecodeJpeg/contents": data.shape}

    ##### Convert tensorflow model 
    mod, params = relay.frontend.from_tensorflow(graph_def, layout=None, shape=shape_dict)
    print("Tensorflow protobuf imported to relay frontend.")
    print("-"*10,"Load frozen model",time.time()-load_model,"s","-"*10)

    if arch_type == "intel":
        target = "llvm"
    else:
        target = tvm.target.arm_cpu()

    ctx = tvm.cpu()

    print("-"*10,"Compile style : create_executor vm ","-"*10)
    build_time = time.time()
    with tvm.transform.PassContext(opt_level=3):
        # executor = relay.build_module.create_executor("vm", mod, tvm.cpu(0), target)
        mod = relay.transform.InferType()(mod)
        executor = relay.vm.compile(mod, target=target, params=params)

    print("-"*10,"Build latency : ",time.time()-build_time,"s","-"*10)

    # executor.evaluate()(data,**params)
    vm = VirtualMachine(executor,ctx)
    _out = vm.invoke("main",data)

    input_data = tvm.nd.array(data)
    
    time_list = []
    for i in range(count):
        start_time = time.time()
        vm.run(input_data)
        running_time = time.time() - start_time
        print(f"VM {model_name}-{batch_size} inference latency : ",(running_time)*1000,"ms")
        time_list.append(running_time)
    time_medium = np.array(time_list).medium()
    return time_medium
