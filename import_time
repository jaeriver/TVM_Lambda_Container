import os
import numpy as np
import time

start_time = time.time()

from tvm import relay
from tvm.relay import testing
import tvm

print("import time", time.time() - start_time)

import pickle
from tvm.contrib import utils
from tvm.contrib import graph_runtime
from tvm.relay.testing import resnet
from tvm.relay.testing import mobilenet
from tvm.relay.testing import inception_v3

print("import model time", time.time() - start_time)

target = 'llvm'
ctx = tvm.cpu()

print("wall time", time.time() - start_time)
