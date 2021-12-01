/*
 * Licensed to the Apache Software Foundation (ASF) under one
 * or more contributor license agreements.  See the NOTICE file
 * distributed with this work for additional information
 * regarding copyright ownership.  The ASF licenses this file
 * to you under the Apache License, Version 2.0 (the
 * "License"); you may not use this file except in compliance
 * with the License.  You may obtain a copy of the License at
 *
 *   http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing,
 * software distributed under the License is distributed on an
 * "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY
 * KIND, either express or implied.  See the License for the
 * specific language governing permissions and limitations
 * under the License.
 */

/*!
 * \file ethosn_runtime.cc
 * \brief Execution handling of Arm(R) Ethos(TM)-N command streams.
 */

#include "ethosn_runtime.h"

#include <dmlc/memory_io.h>
#include <tvm/runtime/c_runtime_api.h>
#include <tvm/runtime/memory.h>
#include <tvm/runtime/module.h>
#include <tvm/runtime/object.h>
#include <tvm/runtime/packed_func.h>
#include <tvm/runtime/registry.h>

#include <string>
#include <utility>
#include <vector>

#include "../../file_utils.h"
#include "ethosn_device.h"
#include "ethosn_driver_library/Inference.hpp"
#include "ethosn_driver_library/Network.hpp"

namespace tvm {
namespace runtime {
namespace ethosn {

namespace dl = ::ethosn::driver_library;

EthosnModule::EthosnModule(std::vector<OrderedCompiledNetwork>* cmms) {
  for (auto& it : *cmms) {
    network_map_[it.name].name = it.name;
    if (it.compiled_cmm != nullptr) {
      network_map_[it.name].compiled_cmm = std::move(it.compiled_cmm);
    }
#if _ETHOSN_API_VERSION_ > 2102
    if (it.runtime_cmm != nullptr) {
      network_map_[it.name].runtime_cmm = std::move(it.runtime_cmm);
    }
#endif
    network_map_[it.name].inputs = it.inputs;
    network_map_[it.name].outputs = it.outputs;
  }
}

PackedFunc EthosnModule::GetFunction(const std::string& name,
                                     const ObjectPtr<Object>& sptr_to_self) {
  if (network_map_.find(name) != network_map_.end()) {
    return PackedFunc([sptr_to_self, this, name](TVMArgs args, TVMRetValue* rv) {
#if _ETHOSN_API_VERSION_ <= 2102
      *rv = Inference(args, network_map_[name].compiled_cmm.get(), network_map_[name].inputs,
                      network_map_[name].outputs);
#else
      *rv = Inference(args, network_map_[name].runtime_cmm.get(), network_map_[name].inputs,
                      network_map_[name].outputs);
#endif
    });
  } else {
    return PackedFunc();
  }
}

void EthosnModule::SaveToBinary(dmlc::Stream* stream) {
  stream->Write(network_map_.size());
  for (const auto& it : network_map_) {
    stream->Write(it.first);
    std::stringstream ss;
    ICHECK(it.second.compiled_cmm != nullptr);
    it.second.compiled_cmm->Serialize(ss);
    stream->Write(ss.str());
    stream->Write(it.second.inputs.size());
    stream->Write(&it.second.inputs[0], sizeof(uint32_t) * it.second.inputs.size());
    stream->Write(it.second.outputs.size());
    stream->Write(&it.second.outputs[0], sizeof(uint32_t) * it.second.outputs.size());
  }
}

Module EthosnModule::LoadFromBinary(void* strm) {
  auto stream = static_cast<dmlc::Stream*>(strm);
  size_t func_count;
  // Read the number of functions
  stream->Read(&func_count);
  std::vector<OrderedCompiledNetwork> cmms;
  cmms.resize(func_count);
  for (unsigned int i = 0; i < func_count; i++) {
    OrderedCompiledNetwork& compiled = cmms[i];
    std::string ext_symbol;
    std::string cmm;
    uint64_t input_size;
    uint64_t output_size;
    // Read the symbol name
    stream->Read(&compiled.name);
    // Read the serialized command stream
    stream->Read(&cmm);
    std::istringstream cmm_strm(cmm);
#if _ETHOSN_API_VERSION_ <= 2102
    compiled.compiled_cmm = sl::DeserializeCompiledNetwork(cmm_strm);
#else
    compiled.runtime_cmm = std::make_unique<dl::Network>(cmm.c_str(), cmm.size());
#endif
    // Read the number of inputs
    stream->Read<uint64_t>(&input_size);
    auto size = static_cast<size_t>(input_size);
    compiled.inputs.resize(size);
    // Read the order of inputs
    stream->Read(&compiled.inputs[0], sizeof(uint32_t) * size);
    // Read the number of outputs
    stream->Read<uint64_t>(&output_size);
    size = static_cast<size_t>(output_size);
    compiled.outputs.resize(size);
    // Read the order of outputs
    stream->Read(&compiled.outputs[0], sizeof(uint32_t) * size);
  }
  auto n = make_object<EthosnModule>(&cmms);
  return Module(n);
}

void EthosnModule::SaveToFile(const std::string& path, const std::string& format) {
  std::string data;
  dmlc::MemoryStringStream writer(&data);
  dmlc::SeekStream* strm = &writer;
  SaveToBinary(strm);
  SaveBinaryToFile(path, data);
}

TVM_REGISTER_GLOBAL("runtime.module.loadbinary_ethos-n")
    .set_body([](TVMArgs args, TVMRetValue* rv) { *rv = EthosnModule::LoadFromBinary(args[0]); });
}  // namespace ethosn
}  // namespace runtime
}  // namespace tvm