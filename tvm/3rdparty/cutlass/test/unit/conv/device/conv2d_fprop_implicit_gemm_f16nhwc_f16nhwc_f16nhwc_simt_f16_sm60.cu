/***************************************************************************************************
 * Copyright (c) 2017-2021, NVIDIA CORPORATION.  All rights reserved.
 *
 * Redistribution and use in source and binary forms, with or without modification, are permitted
 * provided that the following conditions are met:
 *     * Redistributions of source code must retain the above copyright notice, this list of
 *       conditions and the following disclaimer.
 *     * Redistributions in binary form must reproduce the above copyright notice, this list of
 *       conditions and the following disclaimer in the documentation and/or other materials
 *       provided with the distribution.
 *     * Neither the name of the NVIDIA CORPORATION nor the names of its contributors may be used
 *       to endorse or promote products derived from this software without specific prior written
 *       permission.
 *
 * THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS" AND ANY EXPRESS OR
 * IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND
 * FITNESS FOR A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL NVIDIA CORPORATION BE LIABLE
 * FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING,
 * BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS;
 * OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT,
 * STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
 * OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
 *
 **************************************************************************************************/
/*! \file
    \brief Tests for device-wide Implicit GEMM interface
*/

#include "../../common/cutlass_unit_test.h"
#include "cutlass/cutlass.h"


#include "cutlass/conv/kernel/default_conv2d_fprop.h"
#include "cutlass/conv/device/implicit_gemm_convolution.h"

#include "conv2d_testbed.h"


////////////////////////////////////////////////////////////////////////////////
TEST(SM60_Device_Conv2d_Fprop_Analytic_ImplicitGemm_f16nhwc_f16nhwc_f16nhwc_simt_f16,
  128x128_8x2_64x64x8) {

  /// Conv operation element types for the Gemm equivalent (ImplicitGemm)
  using ElementA           = cutlass::half_t;
  using ElementB           = cutlass::half_t;
  using ElementC           = cutlass::half_t;
  using ElementAccumulator = cutlass::half_t;
  using ElementCompute     = cutlass::half_t;


  /// Device-level Conv2d instance
  using Conv2dFpropKernel = typename cutlass::conv::kernel::DefaultConv2dFprop<
    ElementA, 
    cutlass::layout::TensorNHWC,
    ElementB, 
    cutlass::layout::TensorNHWC,
    ElementC, 
    cutlass::layout::TensorNHWC,
    ElementAccumulator,
    cutlass::arch::OpClassSimt,
    cutlass::arch::Sm60,
    cutlass::gemm::GemmShape<128, 128, 8>,
    cutlass::gemm::GemmShape<64, 64, 8>, 
    cutlass::gemm::GemmShape<1, 1, 1>,
    cutlass::epilogue::thread::LinearCombination<
      ElementC,
      1,
      ElementAccumulator,
      ElementCompute
    >,
    cutlass::gemm::threadblock::GemmIdentityThreadblockSwizzle<>,
    2,
    cutlass::arch::OpMultiplyAdd,
    cutlass::conv::IteratorAlgorithm::kAnalytic
  >::Kernel;

  using Conv2dFprop = cutlass::conv::device::ImplicitGemmConvolution<Conv2dFpropKernel>;

  /// Run all unit test sizes with device-level Conv2d instance
  EXPECT_TRUE(test::conv::device::TestAllConv2d<Conv2dFprop>());

}

////////////////////////////////////////////////////////////////////////////////

TEST(SM60_Device_Conv2d_Fprop_Optimized_ImplicitGemm_f16nhwc_f16nhwc_f16nhwc_simt_f16,
  128x128_8x2_64x64x8) {

  /// Conv operation element types for the Gemm equivalent (ImplicitGemm)
  using ElementA           = cutlass::half_t;
  using ElementB           = cutlass::half_t;
  using ElementC           = cutlass::half_t;
  using ElementAccumulator = cutlass::half_t;
  using ElementCompute     = cutlass::half_t;


  /// Device-level Conv2d instance
  using Conv2dFpropKernel = typename cutlass::conv::kernel::DefaultConv2dFprop<
    ElementA, 
    cutlass::layout::TensorNHWC,
    ElementB, 
    cutlass::layout::TensorNHWC,
    ElementC, 
    cutlass::layout::TensorNHWC,
    ElementAccumulator,
    cutlass::arch::OpClassSimt,
    cutlass::arch::Sm60,
    cutlass::gemm::GemmShape<128, 128, 8>,
    cutlass::gemm::GemmShape<64, 64, 8>, 
    cutlass::gemm::GemmShape<1, 1, 1>,
    cutlass::epilogue::thread::LinearCombination<
      ElementC,
      1,
      ElementAccumulator,
      ElementCompute
    >,
    cutlass::gemm::threadblock::GemmIdentityThreadblockSwizzle<4>,
    2,
    cutlass::arch::OpMultiplyAdd,
    cutlass::conv::IteratorAlgorithm::kOptimized
  >::Kernel;

  using Conv2dFprop = cutlass::conv::device::ImplicitGemmConvolution<Conv2dFpropKernel>;

  /// Run all unit test sizes with device-level Conv2d instance
  EXPECT_TRUE(test::conv::device::TestAllConv2d<Conv2dFprop>());

}
