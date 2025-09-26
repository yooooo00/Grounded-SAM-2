# Copyright (c) 2025 Huawei Technologies Co., Ltd
# [Software Name] is licensed under Mulan PSL v2.
# You can use this software according to the terms and conditions of the Mulan PSL v2.
# You may obtain a copy of Mulan PSL v2 at:
#          http://license.coscl.org.cn/MulanPSL2
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND,
# EITHER EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT,
# MERCHANTABILITY OR FIT FOR A PARTICULAR PURPOSE.
# See the Mulan PSL v2 for more details.


from typing import (
    Any,
    Callable,
    ContextManager,
    Iterable,
    List,
    Literal,
    NamedTuple,
    Optional,
    Sequence,
    Tuple,
    TypeVar,
    Union,
    overload,
)

import torch
from torch import Generator, contiguous_format, inf, strided, SymInt
from torch.types import Device, Number, _bool, _complex, _device, _dtype, _float, _int, _layout, _qscheme, _size
from torchair._ge_concrete_graph import ge_apis as ge
from torchair._ge_concrete_graph.fx2ge_converter import register_fx_node_ge_converter, declare_supported
from torchair._ge_concrete_graph.supported_declaration import _TypedTensor, F32, F16, F64, I32, I16, I64, I8, U8, BOOL, \
    Support
from torchair.ge._ge_graph import Tensor, TensorSpec
from torchair._ge_concrete_graph.utils import specific_op_input_layout, specific_op_output_layout


@declare_supported([
    Support(F32(1, 128, 200, 300), kernel_size=[2, 2], dilation=[1, 1], padding=[0, 0], stride=[2, 2])
])
@register_fx_node_ge_converter(torch.ops.aten.im2col.default)
def conveter_aten_im2col_default(
    self: Tensor,
    kernel_size: List[int],
    dilation: List[int],
    padding: List[int],
    stride: List[int],
    meta_outputs: TensorSpec = None,
):
    """NB: aten::im2col(Tensor self, int[2] kernel_size, int[2] dilation, int[2] padding, int[2], stride) -> Tensor"""
    if len(kernel_size) == 1:
        kernel_size = [kernel_size[0], kernel_size[0]]
    elif len(kernel_size) != 2:
        raise AssertionError(f"torch.ops.aten.im2col.default kernel_size must be in size(1, 2), but got {len(kernel_size)}")
    if len(padding) == 1:
        pads = [padding[0], padding[0], padding[0], padding[0]]
    elif len(padding) == 2:
        pads = [padding[0], padding[0], padding[1], padding[1]]
    else:
        raise AssertionError(f"torch.ops.aten.im2col.default padding must be in size(1, 2), but got {len(padding)}")
    if len(stride) == 1:
        stride = [stride[0], stride[0]]
    elif len(stride) != 2:
        raise AssertionError(f"torch.ops.aten.im2col.default stride must be in size(1, 2), but got {len(stride)}")
    if len(dilation) == 1:
        dilation = [dilation[0], dilation[0]]
    elif len(dilation) != 2:
        raise AssertionError(f"torch.ops.aten.im2col.default dilation must be in size(1, 2), but got {len(dilation)}")
    
    im2col_output = ge.Im2col(self, ksizes=kernel_size, strides=stride, dilations=dilation, pads=pads)
    specific_op_input_layout(im2col_output, indices=0, layout="NCHW")
    specific_op_output_layout(im2col_output, indices=0, layout="NCHW")
    reshape_output = ge.FlattenV2(im2col_output, axis=2, end_axis=3)
    return reshape_output