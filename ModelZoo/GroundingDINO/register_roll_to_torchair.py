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
from torchair._ge_concrete_graph.fx2ge_converter import register_fx_node_ge_converter
from torchair.ge._ge_graph import Tensor, TensorSpec, DataType
from torchair._ge_concrete_graph.utils import dtype_promote, specific_op_input_layout, specific_op_output_layout


@register_fx_node_ge_converter(torch.ops.aten.roll.default)
def conveter_aten_roll_default(
    self: Tensor,
    shifts: Union[List[int], Tensor],
    dims: List[int],
    meta_outputs: TensorSpec = None,
):
    """NB: aten::roll(Tensor self, SymInt[1] shifts, int[1] dims=[]) -> Tensor"""
    first_dim = 0
    input_dim_len = self.rank
    dims_len = len(dims)
    perm = []
    for idx in range(input_dim_len):
        perm.append(idx)
    roll_output = self
    for i in range(dims_len):
        axis = dims[i]
        if i == 0 and axis == first_dim:
            roll_output = ge.Roll(roll_output, shifts=[shifts[i]], dims=[first_dim])
            specific_op_input_layout(roll_output, indices=0, layout="ND")
            specific_op_output_layout(roll_output, indices=0, layout="ND")
        else:
            # roll算子计算逻辑：把需要处理的维度放到第0维
            perm[axis], perm[first_dim] = perm[first_dim], perm[axis]
            trans_perm = dtype_promote(perm, target_dtype=DataType.DT_INT64)
            roll_output = ge.Transpose(roll_output, perm=trans_perm)
            roll_output = ge.Roll(roll_output, shifts=[shifts[i]], dims=[first_dim])
            specific_op_input_layout(roll_output, indices=0, layout="ND")
            specific_op_output_layout(roll_output, indices=0, layout="ND")
            roll_output = ge.Transpose(roll_output, perm=trans_perm)
            perm[axis], perm[first_dim] = perm[first_dim], perm[axis]
    
    return roll_output