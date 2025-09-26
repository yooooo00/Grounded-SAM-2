# Copyright (c) OpenMMLab. All rights reserved.
# Copyright (c) 2025 Huawei Technologies Co., Ltd
# [Software Name] is licensed under Mulan PSL v2.
# You can use this software according to the terms and conditions of the Mulan PSL v2.
# You may obtain a copy of Mulan PSL v2 at:
#          http://license.coscl.org.cn/MulanPSL2
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND,
# EITHER EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT,
# MERCHANTABILITY OR FIT FOR A PARTICULAR PURPOSE.
# See the Mulan PSL v2 for more details.


from argparse import ArgumentParser
import ast
import os
import time
import torch
import torch_npu
import torchair as tng

from torchair.configs.compiler_config import CompilerConfig
from mmengine.logging import print_log
from mmdet.apis import DetInferencer
from mmdet.evaluation import get_classes
import demo.register_im2col_to_torchair
import demo.register_roll_to_torchair


def parse_args():
    parser = ArgumentParser()
    parser.add_argument(
        'inputs', type=str, help='Input image file or folder path.')
    parser.add_argument(
        'model',
        type=str,
        help='Config or checkpoint .pth file or the model name '
        'and alias defined in metafile. The model configuration '
        'file will try to read from .pth if the parameter is '
        'a .pth weights file.')
    parser.add_argument('--weights', default=None, help='Checkpoint file')
    parser.add_argument(
        '--out-dir',
        type=str,
        default='outputs',
        help='Output directory of images or prediction results.')
    # Once you input a format similar to $: xxx, it indicates that
    # the prompt is based on the dataset class name.
    # support $: coco, $: voc, $: cityscapes, $: lvis, $: imagenet_det.
    # detail to `mmdet/evaluation/functional/class_names.py`
    parser.add_argument(
        '--texts', help='text prompt, such as "bench . car .", "$: coco"')
    parser.add_argument(
        '--device', default='npu', help='Device used for inference')
    parser.add_argument(
        '--pred-score-thr',
        type=float,
        default=0.3,
        help='bbox score threshold')
    parser.add_argument(
        '--batch-size', type=int, default=1, help='Inference batch size.')
    parser.add_argument(
        '--show',
        action='store_true',
        help='Display the image in a popup window.')
    parser.add_argument(
        '--no-save-vis',
        action='store_true',
        help='Do not save detection vis results')
    parser.add_argument(
        '--no-save-pred',
        action='store_true',
        help='Do not save detection json results')
    parser.add_argument(
        '--print-result',
        action='store_true',
        help='Whether to print the results.')
    parser.add_argument(
        '--palette',
        default='none',
        choices=['coco', 'voc', 'citys', 'random', 'none'],
        help='Color palette used for visualization')
    # only for GLIP and Grounding DINO
    parser.add_argument(
        '--custom-entities',
        '-c',
        action='store_true',
        help='Whether to customize entity names? '
        'If so, the input text should be '
        '"cls_name1 . cls_name2 . cls_name3 ." format')
    parser.add_argument(
        '--chunked-size',
        '-s',
        type=int,
        default=-1,
        help='If the number of categories is very large, '
        'you can specify this parameter to truncate multiple predictions.')
    # only for Grounding DINO
    parser.add_argument(
        '--tokens-positive',
        '-p',
        type=str,
        help='Used to specify which locations in the input text are of '
        'interest to the user. -1 indicates that no area is of interest, '
        'None indicates ignoring this parameter. '
        'The two-dimensional array represents the start and end positions.')
    parser.add_argument('--warmup', type=int, default=1, help='Warm up times')
    parser.add_argument('--loop', type=int, default=10, help='Loop times')

    call_args = vars(parser.parse_args())

    if call_args['no_save_vis'] and call_args['no_save_pred']:
        call_args['out_dir'] = ''

    if call_args['model'].endswith('.pth'):
        print_log('The model is a weight file, automatically '
                  'assign the model to --weights')
        call_args['weights'] = call_args['model']
        call_args['model'] = None

    if call_args['texts'] is not None:
        if call_args['texts'].startswith('$:'):
            dataset_name = call_args['texts'][3:].strip()
            class_names = get_classes(dataset_name)
            call_args['texts'] = [tuple(class_names)]

    if call_args['tokens_positive'] is not None:
        call_args['tokens_positive'] = ast.literal_eval(
            call_args['tokens_positive'])

    init_kws = ['model', 'weights', 'device', 'palette']
    init_args = {}
    for init_kw in init_kws:
        init_args[init_kw] = call_args.pop(init_kw)

    return init_args, call_args


def compile_model(model):
    config = CompilerConfig()
    config.experimental_config.frozen_parameter = True
    config.experimental_config.tiling_schedule_optimize = True
    npu_backend = tng.get_npu_backend(compiler_config=config)
    for module in ['backbone', 'encoder', 'decoder', 'language_model']:
        setattr(model, module, torch.compile(getattr(model, module), dynamic=False, fullgraph=True, backend=npu_backend))
        tng.use_internal_format_weight(getattr(model, module))


def main():
    init_args, call_args = parse_args()

    inferencer = DetInferencer(**init_args)
    chunked_size = call_args.pop('chunked_size')
    inferencer.model.test_cfg.chunked_size = chunked_size
    warmup = call_args.pop('warmup')
    loop = call_args.pop('loop')

    # torchair compile
    compile_model(inferencer.model)

    #warmup
    for _ in range(warmup):
        with torch.no_grad():
            inferencer(**call_args)

    # infer
    st = time.time()
    for _ in range(loop):
        with torch.no_grad():
            inferencer(**call_args)
    ed = time.time()
    print_log(f"avg infer time: {(ed - st) * 1000 / loop}ms")

    if call_args['out_dir'] != '' and not (call_args['no_save_vis']
                                           and call_args['no_save_pred']):
        print_log(f'results have been saved at {call_args["out_dir"]}')


if __name__ == '__main__':
    main()
