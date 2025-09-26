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


import argparse
import time
import warnings
import cv2
import mmcv
import numpy as np
import torch
import torch_npu
import torchair as tng

from tqdm import tqdm
from torchair.configs.compiler_config import CompilerConfig
from mmcv.transforms import Compose
from mmdet.apis import init_detector
from mmdet.registry import VISUALIZERS
from mmdet.models.language_models.bert import generate_masks_with_special_tokens_and_transfer_map
import demo.register_im2col_to_torchair
import demo.register_roll_to_torchair


def parse_args():
    parser = argparse.ArgumentParser(description='MMDetection video demo')
    parser.add_argument('video', help='Video file')
    parser.add_argument('config', help='Config file')
    parser.add_argument('checkpoint', help='Checkpoint file')
    parser.add_argument('--batch_size', type=int, default=1, help='Batch size')
    parser.add_argument('--warmup', type=int, default=1, help='Warm up times')
    parser.add_argument('--loop', type=int, default=1, help='Loop times')
    parser.add_argument(
        '--device', default='npu', help='Device used for inference')
    parser.add_argument(
        '--score-thr', type=float, default=0.3, help='Bbox score threshold')
    parser.add_argument('--out', type=str, help='Output video file')
    parser.add_argument('--show', action='store_true', help='Show video')
    parser.add_argument(
        '--wait-time',
        type=float,
        default=1,
        help='The interval of show (s), 0 is block')
    args = parser.parse_args()
    return args


def compile_model(model):
    config = CompilerConfig()
    config.experimental_config.frozen_parameter = True
    config.experimental_config.tiling_schedule_optimize = True
    npu_backend = tng.get_npu_backend(compiler_config=config)
    for module in ['backbone', 'encoder', 'decoder', 'language_model']:
        setattr(model, module, torch.compile(getattr(model, module), dynamic=False, fullgraph=True, backend=npu_backend))
        tng.use_internal_format_weight(getattr(model, module))


# The content of this function is copied from mmdetection/demo/video_demo.py
def init_video_tools(args, model):
    # init visualizer
    visualizer = VISUALIZERS.build(model.cfg.visualizer)
    # the dataset_meta is loaded from the checkpoint and
    # then pass to the model in init_detector
    visualizer.dataset_meta = model.dataset_meta

    video_reader = mmcv.VideoReader(args.video)
    video_writer = None
    if args.out:
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        video_writer = cv2.VideoWriter(
            args.out, fourcc, video_reader.fps,
            (video_reader.width, video_reader.height))
    
    video_tools = {
        "visualizer": visualizer,
        "video_reader": video_reader,
        "video_writer": video_writer
    }
    
    return video_tools


def preprocess(model, batch_frame, text_prompt):
    # build test pipeline
    model.cfg.test_dataloader.dataset.pipeline[
        0].type = 'mmdet.LoadImageFromNDArray'
    test_pipeline = Compose(model.cfg.test_dataloader.dataset.pipeline)
    if model.data_preprocessor.device.type == 'cpu':
        for m in model.modules():
            if isinstance(m, RoIPool):
                raise AssertionError('CPU inference with RoIPool is not supported currently.')

    for i, img in enumerate(batch_frame):
        # prepare data
        if isinstance(img, np.ndarray):
            data_ = dict(img=img, img_id=0)
        else:
            data_ = dict(img_path=img, img_id=0)

        if text_prompt:
            data_['text'] = text_prompt
            data_['custom_entities'] = False

        # build the data pipeline
        data_ = test_pipeline(data_)
        if i == 0:
            batch_data = data_
            batch_data['inputs'] = [data_['inputs']]
            batch_data['data_samples'] = [data_['data_samples']]
        else:
            batch_data['inputs'].append(data_['inputs'])
            batch_data['data_samples'].append(data_['data_samples'])

    return batch_data


def infer(model, batch_data, tokenized, attention_mask, position_ids):
    # forward the model
    with torch.no_grad():
        batch_data = model.data_preprocessor(batch_data, False)
        batch_data_samples, results_list = model.predict(batch_data['inputs'], batch_data['data_samples'],
                                                         tokenized, attention_mask, position_ids, isvisualize=False)
    return batch_data_samples, results_list


# The content of this function is copied from 
# mmdetection/demo/video_demo.py and mmdetection/mmdet/models/detectors/grounding_dino.py:predict
def postprocess(video_tools, data_samples, results, text_prompt, args):
    visualizer = video_tools.get("visualizer")
    video_reader = video_tools.get("video_reader")
    video_writer = video_tools.get("video_writer")
    for data_sample, pred_instances in zip(
            data_samples, results):
        if len(pred_instances) > 0:
            label_names = []
            for labels in pred_instances.labels:
                if labels >= len(text_prompt):
                    warnings.warn(
                        'The unexpected output indicates an issue with '
                        'named text_prompt recognition. You can try '
                        'setting custom_entities=True and running '
                        'again to see if it helps.')
                    label_names.append('unobject')
                else:
                    label_names.append(text_prompt[labels])
            # for visualization
            pred_instances.label_names = label_names
        data_sample.pred_instances = pred_instances

    for frame, result in zip(video_reader, data_samples):
        visualizer.add_datasample(
            name='video',
            image=frame,
            data_sample=result,
            draw_gt=False,
            show=False,
            pred_score_thr=args.score_thr)
        frame = visualizer.get_image()

        if args.show:
            cv2.namedWindow('video', 0)
            mmcv.imshow(frame, 'video', args.wait_time)
        if args.out:
            video_writer.write(frame)

    if video_writer:
        video_writer.release()
    cv2.destroyAllWindows()


def tokenize(model, text_prompts, device):
    tokenized = model.language_model.tokenizer.batch_encode_plus(
        list(text_prompts),
        max_length=model.language_model.max_tokens,
        padding='max_length' if model.language_model.pad_to_max else 'longest',
        return_special_tokens_mask=True,
        return_tensors='pt',
        truncation=True
    ).to(device)
    attention_mask, position_ids = \
        generate_masks_with_special_tokens_and_transfer_map(
            tokenized, model.language_model.special_tokens)
    return tokenized, attention_mask, position_ids


def main():
    args = parse_args()
    device = args.device
    batch_size = args.batch_size

    # build the model from a config file and a checkpoint file
    model = init_detector(args.config, args.checkpoint, device=device)
    model.half()

    # torchair compile
    compile_model(model)

    # init video tools
    video_tools = init_video_tools(args, model)
    video_reader = video_tools.get("video_reader")
    if len(video_reader) < batch_size:
        raise AssertionError(f"video frame len cannot be less than batch_size, "
                             f"now frame len: {len(video_reader)}, batch_size: {batch_size}")

    # tokenizer
    text_prompt = ["person", "car"]
    text_prompts = ['person. car. '] * batch_size
    tokenized, attention_mask, position_ids = tokenize(model, text_prompts, device)

    # warmup
    for _ in range(args.warmup):
        batch_frame = video_reader[0:batch_size]
        batch_data = preprocess(model, batch_frame, text_prompt)
        _, _ = infer(model, batch_data, tokenized, attention_mask, position_ids)

    # infer the model
    results = []
    data_samples = []
    infertime = 0
    total = int(np.ceil(float(len(video_reader)) / batch_size))
    totalInfer = int(len(video_reader) / batch_size)
    for _ in range(args.loop):
        for i in tqdm(range(0, len(video_reader), batch_size), total=total):
            # 跳过最后不满一个batch的数据
            if total != totalInfer and totalInfer * batch_size == i:
                continue
            batch_frame = video_reader[i:i + batch_size]
            batch_data = preprocess(model, batch_frame, text_prompt)

            st = time.time()
            batch_data_samples, results_list = infer(model, batch_data, tokenized, attention_mask, position_ids)
            infertime += (time.time() - st)

            data_samples.extend(batch_data_samples)
            results.extend(results_list)
    print(f"batch infer time: {infertime * 1000 / args.loop}ms",
          f" per frame infer time: {infertime * 1000 / args.loop / totalInfer / batch_size}ms")
    
    postprocess(video_tools, data_samples, results, text_prompt, args)


if __name__ == '__main__':
    main()