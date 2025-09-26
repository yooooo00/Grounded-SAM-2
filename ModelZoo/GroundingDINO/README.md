# GroundingDINO(TorchAir)-推理指导

- [GroundingDINO-推理指导](#GroundingDINO(TorchAir)-推理指导)
- [概述](#概述)
- [推理环境准备](#推理环境准备)
- [快速上手](#快速上手)
  - [获取源码](#获取源码)
  - [模型推理](#模型推理)
    - [1 开始推理验证](#1-开始推理验证)
    - [2 性能](#2-性能)
  - [FAQ](#faq)

******

# 概述
&emsp;&emsp;‌`GroundingDINO` 是一种最先进的开放集检测模型，可解决多项视觉任务，包括开放词汇检测(OVD)、Phrase Grounding(PG)、和指代性表达式理解(REC)。它的有效性已使其被广泛采用，成为各种下游应用的主流架构。

- 版本说明：
  ```
  url=https://github.com/open-mmlab/mmdetection
  commit_id=cfd5d3a9
  model_name=MM-GroundingDINO
  ```

# 推理环境准备
- 该模型需要以下插件与驱动  
  **表 1**  版本配套表

  | 配套                                                            |   版本 | 环境准备指导                                                                                          |
  | ------------------------------------------------------------    | ------ | ------------------------------------------------------------                                          |
  | 固件与驱动                                                       | 25.0.RC1 | [Pytorch框架推理环境准备](https://www.hiascend.com/document/detail/zh/ModelZoo/pytorchframework/pies) |
  | CANN                                                            |  8.1.RC1 | 包含kernels包和toolkit包                                                                                                   |
  | Python                                                          |  3.8 | -                                                                                                     |
  | PyTorch                                                         | 2.1.0 | -                                                                                                     |
  | Ascend Extension PyTorch                                        | 2.1.0.post10 | -                                                                                                     |
  | 说明：Atlas 800I A2 推理卡和Atlas 300I DUO 推理卡请以CANN版本选择实际固件与驱动版本。 |      \ | \                                                                                                     |


# 快速上手

## 获取源码
1. 获取本仓源码
   
   ```
   git clone https://gitee.com/ascend/ModelZoo-PyTorch.git
   cd ModelZoo-PyTorch/ACL_PyTorch/built-in/cv/GroundingDINO 
   ```

1. 获取开源模型源码
   ```
   git clone https://github.com/open-mmlab/mmdetection
   cd mmdetection
   git reset --hard cfd5d3a9
   ```
2. 下载相关权重和图片
   - 下载[BERT权重](https://huggingface.co/google-bert/bert-base-uncased/tree/main)，并放置于mmdetection目录下
   - 下载[MM-GroundingDINO权重](https://download.openmmlab.com/mmdetection/v3.0/mm_grounding_dino/grounding_dino_swin-b_pretrain_obj365_goldg_v3det/grounding_dino_swin-b_pretrain_obj365_goldg_v3de-f83eef00.pth)，并放置于weights目录下
   - 下载[animals.png](https://github.com/microsoft/X-Decoder/tree/main/inference_demo/images)，并放置于images目录下
   - 下载NLTK权重（可选）。MM-GroundingDINO在进行Phrase Grounding推理时可能会进行名词短语提取，虽然会在运行时下载特定的模型，但是考虑到有些用户运行环境无法联网，因此可以提前下载。
      - 方式一：下载[模型](https://www.nltk.org/nltk_data)到`~/nltk_data`路径下。
      - 方式二：采用如下命令下载
         ```
         import nltk
         nltk.download('punkt', download_dir='~/nltk_data')
         nltk.download('averaged_perceptron_tagger', download_dir='~/nltk_data')
         ```

3. 本地下载完成后的目录树如下，检查依赖项和脚本是否归档正确。
   ```shell
    mmdetection
    ├── demo
    │   ├── image_demo.py
    │   ├── video_demo.py
    │   ├── image_demo_npu.py //本仓提供单图推理脚本
    │   ├── video_demo_npu.py //本仓提供视频推理脚本
    │   ├── register_im2col_to_torchair.py //本仓提供torchair算子注册文件
    │   ├── register_roll_to_torchair.py //本仓提供torchair算子注册文件
    │   ├── requirements.txt //本仓提供
    │   └── install_requirements.sh //本仓提供依赖一键安装脚本
    ├── diff_patch
    │   ├── mmdetection_diff.patch  //本仓提供
    │   ├── mmengine_diff.patch     //本仓提供
    │   └── mmcv_diff.patch         //本仓提供
    ├── bert-base-uncased //BERT权重
    ├── weights
    │   └── grounding_dino_swin-b_pretrain_obj365_goldg_v3de-f83eef00.pth //MM-GroundingDINO权重
    ├── images
    │   └── animals.png
    ├── mmdet
    ├── resources
    ├── README.md
    ├── tests
    ├── tools
    ├── config
    └── ...
    ```


4. 安装依赖  
   ```
   conda create -n groundingdino python=3.8
   conda activate groundingdino

   #在mmdetection目录下执行依赖一键安装脚本，会在自动拉取依赖仓并应用diff patch文件
   source demo/install_requirements.sh
   ```


## 模型推理

### 1 开始推理验证

   1. 设置环境变量，执行推理命令

      ```
      # 指定使用NPU ID，默认为0
      export ASCEND_RT_VISIBLE_DEVICES=0

      # 执行图片推理命令
      python demo/image_demo_npu.py images/animals.png configs/mm_grounding_dino/grounding_dino_swin-b_pretrain_obj365_goldg_v3det.py --weight weights/grounding_dino_swin-b_pretrain_obj365_goldg_v3de-f83eef00.pth --texts '$: coco' --device npu (--loop 10)

      # 执行视频推理命令
      python demo/video_demo_npu.py demo/demo.mp4 configs/mm_grounding_dino/grounding_dino_swin-b_pretrain_obj365_goldg_v3det.py weights/grounding_dino_swin-b_pretrain_obj365_goldg_v3de-f83eef00.pth (--batch_size 16)
      ```
      在推理开始后，首先会默认执行warm_up，目的是执行首次编译，首次编译时间较长，在warm_up结束后，会执行推理操作，并打屏计算结果和性能数据。

### 2 性能

   - 单图性能，以animals.png为例，测得以下数据：
     

      |模型|芯片|E2E|forward|
      |------|------|------|---|
      |MM-GroundingDINO|Atlas 800I A2|877ms|274ms|
      |MM-GroundingDINO|Atlas 300I DUO|1378ms|740ms|
      
      - forward性能在`mmdet/apis/det_inferencer.py`中的`preds = self.forward(data, **forward_kwargs)`前后打点得到。

   - 视频性能：

      |模型|芯片|视频分辨率|batch_size|forward|
      |------|----|---|---|---|
      |MM-GroundingDINO|Atlas 800I A2|1080P|1bs|140ms|
      |MM-GroundingDINO|Atlas 800I A2|1080P|16bs|124ms|

      - forward性能从`demo/video_demo_npu.py`执行结束打屏信息获取，对应单针耗时`per frame infer time`
      - config配置文件`configs\mm_grounding_dino\grounding_dino_swin-t_pretrain_obj365.py`中的`scale=(800, 1333)`控制图片缩放，减小缩放比例可提升模型性能，已经测试coco数据集精度在(800, 1152)配置下精度mAP为0.522，(800, 1024)配置下精度mAP为0.52，具体配置选择由用户自行评估。

## FAQ
1. mmcv源码安装报错参考：https://mmcv.readthedocs.io/zh-cn/2.x/faq.html

