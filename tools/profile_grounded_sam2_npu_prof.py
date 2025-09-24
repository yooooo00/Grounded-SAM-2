import os
import argparse
import inspect
import time
from pathlib import Path

import numpy as np
import torch
import torch_npu  # noqa: F401
from torchvision.ops import box_convert

from grounding_dino.groundingdino.util.inference import (
    load_model,
    load_image,
    predict as gdino_predict,
)
from sam2.build_sam import build_sam2
from sam2.sam2_image_predictor import SAM2ImagePredictor


def autocast_ctx(device: str, enabled: bool = True, dtype=torch.bfloat16):
    if not enabled:
        from contextlib import nullcontext

        return nullcontext()
    if device == "npu":
        return torch.npu.amp.autocast(dtype=dtype)
    else:
        try:
            return torch.autocast(device_type=device, dtype=dtype)
        except TypeError:
            return torch.autocast(device, dtype=dtype)


def sync(device: str):
    if device == "npu" and hasattr(torch, "npu") and torch.npu.is_available():
        torch.npu.synchronize()
    elif device == "cuda" and torch.cuda.is_available():
        torch.cuda.synchronize()


def run_step(
    gdino,
    predictor: SAM2ImagePredictor,
    image_source,
    image_tensor,
    text: str,
    box_th: float,
    text_th: float,
    device: str,
    use_amp: bool,
):
    """一次完整的推理步骤（检测→嵌入→解码），用于 profiler 步进。"""
    h, w, _ = image_source.shape
    with autocast_ctx(device, enabled=use_amp, dtype=torch.bfloat16):
        # 检测（G-DINO）
        boxes, confs, labels = gdino_predict(
            gdino, image_tensor, text, box_th, text_th, device
        )
        sync(device)

        # SAM2 set_image（图像嵌入）
        predictor.set_image(image_source)
        sync(device)

        # 框坐标转换并分割（decode）
        boxes = boxes * torch.tensor([w, h, w, h])
        xyxy = box_convert(boxes=boxes, in_fmt="cxcywh", out_fmt="xyxy").numpy()
        predictor.predict(
            point_coords=None, point_labels=None, box=xyxy, multimask_output=False
        )
        sync(device)


def _build_experimental_config(npu_prof):
    """按文档推荐构建 ExperimentalConfig；兼容不同 CANN 版本参数。"""
    # 目标配置（尽量完整，后续按签名过滤）
    wanted = dict(
        export_type=[
            getattr(npu_prof.ExportType, "Text", None),
            getattr(npu_prof.ExportType, "Db", None),
        ],
        profiler_level=getattr(npu_prof.ProfilerLevel, "Level0", None),
        msprof_tx=False,
        mstx_domain_include=[],
        mstx_domain_exclude=[],
        aic_metrics=getattr(npu_prof.AiCMetrics, "AiCoreNone", None),
        l2_cache=False,
        op_attr=False,
        data_simplification=False,
        record_op_args=False,
        gc_detect_threshold=None,
        host_sys=[
            getattr(npu_prof.HostSystem, "CPU", None),
            getattr(npu_prof.HostSystem, "MEM", None),
        ],
        sys_io=False,
        sys_interconnection=False,
    )
    # 去除 None 项
    wanted = {k: v for k, v in wanted.items() if v is not None}
    # 过滤出 _ExperimentalConfig 支持的参数
    sig = inspect.signature(npu_prof._ExperimentalConfig)
    supported = {k: v for k, v in wanted.items() if k in sig.parameters}
    return npu_prof._ExperimentalConfig(**supported)


def main():
    p = argparse.ArgumentParser(description="使用 torch_npu.profiler.profile 采集 Grounded-SAM-2")
    p.add_argument("--device", choices=["npu", "cpu"], default="npu")
    p.add_argument("--image", default="notebooks/images/truck.jpg")
    p.add_argument("--text", default="car. tire.")
    p.add_argument("--sam_cfg", default="configs/sam2.1/sam2.1_hiera_l.yaml")
    p.add_argument("--sam_ckpt", default="checkpoints/sam2.1_hiera_large.pt")
    p.add_argument("--gd_cfg", default="grounding_dino/groundingdino/config/GroundingDINO_SwinT_OGC.py")
    p.add_argument("--gd_ckpt", default="gdino_checkpoints/groundingdino_swint_ogc.pth")
    p.add_argument("--box_th", type=float, default=0.35)
    p.add_argument("--text_th", type=float, default=0.25)
    p.add_argument("--amp", action="store_true", default=True)
    p.add_argument("--warmup", type=int, default=3)
    p.add_argument("--steps", type=int, default=5, help="采集步数，与 prof.step() 配套")
    p.add_argument("--out", default="outputs/npu_prof")
    p.add_argument("--export-chrome", action="store_true", help="同时导出单个 chrome_trace.json")
    args = p.parse_args()

    device = "npu" if (args.device == "npu" and hasattr(torch, "npu") and torch.npu.is_available()) else "cpu"
    print(f"[device] {device}")

    # 模型与输入
    gdino = load_model(args.gd_cfg, args.gd_ckpt, device=device)
    sam2 = build_sam2(args.sam_cfg, args.sam_ckpt, device=device)
    predictor = SAM2ImagePredictor(sam2)
    image_source, image_tensor = load_image(args.image)

    gdino.eval()
    predictor.model.eval()
    torch.set_grad_enabled(False)

    # 预热（不采样）
    for _ in range(max(0, args.warmup)):
        run_step(
            gdino,
            predictor,
            image_source,
            image_tensor,
            args.text,
            args.box_th,
            args.text_th,
            device,
            args.amp,
        )

    # torch_npu.profiler.profile（with 语句 + prof.step()）
    import torch_npu.profiler as npu_prof

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    experimental_config = _build_experimental_config(npu_prof)

    schedule = npu_prof.schedule(wait=0, warmup=0, active=1, repeat=1, skip_first=1)
    activities = [npu_prof.ProfilerActivity.CPU, npu_prof.ProfilerActivity.NPU]

    with npu_prof.profile(
        activities=activities,
        schedule=schedule,
        on_trace_ready=npu_prof.tensorboard_trace_handler(str(out_dir)),
        record_shapes=False,
        profile_memory=False,
        with_stack=True,
        with_modules=False,
        with_flops=False,
        experimental_config=experimental_config,
    ) as prof:
        for step in range(args.steps):
            run_step(
                gdino,
                predictor,
                image_source,
                image_tensor,
                args.text,
                args.box_th,
                args.text_th,
                device,
                args.amp,
            )
            prof.step()

    if args.export_chrome:
        # 根据文档，也可以导出单个 chrome trace（可与 tensorboard_trace_handler 并存）
        try:
            ct_path = out_dir / f"chrome_trace.json"
            # 需重新开启一次最小 profile 才能导出单文件（当前 API 无直接句柄）
            with npu_prof.profile() as p2:  # noqa: F821
                pass
            p2.export_chrome_trace(str(ct_path))  # type: ignore[attr-defined]
            print(f"[export] chrome trace saved: {ct_path}")
        except Exception as e:
            print("[export] chrome trace failed:", e)

    print(f"[npu-prof] 完成，结果目录: {str(out_dir.resolve())}")


if __name__ == "__main__":
    main()

