import os
import time
import json
import argparse
from pathlib import Path
from typing import List, Dict, Any

import numpy as np
import cv2
import torch
from torchvision.ops import box_convert
import pycocotools.mask as mask_util

from grounding_dino.groundingdino.util.inference import load_model, predict as gdino_predict
from sam2.build_sam import build_sam2
from sam2.sam2_image_predictor import SAM2ImagePredictor

from torch_npu.contrib import transfer_to_npu

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


def encode_mask_rle(mask: np.ndarray) -> Dict[str, Any]:
    rle = mask_util.encode(np.array(mask[:, :, None], order="F", dtype="uint8"))[0]
    rle["counts"] = rle["counts"].decode("utf-8")
    return rle


def list_images(image_dir: Path, num: int) -> List[Path]:
    exts = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
    imgs = [p for p in image_dir.iterdir() if p.suffix.lower() in exts]
    imgs.sort()
    return imgs[:num] if num > 0 else imgs


def run_tests(
    image_dir: Path,
    text_prompt: str,
    device: str,
    sam_cfg: str,
    sam_ckpt: str,
    gd_cfg: str,
    gd_ckpt: str,
    box_th: float,
    text_th: float,
    num_runs: int,
    out_dir: Path,
    amp: bool = True,
    server_preproc: bool = True,
    global_warmup: int = 0,
    per_shape_warmup: bool = False,
    compile_gdino: bool = False,
    compile_backend: str = "",
):
    out_dir.mkdir(parents=True, exist_ok=True)

    # 加载模型
    gdino = load_model(gd_cfg, gd_ckpt, device=device)
    # Optional: compile GroundingDINO (torch.compile / torchair backend)
    if compile_gdino:
        try:
            import torch._dynamo  # type: ignore
            # Allow partial compilation: unsupported parts fall back to eager
            torch._dynamo.config.suppress_errors = True  # noqa: SLF001
            # Enable capturing 0-d tensor scalar outputs used in size/steps (e.g., linspace/arange)
            torch._dynamo.config.capture_scalar_outputs = True

            if compile_backend.lower() == "torchair":
                # Compile hotspot submodule only to avoid tokenizer/backbone Python control flow
                import torchair as tng  # type: ignore
                from torchair.configs.compiler_config import CompilerConfig  # type: ignore
                cfg = CompilerConfig()
                cfg.experimental_config.frozen_parameter = True
                cfg.experimental_config.tiling_schedule_optimize = True
                npu_backend = tng.get_npu_backend(compiler_config=cfg)

                if hasattr(gdino, "transformer"):
                    gdino.transformer = torch.compile(
                        gdino.transformer, dynamic=False, fullgraph=False, backend=npu_backend
                    )
                    try:
                        tng.use_internal_format_weight(gdino.transformer)
                    except Exception:
                        pass
                    print("[compile] compiled module: transformer (torchair)")
                else:
                    # Fallback to partial model compile (may graph-break often)
                    gdino = torch.compile(gdino, dynamic=False, fullgraph=False, backend=npu_backend)
                    print("[compile] compiled model (partial, torchair)")
            else:
                # Inductor: allow graph breaks and compile hotspots
                if hasattr(gdino, "transformer"):
                    gdino.transformer = torch.compile(
                        gdino.transformer, fullgraph=False, dynamic=False, mode="reduce-overhead"
                    )
                    print("[compile] compiled module: transformer (inductor)")
                else:
                    gdino = torch.compile(gdino, fullgraph=False, dynamic=False, mode="reduce-overhead")
                    print("[compile] compiled model (partial, inductor)")
            print("[compile] GroundingDINO compiled with backend:", (compile_backend or "inductor"))
        except Exception as e:
            print("[compile] skip compile due to:", repr(e))
    sam2 = build_sam2(sam_cfg, sam_ckpt, device=device)
    predictor = SAM2ImagePredictor(sam2)
    gdino.eval(); predictor.model.eval(); 
    import torch as _torch
    _torch.set_grad_enabled(False)


    # 选图
    images = list_images(image_dir, num_runs)
    if not images:
        raise FileNotFoundError(f"未在 {image_dir} 找到测试图片")

    # 结果集合
    dino_details, sam2_details, end2end_details = [], [], []
    dino_times, sam2_times, end2end_times = [], [], []

    # 可选：全局预热（不计时），跑完整条链路以构建初始图/缓存
    if global_warmup and global_warmup > 0:
        for _ in range(global_warmup):
            for img_path in images:
                img_path_str = str(img_path)
                img_bgr = cv2.imread(img_path_str)
                if img_bgr is None:
                    continue
                image_source = img_bgr.copy()
                image_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
                image = (image_rgb.astype(np.float32) / 255.0).transpose(2, 0, 1)
                image_tensor = torch.tensor(image, dtype=torch.float32).to(device)
                with autocast_ctx(device, enabled=amp, dtype=torch.bfloat16):
                    boxes, confs, labels = gdino_predict(
                        gdino, image_tensor, text_prompt, box_th, text_th, device
                    )
                    sync(device)
                    predictor.set_image(image_source)
                    sync(device)
                    # 转换坐标供 predict 使用
                    h, w, _ = image_source.shape
                    boxes_px = boxes * torch.tensor([w, h, w, h])
                    xyxy = box_convert(boxes=boxes_px, in_fmt="cxcywh", out_fmt="xyxy").numpy().astype(np.float32, copy=True)
                    predictor.predict(point_coords=None, point_labels=None, box=xyxy, multimask_output=False)
                    sync(device)

    # 可选：按形状预热一次（不计时），避免首次遇到新形状的编译噪声
    seen_shapes = set()

    for idx, img_path in enumerate(images, 1):
        img_path_str = str(img_path)
        # 与客户服务端一致的预处理：BGR -> RGB -> /255 -> HWC->CHW -> float32 tensor
        img_bgr = cv2.imread(img_path_str)
        if img_bgr is None:
            raise FileNotFoundError(f"读取图片失败: {img_path_str}")
        image_source = img_bgr.copy()
        image_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
        image = (image_rgb.astype(np.float32) / 255.0).transpose(2, 0, 1)
        image_tensor = torch.tensor(image, dtype=torch.float32).to(device)
        h, w, _ = image_source.shape

        # 可选：遇到新形状，先做一次完整链路预热（不计时）
        if per_shape_warmup:
            shape_key = (image_source.shape[0], image_source.shape[1])
            if shape_key not in seen_shapes:
                with autocast_ctx(device, enabled=amp, dtype=torch.bfloat16):
                    _boxes, _confs, _labels = gdino_predict(
                        gdino, image_tensor, text_prompt, box_th, text_th, device
                    )
                    sync(device)
                    predictor.set_image(image_source)
                    sync(device)
                    _boxes_px = _boxes * torch.tensor([w, h, w, h])
                    _xyxy = box_convert(boxes=_boxes_px, in_fmt="cxcywh", out_fmt="xyxy").numpy().astype(np.float32, copy=True)
                    if len(_xyxy) > 0:
                        predictor.predict(point_coords=None, point_labels=None, box=_xyxy, multimask_output=False)
                        sync(device)
                seen_shapes.add(shape_key)

        # Grounding DINO 计时
        t0 = time.perf_counter()
        with autocast_ctx(device, enabled=amp, dtype=torch.bfloat16):
            boxes, confs, labels = gdino_predict(
                gdino, image_tensor, text_prompt, box_th, text_th, device
            )
            sync(device)
        t1 = time.perf_counter()
        dino_times.append(t1 - t0)

        boxes_px = boxes * torch.tensor([w, h, w, h])
        xyxy = box_convert(boxes=boxes_px, in_fmt="cxcywh", out_fmt="xyxy").numpy().astype(np.float32, copy=True)

        dino_details.append({
            "test_index": idx,
            "image_path": img_path_str,
            "text_prompt": text_prompt,
            "inference_time_s": round(t1 - t0, 4),
            "boxes_xyxy": xyxy.tolist(),
            "confidences": confs.numpy().tolist(),
            "labels": labels,
            "image_height": h,
            "image_width": w,
            "success": True,
        })

        # SAM2 计时（若无框则跳过）
        if len(xyxy) == 0:
            sam2_details.append({
                "test_index": idx,
                "image_path": img_path_str,
                "error": "no detections",
                "success": False,
            })
            end2end_details.append({
                "test_index": idx,
                "image_path": img_path_str,
                "error": "no detections",
                "success": False,
            })
            continue

        # set_image（嵌入）+ decode（分割）总时间视为 SAM2 推理时间
        t2 = time.perf_counter()
        with autocast_ctx(device, enabled=amp, dtype=torch.bfloat16):
            predictor.set_image(image_source)
            sync(device)
            masks, scores, logits = predictor.predict(
                point_coords=None, point_labels=None, box=xyxy, multimask_output=False
            )
            sync(device)
        t3 = time.perf_counter()
        sam2_times.append(t3 - t2)

        if masks.ndim == 4:
            masks = masks.squeeze(1)
        mask_rles = [encode_mask_rle(mask) for mask in masks]

        sam2_details.append({
            "test_index": idx,
            "image_path": img_path_str,
            "inference_time_s": round(t3 - t2, 4),
            "input_bboxes_xyxy": xyxy.tolist(),
            "masks_rle": mask_rles,
            "scores": scores.tolist(),
            "image_height": h,
            "image_width": w,
            "success": True,
        })

        # 端到端耗时
        end2end_times.append((t3 - t0))
        end2end_details.append({
            "test_index": idx,
            "image_path": img_path_str,
            "end2end_inference_time_s": round(t3 - t0, 4),
            "boxes": xyxy.tolist(),
            "confidences": confs.numpy().tolist(),
            "labels": labels,
            "masks_rle": mask_rles,
            "segmentation_scores": scores.tolist(),
            "image_height": h,
            "image_width": w,
            "success": True,
        })

    # 保存详细结果
    def dump(obj, name):
        p = out_dir / name
        with open(p, "w", encoding="utf-8") as f:
            json.dump(obj, f, ensure_ascii=False, indent=2)
        return p

    dump(dino_details, "grounding_dino_detailed_results.json")
    dump(sam2_details, "sam2_detailed_results.json")
    dump(end2end_details, "combined_end2end_detailed_results.json")

    # 汇总
    def summarize(times: List[float], total: int, label: str):
        ok = len(times)
        if ok == 0:
            return {"模型": label, "成功测试数": 0, "测试图片总数": total}
        times = np.array(times)
        return {
            "模型": label,
            "测试图片总数": total,
            "成功测试数": ok,
            "平均耗时(s)": round(float(times.mean()), 4),
            "耗时标准差(s)": round(float(times.std()), 4) if ok > 1 else 0.0,
            "最小耗时(s)": round(float(times.min()), 4),
            "最大耗时(s)": round(float(times.max()), 4),
            "吞吐量(img/s)": round(float(ok / times.sum()), 4),
        }

    summary = {
        "Grounding DINO": summarize(dino_times, len(images), "Grounding DINO"),
        "SAM2": summarize(sam2_times, len(images), "SAM2"),
        "End2End": summarize(end2end_times, len(images), "Grounding DINO + SAM2"),
        "输出目录": str(out_dir),
        "图片目录": str(image_dir),
        "提示词": text_prompt,
    }
    dump(summary, "perf_summary.json")
    print("保存测试结果到:", out_dir)
    return summary


def main():
    p = argparse.ArgumentParser(description="本地批量测试 Grounding DINO + SAM2（不经 HTTP）")
    p.add_argument("--image-dir", default="test/groundino_sam21_xn/img_src")
    p.add_argument("--text", default="person. car. dog. Burgers. Pizza. bicycles. boat. mouses. keyboards. tv. microwave. oven. refrigerator. bottles. banana. oranges. chairs. potted plant.")
    p.add_argument("--sam-cfg", default="configs/sam2.1/sam2.1_hiera_l.yaml")
    p.add_argument("--sam-ckpt", default="checkpoints/sam2.1_hiera_large.pt")
    p.add_argument("--gd-cfg", default="grounding_dino/groundingdino/config/GroundingDINO_SwinT_OGC.py")
    p.add_argument("--gd-ckpt", default="gdino_checkpoints/groundingdino_swint_ogc.pth")
    p.add_argument("--box-th", type=float, default=0.35)
    p.add_argument("--text-th", type=float, default=0.25)
    p.add_argument("--num-runs", type=int, default=10)
    p.add_argument("--device", choices=["npu", "cpu"], default="npu")
    p.add_argument("--out", default="outputs/test_run")
    p.add_argument("--server-preproc", action="store_true", default=True, help="使用与客户服务端一致的图像预处理")
    p.add_argument("--global-warmup", type=int, default=0, help="全局预热轮次（不计时，先构建图/缓存）")
    p.add_argument("--per-shape-warmup", action="store_true", help="遇到新形状先预热一次（不计时）")
    p.add_argument("--compile-gdino", action="store_true", help="对 GroundingDINO 启用 torch.compile")
    p.add_argument("--compile-backend", choices=["", "inductor", "torchair"], default="", help="编译后端选择；空为 inductor")
    p.add_argument("--no-amp", action="store_true")
    args = p.parse_args()

    device = "npu" if (args.device == "npu" and hasattr(torch, "npu") and torch.npu.is_available()) else "cpu"
    out_dir = Path(args.out)
    image_dir = Path(args.image_dir)
    amp = not args.no_amp

    summary = run_tests(
        image_dir=image_dir,
        text_prompt=args.text,
        device=device,
        sam_cfg=args.sam_cfg,
        sam_ckpt=args.sam_ckpt,
        gd_cfg=args.gd_cfg,
        gd_ckpt=args.gd_ckpt,
        box_th=args.box_th,
        text_th=args.text_th,
        num_runs=args.num_runs,
        out_dir=out_dir,
        amp=amp,
        server_preproc=args.server_preproc,
        global_warmup=args.global_warmup,
        per_shape_warmup=args.per_shape_warmup,
        compile_gdino=args.compile_gdino,
        compile_backend=args.compile_backend,
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
