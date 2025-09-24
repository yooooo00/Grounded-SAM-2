# bench_grounded_sam2.py
import os, time, statistics, argparse
import torch, numpy as np, cv2
from torchvision.ops import box_convert
from grounding_dino.groundingdino.util.inference import load_model, load_image, predict
from sam2.build_sam import build_sam2
from sam2.sam2_image_predictor import SAM2ImagePredictor
from contextlib import nullcontext

def autocast_ctx(device: str, enabled: bool = True, dtype=torch.bfloat16):
    """
    返回一个正确的 autocast 上下文：
    - NPU：用 torch.npu.amp.autocast(dtype=...)
    - CPU/CUDA：用 torch.autocast('cpu'/'cuda', dtype=...)
    - 兼容老 PyTorch：回退到 positional 形式 torch.autocast('cpu', dtype=...)
    """
    if not enabled:
        return nullcontext()
    if device == "npu":
        return torch.npu.amp.autocast(dtype=dtype)
    else:
        try:
            return torch.autocast(device_type=device, dtype=dtype)
        except TypeError:
            return torch.autocast(device, dtype=dtype)


def sync(device):
    if device == "cuda": torch.cuda.synchronize()
    elif device == "npu": torch.npu.synchronize()

def main(args):
    device = "npu" if torch.npu.is_available() and args.device=="npu" else "cpu"
    print(f"[device] {device}")

    # ---- load once
    gdino = load_model(args.gd_cfg, args.gd_ckpt, device=device)
    sam2  = build_sam2(args.sam_cfg, args.sam_ckpt, device=device)
    predictor = SAM2ImagePredictor(sam2)

    # input
    image_source, image = load_image(args.image)
    h, w, _ = image_source.shape

    # set eval / no grad
    gdino.eval(); predictor.model.eval()
    torch.set_grad_enabled(False)

    # bf16 autocast for NPU/CPU if可用
    amp_ctx = (torch.npu.amp.autocast if device=="npu"
               else torch.autocast) if args.amp else None

    # warmup
    for _ in range(args.warmup):
        if amp_ctx:
            with autocast_ctx(device, enabled=args.amp, dtype=torch.bfloat16):
                predictor.set_image(image_source)
                boxes, confs, labels = predict(gdino, image, args.text, args.box_th, args.text_th, device)
                boxes = boxes * torch.tensor([w, h, w, h])
                xyxy = box_convert(boxes=boxes, in_fmt="cxcywh", out_fmt="xyxy").numpy()
                predictor.predict(point_coords=None, point_labels=None, box=xyxy, multimask_output=False)
        else:
            predictor.set_image(image_source)
            boxes, confs, labels = predict(gdino, image, args.text, args.box_th, args.text_th, device)
            boxes = boxes * torch.tensor([w, h, w, h])
            xyxy = box_convert(boxes=boxes, in_fmt="cxcywh", out_fmt="xyxy").numpy()
            predictor.predict(point_coords=None, point_labels=None, box=xyxy, multimask_output=False)
        sync(device)

    # measure
    t_det, t_seg = [], []
    for _ in range(args.repeat):
        # detection
        if amp_ctx:
            with autocast_ctx(device, enabled=args.amp, dtype=torch.bfloat16):
                t0 = time.perf_counter()
                boxes, confs, labels = predict(gdino, image, args.text, args.box_th, args.text_th, device)
                sync(device); t1 = time.perf_counter()
                boxes = boxes * torch.tensor([w, h, w, h])
                xyxy = box_convert(boxes=boxes, in_fmt="cxcywh", out_fmt="xyxy").numpy()
                predictor.set_image(image_source)
                masks, scores, logits = predictor.predict(
                    point_coords=None, point_labels=None, box=xyxy, multimask_output=False)
                sync(device); t2 = time.perf_counter()
        else:
            t0 = time.perf_counter()
            boxes, confs, labels = predict(gdino, image, args.text, args.box_th, args.text_th, device)
            sync(device); t1 = time.perf_counter()
            boxes = boxes * torch.tensor([w, h, w, h])
            xyxy = box_convert(boxes=boxes, in_fmt="cxcywh", out_fmt="xyxy").numpy()
            predictor.set_image(image_source)
            masks, scores, logits = predictor.predict(
                point_coords=None, point_labels=None, box=xyxy, multimask_output=False)
            sync(device); t2 = time.perf_counter()

        t_det.append(t1 - t0)
        t_seg.append(t2 - t1)

    def rep(x): 
        return f"avg={np.mean(x)*1000:.1f}ms p50={np.percentile(x,50)*1000:.1f} p90={np.percentile(x,90)*1000:.1f}"
    print(f"[G-DINO] {rep(t_det)}")
    print(f"[SAM2 ] {rep(t_seg)}")
    print(f"[TOTAL] {rep([a+b for a,b in zip(t_det,t_seg)])}")

if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--image", default="notebooks/images/truck.jpg")
    p.add_argument("--text", default="car. tire.")
    p.add_argument("--sam_cfg", default="configs/sam2.1/sam2.1_hiera_l.yaml")
    p.add_argument("--sam_ckpt", default="checkpoints/sam2.1_hiera_large.pt")
    p.add_argument("--gd_cfg", default="grounding_dino/groundingdino/config/GroundingDINO_SwinT_OGC.py")
    p.add_argument("--gd_ckpt", default="gdino_checkpoints/groundingdino_swint_ogc.pth")
    p.add_argument("--box_th", type=float, default=0.35)
    p.add_argument("--text_th", type=float, default=0.25)
    p.add_argument("--device", choices=["cpu","npu"], default="npu")
    p.add_argument("--repeat", type=int, default=20)
    p.add_argument("--warmup", type=int, default=5)
    p.add_argument("--amp", action="store_true", default=True)
    args = p.parse_args()
    main(args)
