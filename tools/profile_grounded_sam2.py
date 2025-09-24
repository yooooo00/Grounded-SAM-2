import os
import time
import argparse
import numpy as np
import torch
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
            # 兼容旧版 PyTorch 的 positional 形式
            return torch.autocast(device, dtype=dtype)


def sync(device: str):
    if device == "npu" and hasattr(torch, "npu") and torch.npu.is_available():
        torch.npu.synchronize()
    elif device == "cuda" and torch.cuda.is_available():
        torch.cuda.synchronize()


def enable_cpu_move_tracer():
    """追踪 .cpu() / .to('cpu')，帮助发现 CPU 往返。"""
    _orig_cpu = torch.Tensor.cpu
    _orig_to = torch.Tensor.to

    def _cpu(self):
        print(f"[trace] .cpu() {tuple(getattr(self,'shape',()))} {self.device} -> cpu")
        return _orig_cpu(self)

    def _to(self, *args, **kw):
        tgt = None
        if args and isinstance(args[0], (str, torch.device)):
            tgt = str(args[0])
        if kw.get("device", None) is not None:
            tgt = str(kw["device"]) 
        if tgt and tgt.startswith("cpu"):
            print(f"[trace] .to(cpu) {tuple(getattr(self,'shape',()))} {self.device} -> cpu")
        return _orig_to(self, *args, **kw)

    torch.Tensor.cpu = _cpu
    torch.Tensor.to = _to


def hook_deform_modules(model, device: str = "npu"):
    """给模型内包含 'deform' 字样的模块打 forward 钩子，打印 in/out 设备与耗时。"""
    import re

    patt = re.compile(r"deform|msdeform|multiscale.*deform", re.I)

    def wrap(mod, name):
        if getattr(mod, "_hooked", False):
            return
        orig = getattr(mod, "forward", None)
        if not callable(orig):
            return

        def fwd(*inp, **kw):
            # 只打印前 3 个张量信息
            printed = 0
            for i, t in enumerate(inp):
                if hasattr(t, "device") and hasattr(t, "shape"):
                    print(
                        f"[{name}] IN{i}: device={t.device} shape={tuple(t.shape)} dtype={getattr(t,'dtype',None)}"
                    )
                    printed += 1
                if printed >= 3:
                    break
            sync(device)
            t0 = time.perf_counter()
            out = orig(*inp, **kw)
            sync(device)
            t1 = time.perf_counter()
            if hasattr(out, "device") and hasattr(out, "shape"):
                print(f"[{name}] OUT: device={out.device} shape={tuple(out.shape)}")
            print(f"[{name}] time={1000*(t1-t0):.2f} ms")
            return out

        mod.forward = fwd
        mod._hooked = True

    for n, m in model.named_modules():
        cls = m.__class__.__name__
        if patt.search((n + "." + cls).lower()):
            print("[hook] attach ->", n or cls)
            wrap(m, n or cls)


def run_once(
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
    h, w, _ = image_source.shape
    with autocast_ctx(device, enabled=use_amp, dtype=torch.bfloat16):
        # 检测（G-DINO）
        t0 = time.perf_counter()
        boxes, confs, labels = gdino_predict(
            gdino, image_tensor, text, box_th, text_th, device
        )
        sync(device)
        t1 = time.perf_counter()

        # SAM2 set_image（图像嵌入）
        t2s = time.perf_counter()
        predictor.set_image(image_source)
        sync(device)
        t2e = time.perf_counter()

        # 框坐标转换并分割（decode）
        boxes = boxes * torch.tensor([w, h, w, h])
        xyxy = box_convert(boxes=boxes, in_fmt="cxcywh", out_fmt="xyxy").numpy()
        t3 = time.perf_counter()
        masks, scores, logits = predictor.predict(
            point_coords=None, point_labels=None, box=xyxy, multimask_output=False
        )
        sync(device)
        t4 = time.perf_counter()

    return {
        "det_ms": (t1 - t0) * 1000.0,
        "embed_ms": (t2e - t2s) * 1000.0,
        "decode_ms": (t4 - t3) * 1000.0,
        "total_ms": (t4 - t0) * 1000.0,
        "n_boxes": int(len(confs)),
    }


def torch_profile_run(
    gdino,
    predictor,
    image_source,
    image_tensor,
    text,
    box_th,
    text_th,
    device,
    use_amp,
    out_dir,
):
    from torch.profiler import profile, ProfilerActivity

    acts = [ProfilerActivity.CPU]
    # Ascend NPU 归为 PrivateUse1
    try:
        acts.append(ProfilerActivity.PrivateUse1)
    except Exception:
        pass

    os.makedirs(out_dir, exist_ok=True)
    trace_path = os.path.join(out_dir, f"trace_{device}.json")

    def _step():
        return run_once(
            gdino,
            predictor,
            image_source,
            image_tensor,
            text,
            box_th,
            text_th,
            device,
            use_amp,
        )

    with profile(activities=acts, record_shapes=True, with_stack=False) as prof:
        stats = _step()
    try:
        prof.export_chrome_trace(trace_path)
        print(f"[profiler] chrome trace saved: {trace_path}")
    except Exception as e:
        print("[profiler] export failed:", e)

    # 粗略统计（仅供参考）
    try:
        ev = prof.key_averages()
        cpu_ms = sum([e.self_cpu_time_total for e in ev]) / 1000.0
        npu_ms = sum([getattr(e, "self_privateuse1_time_total", 0.0) for e in ev]) / 1000.0
        print(f"[profiler] CPU self_time≈{cpu_ms:.1f} ms, NPU self_time≈{npu_ms:.1f} ms")
    except Exception:
        pass

    return stats


def npu_profile_run(
    gdino,
    predictor,
    image_source,
    image_tensor,
    text,
    box_th,
    text_th,
    device,
    use_amp,
    out_dir,
):
    import torch_npu.profiler as npu_prof

    os.makedirs(out_dir, exist_ok=True)
    exp = npu_prof._ExperimentalConfig(
        profiler_level=npu_prof.ProfilerLevel.Level1, data_simplification=False
    )
    with npu_prof.profile(
        activities=[npu_prof.ProfilerActivity.CPU, npu_prof.ProfilerActivity.NPU],
        with_stack=False,
        record_shapes=False,
        profile_memory=False,
        schedule=npu_prof.schedule(wait=1, warmup=2, active=1, repeat=1, skip_first=0),
        experimental_config=exp,
        on_trace_ready=npu_prof.tensorboard_trace_handler(out_dir),
    ) as prof:
        stats = run_once(
            gdino,
            predictor,
            image_source,
            image_tensor,
            text,
            box_th,
            text_th,
            device,
            use_amp,
        )
    print(f"[profiler] ascend trace (tensorboard) at: {out_dir}")
    return stats


def main():
    p = argparse.ArgumentParser(description="Operator-level profiling for Grounded-SAM-2")
    p.add_argument("--image", default="notebooks/images/truck.jpg")
    p.add_argument("--text", default="car. tire.")
    p.add_argument("--sam_cfg", default="configs/sam2.1/sam2.1_hiera_l.yaml")
    p.add_argument("--sam_ckpt", default="checkpoints/sam2.1_hiera_large.pt")
    p.add_argument("--gd_cfg", default="grounding_dino/groundingdino/config/GroundingDINO_SwinT_OGC.py")
    p.add_argument("--gd_ckpt", default="gdino_checkpoints/groundingdino_swint_ogc.pth")
    p.add_argument("--box_th", type=float, default=0.35)
    p.add_argument("--text_th", type=float, default=0.25)
    p.add_argument("--device", choices=["npu", "cpu"], default="npu")
    p.add_argument("--amp", action="store_true", default=True)
    p.add_argument("--warmup", type=int, default=3)
    p.add_argument("--repeat", type=int, default=0, help="额外非 profile 的重复次数")
    p.add_argument(
        "--profile",
        choices=["torch", "npu", "none"],
        default="torch",
        help="选择 torch.profiler 或 torch_npu.profiler（或不做profile，仅计时）",
    )
    p.add_argument("--out", default="outputs/profile")
    p.add_argument("--hook-deform", action="store_true", default=False)
    p.add_argument("--trace-cpu-move", action="store_true", default=False)
    args = p.parse_args()

    device = "npu" if (args.device == "npu" and hasattr(torch, "npu") and torch.npu.is_available()) else "cpu"
    print(f"[device] {device}")

    # 构建模型（只加载一次）
    gdino = load_model(args.gd_cfg, args.gd_ckpt, device=device)
    sam2 = build_sam2(args.sam_cfg, args.sam_ckpt, device=device)
    predictor = SAM2ImagePredictor(sam2)

    # 加载输入
    image_source, image_tensor = load_image(args.image)

    gdino.eval()
    predictor.model.eval()
    torch.set_grad_enabled(False)

    # 可选：追踪 CPU 往返与 deform 模块钩子
    if args.trace_cpu_move:
        enable_cpu_move_tracer()
    if args.hook_deform:
        hook_deform_modules(gdino, device)

    # 预热（不采样）
    for _ in range(max(0, args.warmup)):
        _ = run_once(
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

    # profile 一次
    if args.profile == "torch":
        stats = torch_profile_run(
            gdino,
            predictor,
            image_source,
            image_tensor,
            args.text,
            args.box_th,
            args.text_th,
            device,
            args.amp,
            args.out,
        )
    elif args.profile == "npu":
        stats = npu_profile_run(
            gdino,
            predictor,
            image_source,
            image_tensor,
            args.text,
            args.box_th,
            args.text_th,
            device,
            args.amp,
            args.out,
        )
    else:
        stats = run_once(
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

    print(
        f"[SPLIT] G-DINO={stats['det_ms']:.1f} ms, SAM2_embed={stats['embed_ms']:.1f} ms, SAM2_decode={stats['decode_ms']:.1f} ms, TOTAL={stats['total_ms']:.1f} ms, n_boxes={stats['n_boxes']}"
    )

    # 额外重复（非 profile）
    if args.repeat and args.repeat > 0:
        dets, embeds, decs, totals = [], [], [], []
        for _ in range(args.repeat):
            s = run_once(
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
            dets.append(s["det_ms"])
            embeds.append(s["embed_ms"])
            decs.append(s["decode_ms"])
            totals.append(s["total_ms"])
        import csv

        os.makedirs(args.out, exist_ok=True)
        csv_path = os.path.join(args.out, f"summary_{device}.csv")
        with open(csv_path, "w", newline="") as f:
            w = csv.writer(f)
            w.writerow(["phase", "avg_ms", "p50_ms", "p90_ms", "repeat"])
            rep = lambda xs: (
                float(np.mean(xs)),
                float(np.percentile(xs, 50)),
                float(np.percentile(xs, 90)),
                len(xs),
            )
            w.writerow(["gdino", *rep(dets)])
            w.writerow(["sam2_embed", *rep(embeds)])
            w.writerow(["sam2_decode", *rep(decs)])
            w.writerow(["total", *rep(totals)])
        print(f"[save] CSV written: {csv_path}")


if __name__ == "__main__":
    main()

