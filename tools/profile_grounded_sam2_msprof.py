import os
import sys
import shlex
import argparse
import subprocess
from pathlib import Path


def find_msprof() -> str:
    """在 PATH 或 ASCEND_TOOLKIT_HOME 下查找 msprof 可执行文件。"""
    from shutil import which

    p = which("msprof")
    if p:
        return p
    # 常见安装路径（Linux）
    home = os.environ.get("ASCEND_TOOLKIT_HOME") or os.environ.get("ASCEND_HOME")
    if home:
        cand = Path(home) / "tools" / "profiler" / "bin" / "msprof"
        if cand.exists():
            return str(cand)
    # 兜底：返回命令名，交由系统报错
    return "msprof"


def build_application_cmd(args: argparse.Namespace) -> str:
    """构造被分析的 Python 命令（不在进程内再开 profiler，避免叠加开销）。"""
    script = Path(__file__).resolve().parent / "profile_grounded_sam2.py"
    py = sys.executable
    # 使用我们已有的单次/重复计时逻辑，关闭内部 profiler，保留预热与重复次数
    cmd = [
        py,
        "-u",
        str(script),
        "--device",
        args.device,
        "--image",
        args.image,
        "--text",
        args.text,
        "--sam_cfg",
        args.sam_cfg,
        "--sam_ckpt",
        args.sam_ckpt,
        "--gd_cfg",
        args.gd_cfg,
        "--gd_ckpt",
        args.gd_ckpt,
        "--box_th",
        str(args.box_th),
        "--text_th",
        str(args.text_th),
        "--amp" if args.amp else "",
        "--warmup",
        str(args.warmup),
        "--repeat",
        str(args.repeat),
        "--profile",
        "none",
        "--out",
        args.out,
    ]
    # 去除空项
    cmd = [c for c in cmd if c]
    return shlex.join(cmd)


def build_msprof_cmd(app_cmd: str, args: argparse.Namespace) -> list:
    """构造 msprof 命令，按昇腾官方推荐项采集常用指标（aicore/aicpu/task/runtime）。"""
    msprof = find_msprof()
    out_dir = Path(args.msprof_out).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    # 典型推荐项：采集 Task、Runtime、AICPU、AICore 指标
    # 注：不同 CANN 版本支持的参数可能略有差异，如报未知参数可据实际环境删减。
    cmd = [
        msprof,
        f"--output={out_dir}",
        "--application",
        app_cmd,
        "--aicore",          # 采集 AICore 时间线/算子信息
        "--aicpu",           # 采集 AICPU 时间线
        "--task",            # 采集任务级时间线
        "--runtime",         # 采集 Runtime 接口
        # 可按需追加：'--data-process', '--training-trace' 等
    ]
    # 如果指定了设备 ID，可传递给 msprof（不同版本参数名可能不同，此处仅示例）
    if args.device_id is not None:
        cmd += [f"--device-id={args.device_id}"]
    return cmd


def main():
    p = argparse.ArgumentParser(description="使用 msprof 采集 Grounded-SAM-2 的昇腾推荐指标")
    # 基本推理参数（与现有 profile_grounded_sam2.py 对齐）
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
    p.add_argument("--warmup", type=int, default=5)
    p.add_argument("--repeat", type=int, default=20)
    p.add_argument("--out", default="outputs/profile")

    # msprof 相关
    p.add_argument("--msprof-out", default="outputs/msprof")
    p.add_argument("--device-id", type=int, default=None, help="可选：指定 ASCEND_DEVICE_ID")
    p.add_argument("--dry-run", action="store_true", help="仅打印命令，不执行")
    args = p.parse_args()

    # 构造被分析的 Python 命令
    app_cmd = build_application_cmd(args)
    # 构造 msprof 命令
    cmd = build_msprof_cmd(app_cmd, args)

    print("[msprof] command:")
    print(" ", shlex.join(cmd))

    if args.dry_run:
        return

    # 执行采集
    env = os.environ.copy()
    if args.device_id is not None:
        env["ASCEND_DEVICE_ID"] = str(args.device_id)
    # 推荐：限制 CPU 线程，降低抖动
    env.setdefault("OPENBLAS_NUM_THREADS", "32")
    env.setdefault("OMP_NUM_THREADS", "32")
    env.setdefault("GOTO_NUM_THREADS", "32")
    env.setdefault("NUMEXPR_NUM_THREADS", "32")
    env.setdefault("TOKENIZERS_PARALLELISM", "false")

    proc = subprocess.run(cmd, env=env)
    if proc.returncode != 0:
        print("[msprof] 采集失败，返回码:", proc.returncode)
        sys.exit(proc.returncode)

    print("[msprof] 采集完成，结果目录:", str(Path(args.msprof_out).resolve()))
    print("[hint] 使用 TensorBoard 或 Ascend 自带解析工具查看 AICore/AICPU/Task/Runtime 时间线与指标。")


if __name__ == "__main__":
    main()

