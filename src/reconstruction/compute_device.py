"""统一选择 Open3D 运算设备；不能把检测到显卡当成已能运行 CUDA。"""

import open3d as o3d


def select_device(requested="auto"):
    if requested not in ("auto", "CPU:0", "CUDA:0"):
        raise ValueError("device 必须是 auto、CPU:0 或 CUDA:0")
    available = o3d.core.cuda.is_available()
    if requested == "CUDA:0" and not available:
        raise ValueError("当前 Open3D/Python 构建不支持 CUDA；请使用 auto/CPU:0，或安装匹配的官方 CUDA 构建。")
    device = "CUDA:0" if available and requested != "CPU:0" else "CPU:0"
    return device, {"requested": requested, "selected": device,
                    "cuda_available": available,
                    "cuda_compiled": bool(o3d._build_config.get("BUILD_CUDA_MODULE")),
                    "open3d_version": o3d.__version__,
                    "reason": "CUDA available" if device == "CUDA:0" else "CPU explicitly selected or current CUDA runtime/build unavailable"}
