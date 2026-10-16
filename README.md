# D435i Handheld Reconstruction

使用手持 Intel RealSense D435i 采集 RGB-D 序列，通过 RGB-D Odometry、Pose Graph 和 TSDF 融合生成彩色点云及三维 Mesh。

主要入口为 `run_handheld_reconstruction.py`。数据录制和提取完成后，该入口执行输入检查、轨迹估计、约束验证、重建和报告输出；融合与模型处理复用 `run_pipeline.py`。

长序列扫描仍可能出现轨迹漂移、表面重影、拉伸和缺面。目前没有独立实测标定结果，不能给出确定的毫米误差。建议先验证短序列，再扩大帧范围；生成模型用于实验性建模，不用于精密测量或制造。

## 环境与安装

已验证环境：Windows 11、Python 3.14、Open3D 0.20.0。依赖版本见 [requirements.txt](requirements.txt)。相机录制及 DB3 回放需要兼容的 RealSense SDK。

```powershell
git clone https://github.com/lilili-maker/d435i-handheld-reconstruction.git
cd d435i-handheld-reconstruction
python -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
$env:PYTHONUTF8 = "1"
$env:PYTHONIOENCODING = "utf-8"
```

已有虚拟环境时，可直接使用其中的 Python。以下命令在仓库根目录执行。

## 录制与提取

保持家具和背景静止，关闭 RealSense Viewer。相机缓慢移动，相邻视角保留足够重叠。旋转椅的座面与底座需要固定，避免扫描中发生相对转动。拍摄距离、速度和覆盖范围见[拍摄说明](docs/SCANNING_GUIDE.md)。

```powershell
$scanId = Get-Date -Format 'yyyyMMdd_HHmmss'
$recording = "data/raw/scan_$scanId.db3"
$dataset = "data/extracted_$scanId"
$analysis = "output/analysis_$scanId"
$result = "output/handheld/scan_$scanId"

.\.venv\Scripts\python.exe src/02_record_data.py --output $recording
.\.venv\Scripts\python.exe src/03_extract_rgbd.py --input $recording --output $dataset
.\.venv\Scripts\python.exe src/reconstruction/analyze_handheld_dataset.py `
  --dataset $dataset --output $analysis
```

录制窗口按 q 停止，等待文件关闭后再提取。默认优先使用 640×480、30 FPS 的 Color/Depth；IMU 按设备支持情况录制，当前重建未使用视觉惯性融合。

检查 `dataset_analysis.json` 中的帧配对、时间戳、深度有效率和候选连续区间，结合 `rgb_overview.png` 选择 RGB 清晰、深度稳定的短区间。输入检查只验证数据基本条件，不代表整段轨迹可靠。

## 重建

例如，确认第 10～60 帧可用后运行：

```powershell
.\.venv\Scripts\python.exe run_handheld_reconstruction.py `
  --dataset $dataset --start 10 --end 60 --step 1 --reference 10 `
  --odometry-backend legacy --device CPU:0 --cpu-threads 4 `
  --min-consistency 0.6 --min-loop-gap 50 --max-loop-candidates 10 `
  --fusion-step 1 --no-crop --output $result
```

帧范围应按本次数据调整。`--reference` 指定临时世界坐标系，必须属于采样列表；程序从参考帧向两个方向分别跟踪。尚未确定裁剪范围时使用 `--no-crop`，输出场景 Mesh。

初次运行建议使用 Legacy CPU 后端。Tensor 后端和 GPU 可在同一短区间进行比较；速度和效果取决于数据及 Open3D 构建。`--device auto` 在 CUDA 不可用时退回 CPU，显式指定不可用的 `CUDA:0` 则报错。

| 参数 | 含义 |
| --- | --- |
| `--start`、`--end`、`--step` | 闭区间采样；step 为正且整除 end-start |
| `--reference` | 参考帧，当前需手动选择 |
| `--odometry-backend` | `legacy` 或 `tensor`，默认 tensor |
| `--cpu-threads` | CPU 线程数，默认 4 |
| `--min-consistency` | 双向深度重投影支持门槛，默认 0.4；上例取 0.6 |
| `--max-translation`、`--max-rotation` | 相对运动筛查阈值，默认 0.15 m、15° |
| `--fusion-step` | 每 N 个有效跟踪帧融合一次，默认 1 |
| `--skip-pointcloud` | 跳过独立点云合并，仍生成 Mesh |
| `--max-loop-candidates` | 非相邻约束候选数上限；0 关闭检查 |
| `--max-skip-frames` | 默认 0；允许跳帧时仍需重新配准通过检查 |
| `--max-recovery-gap-ms` | 恢复候选距最后可靠帧的时间差上限，默认 200 ms |
| `--icp-refinement` | 默认 off；geometry 用于显式诊断 |

`--min-consistency 0.4` 表示支持门槛，不是 40% 的位姿准确率。降低阈值可能接纳错误配准。

计算量较大时，可使用 `--fusion-step 3 --skip-pointcloud` 减少融合点数和输出开销，Odometry 采样不变。手持入口生成的融合配置采用深度范围 0.2～3 m、TSDF 体素 0.01 m、截断距离 0.04 m；其他配置见 `run_pipeline.py --help` 及生成的 JSON。

## 检查结果

输出目录独立创建，已有数据和模型不会被覆盖。

```text
output/handheld/<本次运行>/
  handheld.log
  handheld_report.json
  tracking/
    poses_before.json
    registration_quality.json
  pose_graph/
    poses_optimized_proposal.json
    poses_for_reconstruction.json
    pose_graph_report.json
    trajectory_comparison.png
  models/run_0001/
    mesh/scene_mesh.ply
    pointcloud/merged.ply       # 使用 --skip-pointcloud 时不生成
    reports/
    pipeline_report.json
  pipeline_config.json
  model_quality_report.json
```

先核对 `handheld_report.json` 中的 `tracking_complete`、`tracking.valid_frame_count`、`actual_reconstruction_frame_count` 和 `fusion_frame_ids`。`status=complete` 也可能仅表示成功跟踪的子段已完成融合，应以实际帧数和失败记录为准。

在 CloudCompare 或 Open3D 中打开 `scene_mesh.ply`，检查整体比例、双层表面、断裂和悬浮碎片，并与实物测量比较尺寸。裁剪前应在本次模型坐标系中确认 AABB，参数见 `src/reconstruction/clean_mesh.py --help`。手持入口启用裁剪时还会比较指定基线；没有基线模型时，可先重建，再单独裁剪。

## 跟踪失败与坐标约定

配准失败的帧和原因写入 `tracking/registration_quality.json`。默认停止失败方向的跟踪，未估计帧不参与融合，也不使用单位矩阵或上一帧位姿补齐。后续连续片段可以独立重建；未建立可靠相对变换前，不能直接合并不同片段。

```text
T_world_camera：Camera-to-World，平移单位为米
P_world = T_world_camera @ P_camera
P_target = T_source_to_target @ P_source
C_target = C_source @ inverse(T_source_to_target)
TSDF extrinsic = inverse(T_world_camera)
```

位姿来源标记为 `rgbd_odometry`。没有可靠非相邻约束时，Pose Graph 无法消除链式轨迹的累计漂移；优化完成不等于实际位姿准确。金属反光、黑色材料、细杆、遮挡和物体运动也是常见误差来源，详见[限制说明](docs/LIMITATIONS.md)。

## 已有数据格式

已提取的 RGB-D 可以直接用于输入检查和重建。

```text
data/extracted_scan/
  color/000000.png ...
  depth/000000.png ...
  camera_intrinsics.json
  timestamps.csv
```

RGB 与 Depth 按同编号配对，Depth 应对齐到 Color，分辨率与内参一致。深度 PNG 使用原始二维 `uint16`。

内参文件包含 `width`、`height`、`fx`、`fy`、`cx`、`cy`、`depth_scale`，并声明 `intrinsics_unit=pixels`、`depth_scale_unit=meters_per_depth_unit`、`depth_dtype=uint16`、`depth_aligned_to=color`。时间戳 CSV 包含 `frame_id,color_timestamp_ms,depth_timestamp_ms`。

RealSense depth scale 的单位为 m/unit，Open3D 使用其倒数作为深度除数。例如 0.001 m/unit 对应 Open3D depth_scale=1000。程序按文件实际值转换。

仓库不附真实长序列或实验室模型。DB3 输入应来自兼容 SDK 的 RealSense 录制文件。

## 离线示例与测试

无需相机即可用已知三面角的合成 RGB-D 检查软件流程：

```powershell
$demoId = Get-Date -Format 'yyyyMMdd_HHmmss'
$demoInputs = "output/demo_inputs_$demoId"
$demoOutput = "output/demo_reconstruction_$demoId"

.\.venv\Scripts\python.exe scripts/create_synthetic_demo.py --output $demoInputs
.\.venv\Scripts\python.exe run_handheld_reconstruction.py `
  --dataset "$demoInputs/synthetic_dataset" `
  --start 0 --end 2 --step 1 --reference 0 `
  --odometry-backend legacy --device CPU:0 --min-consistency 0.6 `
  --min-loop-gap 50 --max-loop-candidates 0 --no-crop --output $demoOutput

.\.venv\Scripts\python.exe -m unittest discover -s tests -v
```

示例从图像估计运动并融合三帧，已知位姿另存于 `poses/camera.json`，标记为 simulation，不作为估计输入。合成结果用于验证算法实现，不能替代真实扫描精度评估。

现有 130 项离线测试，覆盖变换方向、两个 Odometry 后端、Pose Graph、TSDF、失败处理及输出检查。[GitHub Actions](https://github.com/lilili-maker/d435i-handheld-reconstruction/actions)执行测试和合成重建。

- [拍摄说明](docs/SCANNING_GUIDE.md)
- [实验与验证结果](docs/VALIDATION.md)
- [贡献说明](CONTRIBUTING.md)
- [3DSVG Studio：照片或点云转 SVG 描边底稿](https://github.com/lilili-maker/3dsvg-studio)

## 许可

仓库尚未附开源许可证。代码与数据的复制、修改及再分发需获得相应授权。
