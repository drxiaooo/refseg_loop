# REFSEG_LOOP — 闭环式指代分割 + 自检纠错（SAM → CLIP → VLM）

本项目实现一个“闭环式多模态视觉系统”用于 **Referring Segmentation（指代分割）**：

- **SAM**：在图像上生成候选 masks（不依赖文本）
- **CLIP**：对候选 mask 与指代表达进行相似度排序，取 Top-1
- **VLM（可选）**：当置信度不足触发闭环，VLM 改写/细化 query，再次用 CLIP 重新排序
- 支持：闭环消融、可靠性评测（扰动/分桶）、效率/成本统计、案例可视化、真实图片应用

> 数据集：HuggingFace `moondream/refcoco-m`（validation split）  
> SAM 权重：`checkpoints/sam_vit_b_01ec64.pth`  
> VLM：硅基流动 OpenAI-compatible API（例如 Qwen/Qwen2.5-VL-72B-Instruct）

---

## 1. 环境准备

### 1.1 Python / 虚拟环境
建议 Python 3.10+（你当前可在 3.13 上跑通 CPU 版）。

在项目根目录创建并激活 venv（Windows PowerShell）：

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
```

安装依赖：

```powershell
python -m pip install -U pip
python -m pip install -r requirements.txt
```

### 1.2 SAM 权重文件（权重需自行下载，不随仓库提供）
将 SAM checkpoint 放到：

```
checkpoints/sam_vit_b_01ec64.pth
```

### 1.3 可选：配置 VLM（硅基流动）
在 PowerShell **当前终端会话**里设置环境变量（推荐用 `$env:`，立刻生效）：

```powershell
$env:SILICONFLOW_API_KEY="sk-xxxxxx"
```

> 注意：`setx` 写入系统环境变量后，需要 **重新打开终端/重启 VSCode** 才能在新进程里读到。

---

## 2. 项目结构说明

```
REFSEG_LOOP/
  assets/
    real/                     # 真实图片输入（你自定义的测试图）
      real1.png
      real2.png
      real3.png

  checkpoints/
    sam_vit_b_01ec64.pth      # SAM 权重

  figures/
    app_real/                 # 真实图片输出可视化（one-shot vs loop）
      real1_oneshot.png
      real1_loop_iter2.png
      ...
    app_refcocom/             # RefCOCO-M 案例可视化（自动挑选/对比）
      case_0.png
      case_1.png
      ...

  metrics/
    ious_loop_vlm1_iter1_n50.npy
    ious_loop_vlm1_iter2_n50.npy
    ious_loop_vlm1_iter3_n50.npy

  outputs/                    # 单张调试输出（示例：img/gt/pred）
    img.png
    gt_mask.png
    pred_mask_stub.png

  scripts/                    # 各类实验入口脚本
    eval_stub_clip.py         # one-shot（SAM proposals + CLIP 排序）mIoU
    eval_loop_stub.py         # 闭环评测（max_iter / use_vlm 可控）mIoU+iters
    eval_loop_novlm.py        # 闭环消融：loop 但禁用 VLM
    eval_cost.py              # 效率/成本：时间 + 调用次数统计（SAM/CLIP/VLM）
    eval_reliability.py       # 可靠性：噪声/模糊扰动 + clean 对比
    eval_size_buckets.py      # 小目标/长尾：按目标面积 small/medium/large 分桶
    bootstrap_ci.py           # bootstrap 置信区间（对 metrics/*.npy）
    visualize_cases.py        # 批量可视化案例（one-shot vs loop）
    select_cases_from_metrics.py  # 从 IoU 序列挑选提升/退化/无变化案例
    visualize_selected_cases.py   # 把挑选出的 idx 生成可视化图
    run_real_images.py        # 真实图片应用（one-shot/loop 输出到 figures/app_real）

  src/                        # 核心模块（被 scripts 调用）
    sam_proposer.py           # SAM 候选生成器（SamPointProposer / GridProposer 等）
    clip_ranker.py            # CLIP Ranker：对候选 masks 进行文本一致性排序（含 roi_calls）
    vlm_policy.py             # VLM 策略：refine_query（硅基流动 API 调用）
    loop_runner.py            # 闭环控制器：停止条件 conf/gap + 缓存 proposals + 迭代逻辑
    eval_metrics.py           # iou_mask 等评测指标
    perturb.py                # 扰动：add_gaussian_noise / gaussian_blur
```

---

## 3. 快速开始（推荐顺序）

### 3.1 one-shot 基线（SAM→CLIP）
```powershell
.\.venv\Scripts\python.exe -m scripts.eval_stub_clip --n 50
```

输出：`mIoU`（基线性能）

### 3.2 闭环系统（loop + VLM）
分别跑不同迭代上限（作业常用对比）：

```powershell
.\.venv\Scripts\python.exe -m scripts.eval_loop_stub --n 50 --max_iter 1 --use_vlm 1
.\.venv\Scripts\python.exe -m scripts.eval_loop_stub --n 50 --max_iter 2 --use_vlm 1
.\.venv\Scripts\python.exe -m scripts.eval_loop_stub --n 50 --max_iter 3 --use_vlm 1
```

输出：
- `n=..., mIoU=..., avg_iters=...`
- 生成 `metrics/ious_*.npy`（每张图的 IoU 序列）
- 输出工具调用次数（SAM/CLIP ROI/VLM）

### 3.3 消融：无 VLM loop
```powershell
.\.venv\Scripts\python.exe -m scripts.eval_loop_novlm --n 50
```

用于证明：闭环结构本身 vs VLM 的增益来源。

---

## 4. 可靠性评测

### 4.1 扰动鲁棒性（噪声/模糊）
```powershell
.\.venv\Scripts\python.exe -m scripts.eval_reliability --n 50
```

输出：
- clean / noise10 / noise20 / blur5 / blur9 的 mIoU
-（脚本里同时会打印 one-shot 与 loop+VLM 两组对比）

### 4.2 小目标/长尾分析（面积分桶）
```powershell
.\.venv\Scripts\python.exe -m scripts.eval_size_buckets --n 200
```

输出：
- small / medium / large 的 mIoU（one-shot vs loop+VLM）

---

## 5. 效率 / 成本分析

```powershell
.\.venv\Scripts\python.exe -m scripts.eval_cost --n 50
```

输出包含：
- `total_time_sec`、`avg_time_per_img`
- `avg_iters`
- `sam_calls_total / per_img`
- `clip_roi_calls_total / per_img`
- `vlm_calls_total / per_img`

> 峰值显存：CPU 运行可记为 0；如未来切到 CUDA，可在脚本里加入
> `torch.cuda.max_memory_allocated()` 统计。

---

## 6. 置信区间（bootstrap）

对某次评测产出的 `metrics/*.npy` 做 CI：

```powershell
.\.venv\Scripts\python.exe -m scripts.bootstrap_ci --path metrics\ious_loop_vlm1_iter2_n50.npy --B 2000
```

输出：
- mean
- 95% CI

---

## 7. 案例可视化

### 7.1 自动生成一批对比图
```powershell
.\.venv\Scripts\python.exe -m scripts.visualize_cases
```

输出到：`figures/app_refcocom/`

### 7.2 从 IoU 序列挑选“提升/退化/无变化”样本并可视化
```powershell
.\.venv\Scripts\python.exe -m scripts.select_cases_from_metrics
.\.venv\Scripts\python.exe -m scripts.visualize_selected_cases
```

### 7.3 真实图片应用（assets/real → figures/app_real）
```powershell
.\.venv\Scripts\python.exe -m scripts.run_real_images
```

输出：
- `figures/app_real/*_oneshot.png`
- `figures/app_real/*_loop_iter2.png`

---

## 8. 核心设计要点

- **闭环触发**：当 top1 置信度不足或 top1-top2 间隔不足时触发（见 `src/loop_runner.py`）
- **停止条件**：
  - `conf_t >= tau_conf`
  - `gap_t  >= tau_gap`
- **缓存加速策略**：`cache_proposals=True`，每张图只调用一次 SAM propose（见 `src/loop_runner.py`）
- **调用次数统计**：
  - `SamPointProposer.calls`：SAM propose 次数
  - `ClipRanker.roi_calls`：CLIP ROI 计算次数
  - `VLMPolicy.calls`：VLM 调用次数（如已实现）

---

## 9. 常见问题

### 9.1 `SILICONFLOW_API_KEY` 读不到？
PowerShell 推荐：
```powershell
$env:SILICONFLOW_API_KEY="sk-xxxxxx"
```
然后在同一终端运行脚本。

### 9.2 `ModuleNotFoundError: No module named 'src'`
用 module 方式运行脚本，例如：
```powershell
.\.venv\Scripts\python.exe -m scripts.eval_loop_stub
```

---

## 10. 复现论文主表的建议命令

```powershell
# 1) 主线：闭环对比
.\.venv\Scripts\python.exe -m scripts.eval_loop_stub --n 50 --max_iter 1 --use_vlm 1
.\.venv\Scripts\python.exe -m scripts.eval_loop_stub --n 50 --max_iter 2 --use_vlm 1
.\.venv\Scripts\python.exe -m scripts.eval_loop_stub --n 50 --max_iter 3 --use_vlm 1

# 2) 消融：无 VLM
.\.venv\Scripts\python.exe -m scripts.eval_loop_novlm --n 50

# 3) 可靠性（扰动）
.\.venv\Scripts\python.exe -m scripts.eval_reliability --n 50

# 4) 小目标/长尾（分桶）
.\.venv\Scripts\python.exe -m scripts.eval_size_buckets --n 200

# 5) 成本统计
.\.venv\Scripts\python.exe -m scripts.eval_cost --n 50

# 6) 论文插图
.\.venv\Scripts\python.exe -m scripts.visualize_selected_cases
.\.venv\Scripts\python.exe -m scripts.run_real_images
```

---

## License / Acknowledgement

- Segment Anything (SAM) by Meta AI Research
- CLIP / OpenCLIP
- HuggingFace Datasets: moondream/refcoco-m
- SiliconFlow OpenAI-compatible API for VLM
