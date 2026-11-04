<div align="center">

<img src="docs/assets/banner.webp" alt="NextTransformer-V：NextTransformer 的多模态赛道" width="100%">

<h3>算力对齐的视觉-语言骨干网络试验台，<br>建立在 NextTransformer 协议之上</h3>

<p>
<a href="#结果"><img src="docs/assets/button-results.webp" alt="查看结果" height="60"></a>
<a href="docs/rfc-multimodal-track.md"><img src="docs/assets/button-rfc.webp" alt="阅读 RFC" height="60"></a>
<a href="#复现"><img src="docs/assets/button-reproduce.webp" alt="复现" height="60"></a>
</p>

<p>
<a href="https://github.com/Interactive-Active-AI/NextTransformer"><img alt="Built on NextTransformer" src="https://img.shields.io/badge/built%20on-NextTransformer-8B5CF6?style=flat-square"></a>
<a href="docs/spatial-preregistration.md"><img alt="Pre-registered" src="https://img.shields.io/badge/study-pre--registered-2A78D6?style=flat-square"></a>
<a href="results/compute-audit.txt"><img alt="Compute matched within 1%" src="https://img.shields.io/badge/compute-matched%20within%201%25-1BAF7A?style=flat-square"></a>
<a href="LICENSE"><img alt="License: MIT" src="https://img.shields.io/badge/license-MIT-F2A93B?style=flat-square"></a>
<img alt="Python 3.10-3.13" src="https://img.shields.io/badge/python-3.10--3.13-3776AB?style=flat-square&logo=python&logoColor=white">
<img alt="PyTorch 2.9.1" src="https://img.shields.io/badge/PyTorch-2.9.1-EE4C2C?style=flat-square&logo=pytorch&logoColor=white">
</p>

<a href="README.md">English</a> · <b>简体中文</b>

</div>

<br>

**当每条序列里都有一部分是图像时，骨干网络该怎样花一份固定的训练预算？** NextTransformer-V 沿用
[NextTransformer](https://github.com/Interactive-Active-AI/NextTransformer) 的约定：
- 同样的数据，同样的顺序；
- 用同一套算子账本，把每个 token 的训练 FLOPs 对齐到 1% 以内；
- 同样的评估时间点。

它把冻结视觉编码器输出的 token 和文本交错排进 8,192 长度的上下文，NextTransformer 的每种方法都不用改动就能直接训练。
第一项研究问的是：混合骨干从哪里得到图像的二维布局，以及让它的线性混合层感知二维结构有没有用。这项研究的判定规则在开跑前就已写定。

## 动态

- <img src="https://img.shields.io/badge/NEW-E5607E?style=flat-square" alt="New" height="18"> **2026-11-04** · 第一项研究：混合骨干里图像 token 的二维位置。包含 9 个算力对齐、使用冻结 SigLIP2 token 的 run，两对换种子的 run，以及预注册的判定。
- <img src="https://img.shields.io/badge/RUNNING-8B5CF6?style=flat-square" alt="Running" height="18"> **2026-11-04** · 原始像素复现：输入里没有编码器提供的位置信息，位置只能由骨干自己提供。
- <img src="https://img.shields.io/badge/QUEUED-7A8194?style=flat-square" alt="Queued" height="18"> **2026-11-05** · 8,192 上下文下的骨干对比（GDN、QSA 混合、mHC）。

## 亮点

<img src="docs/assets/card-track.webp" alt="图像 token，同一套约定" width="100%">

<img src="docs/assets/card-test.webp" alt="打乱 token 顺序，内容不变" width="100%">

<img src="docs/assets/card-finding.webp" alt="二维混合改变的是读法，不是准确率" width="100%">

## 结果

所有数字都来自 [`results/`](results/) 里的文件。[`results/tables.md`](results/tables.md) 里的表格和本页的图，都由 `python -m ntv.analysis` 从这些文件生成。

**设置。** 每个 run 的预算相同：
- 5,000 步，每步 128 行 × 8,192 token（共 52.4 亿 token）；
- 每行 12 张图；
- 每 token 的训练 FLOPs 与 Dense 锚点相差不到 1%。

模型参数量为 14–16 亿。各 run 只在三个方面不同：

| | |
|---|---|
| **骨干** | Dense，或 GDN 混合骨干（每 4 层中有 3 层是 Gated DeltaNet） |
| **注意力位置编码** | 展平序列上的 RoPE-1D、不加位置编码（NoPE），或 RoPE-2D |
| **二维短卷积** | 加或不加：在图像位置上，用每张图 token 网格上的 3×3 depthwise 卷积替换线性层原有的一维短卷积 |

图像 token 是冻结的 SigLIP2 特征（so400m/16，384 像素，2×2 合并成 12×12 个 token），按行优先顺序输入。有两个配置用第二个种子各重跑了一次，用来估计噪声。

### 1 · 按预注册规则，二维卷积判为无效

| 二维卷积对比的基线 | 垂直 acc2 提升 | caption BPB 变化 | 文本 BPB 变化 | 判定 |
|---|---:|---:|---:|---|
| NoPE 混合骨干 | +0.1 个百分点 | +0.0025 | +0.0001 | 无效 |
| RoPE-1D 混合骨干 | +0.1 个百分点 | +0.0022 | -0.0002 | 无效 |

规则要求提升至少 5 个百分点，并且超过种子间波动（0.1 个百分点）的 2 倍。这一条达不到，因为主指标已经饱和：每个 run 在合成探针上的垂直 acc2 都在 99.9–100%。原因有两个：
- 训练数据里本来就有和探针布局相同的合成图。
- SigLIP2 token 本身带着位置信息：用岭回归从 token 特征读它的行和列，R² 分别有 0.86 / 0.89；原始像素 patch 只有 0.03 / 0.00。

### 2 · 不同骨干从不同来源读位置

在更难的探针上（网格更细、物体更小、干扰物更多）重新打分，并把图像 token 送进骨干的顺序打乱，内容保持不变：
- 如果骨干是从 token 顺序读水平位置，把每一行的 token 倒过来，它对左右关系的判断就会反转。
- 把 token 的行倒过来，同理可以检验垂直位置。

<img src="docs/figures/token-order.svg" alt="每种骨干从哪里读二维位置" width="100%">

| run | 每行内倒序后的水平准确率 | 行倒序后的垂直准确率 |
|---|---:|---:|
| Dense，RoPE-1D | 80.5% | 31.9% |
| Dense，NoPE | 82.0% | 51.7% |
| 混合，RoPE-1D（种子 42 / 43） | 27.7% / 30.8% | 58.3% / 65.5% |
| 混合，RoPE-2D | 36.9% | 61.2% |
| 混合，NoPE | 19.0% | 87.7% |
| 混合，NoPE + 二维卷积（种子 42 / 43） | 19.5% / 22.1% | 28.6% / 20.9% |
| 混合，RoPE-1D + 二维卷积 | 20.0% | 21.0% |

用两对换种子的 run 测出的波动是：第一列 3.1 个百分点，第二列 7.7 个百分点。下面这些差别都超过了波动的 2 倍：
- **GDN 的循环结构从 token 顺序读水平位置。** 每行倒序后，混合骨干只剩 19–37%，Dense 仍有 80–82%。
- **不加位置编码的混合骨干从编码器特征读垂直位置。** 行倒序后仍有 87.7%。
- **二维卷积在两种位置编码下，都让垂直位置改从 token 网格读。** 行倒序后，垂直准确率在 RoPE-1D 下降 37 个百分点，在 NoPE 下降 59 个百分点。

相比之下，RoPE-2D 没有改变混合骨干读垂直位置的方式：行倒序后它是 61.2%，RoPE-1D 是 58.3%，差别在波动以内。

加了二维卷积的模型，从第一个 checkpoint 起就从 token 网格读垂直位置；不加的 NoPE 混合骨干，则随着训练越来越依赖编码器特征。

<img src="docs/figures/token-order-curves.svg" alt="训练过程中，行倒序后的垂直准确率" width="100%">

### 3 · 准确率和 loss 没有超出种子波动的变化

- **难版探针：** 每个 run 在两个方向上的准确率都是 88–96%。没有任何一组对比超过波动的 2 倍（垂直 2.9、水平 2.6 个百分点）。
- **COCO：** 每个 run 的 caption BPB 都在 0.84–0.86，5 选 1 检索都在 98%。
- **What'sUp：** 在这个规模下约 50%，接近随机。

<img src="docs/figures/accuracy.svg" alt="难版探针上的关系准确率" width="100%">

> **一句话：** 编码器本身带位置信息时，二维机制改变的是骨干*怎么*读位置，而不是读得*准不准*。原始像素 run 还在跑，它检验的是编码器不提供位置的情形。

## 赛道设计

| | |
|---|---|
| **视觉 token** | 冻结的 SigLIP2，每张图 144 个 token，经过两层线性投影。<br>图像通过 token embedding 和 value embedding 上的前向 hook 输入，因此 NextTransformer 的每种方法都不需要改动；没有图像时，模型与 NextTransformer 逐位一致。<br>另有原始像素输入（32 像素 patch，不用编码器）。 |
| **数据** | CC12M、CC3M 图文对，加上合成空间关系图，和 ClimbMix 文本一起打包成 8,193 长度的行。<br>每行 12 个 `[BOS] [image] caption` 文档，打包过程完全确定。<br>文本评估用 dev shard（06541），没有使用验证 shard（06542）和 CORE。 |
| **算力** | 多模态模型的算力用 NextTransformer 的算子账本核算。<br>每个骨干都用 v2 网格规则校准到 8,192 上下文下 Dense 锚点（每 token 5.9956 GFLOPs）的 1% 以内（[`configs/track`](configs/track)）。<br>[`scripts/audit_variants.py`](scripts/audit_variants.py) 逐一核对了每个实验变体，最大偏差 0.29%（[`results/compute-audit.txt`](results/compute-audit.txt)）。 |
| **评估** | caption 与文本 BPB；COCO 5k caption BPB 和基于似然的检索；合成探针、难版探针与打乱 token 顺序的测试；What'sUp。 |

[RFC](docs/rfc-multimodal-track.md) 列出了原型在 NextTransformer 上需要绕开的地方，这些都可以做成上游接口。给算力账本的补丁已经写好并测试过，放在 [`upstream/`](upstream/)。

## 复现

Python 3.10（见 `.python-version`），使用 [uv](https://docs.astral.sh/uv/)。NextTransformer 以子模块形式锁定在本项目所基于的那个 commit，它的文件没有任何改动。下面所有输出都写在 `$NTV_ROOT` 下。

```bash
git clone --recursive https://github.com/Enderfga/NextTransformer-V.git && cd NextTransformer-V
export NTV_ROOT=<一块足够大的本地盘>
bash scripts/bootstrap.sh                                   # 环境（.venv），torch 2.9.1 + fla
```

<details>
<summary><b>数据：</b>锁定版本的下载、合成探针、打包</summary>

```bash
# 锁定版本的输入数据
bash scripts/download.sh climbmix 128 && bash scripts/download.sh siglip2
bash scripts/download.sh cc12m 0 995 && bash scripts/download.sh cc3m 0 575
bash scripts/download.sh coco5k && bash scripts/download.sh whatsup
python -m ntv.prepare_text --data-dir $NTV_ROOT/data/climbmix --resources $NTV_ROOT/resources   # 锁定的 tokenizer
python -m ntv.whatsup --root $NTV_ROOT/data/whatsup

# 合成空间关系图：训练分片、探针、难版探针
python -m ntv.synth train --out $NTV_ROOT/data/synth --shards 120 --per-shard 5000 --seed 1
python -m ntv.synth eval --out $NTV_ROOT/data/synth-eval --count 2000 --seed 2
python -m ntv.synth eval --out $NTV_ROOT/data/synth-hard --count 2000 --seed 3 --grid 8 --radius 12,18 \
    --distractors 3,7 --jitter 6

# 打包（训练 640,000 行；验证 256 行，取自 dev shard）
python -m ntv.pack --tar-root $NTV_ROOT/data --source cc12m/cc12m-train:0:991 --source cc3m/cc3m-train:0:575 \
    --source synth/synth-train:0:119 --text $NTV_ROOT/data/climbmix --text-shards 0 79 \
    --resources $NTV_ROOT/resources --rows 640000 --out $NTV_ROOT/packed/pilot-train
python -m ntv.pack --tar-root $NTV_ROOT/data --source cc12m/cc12m-train:992:995 --text $NTV_ROOT/data/climbmix \
    --text-shards 6541 --resources $NTV_ROOT/resources --rows 256 --out $NTV_ROOT/packed/pilot-val
```

</details>

<details>
<summary><b>训练与报告：</b>训练、评估、应用预注册判定</summary>

```bash
# 在一台 8 卡机器上依次训练并评估所有 run（H200 上每个约 4.5 小时）
bash scripts/pilot.sh configs/pilot/jobs.txt

# 报告、预注册判定、表格和图
python scripts/export_results.py $NTV_ROOT/runs results    # 去掉站点路径的结果摘要
python -m ntv.analysis results --out report.html --json results/decision.json --markdown results/tables.md \
    --figures docs/figures
```

校准结果已在 [`configs/track`](configs/track) 里。要重新校准，运行
`python -m ntv.calibrate --methods dense_full gdn gdn_hybrid nsa qsa qsa_hybrid mhc --out configs/track`。
视觉 token 的线性位置探针是 `python -m ntv.position_probe`。

</details>

## 仓库结构

| 路径 | 内容 |
|---|---|
| `ntv/model.py`、`ntv/spatial.py` | 给任意 NextTransformer 方法接入视觉输入；patch 顺序、RoPE-2D、二维短卷积 |
| `ntv/encoder.py`、`ntv/data.py`、`ntv/pack.py` | 冻结编码器、打包好的行、确定性打包 |
| `ntv/cost.py`、`ntv/calibrate.py` | 用 NextTransformer 的账本做算力核算与校准 |
| `ntv/train.py` | 训练器（沿用 NextTransformer 的优化器、学习率计划和 checkpoint 格式） |
| `ntv/evaluate.py`、`ntv/spatial_eval.py`、`ntv/synth.py`、`ntv/whatsup.py`、`ntv/position_probe.py` | 各项评估 |
| `ntv/analysis.py` | 预注册判定、报告、表格和图 |
| `configs/` | 校准好的骨干配置、实验清单 |
| `docs/` | 预注册、附录、RFC；README 的图和素材 |
| `results/` | 每个已完成 run 的结果摘要（指标和评估文件）、算力核算 |
| `upstream/` | 给 NextTransformer 的补丁 |

## 许可证

MIT，与 NextTransformer 相同（[LICENSE](LICENSE)）。横幅背景是生成的水彩插画；其余图和卡片都由结果文件绘制（[`docs/assets/src`](docs/assets/src)）。
