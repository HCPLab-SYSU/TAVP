<h1 align="center">Learning to See and Act: Task-Aware Virtual View Exploration for Robotic Manipulation</h1>

<p align="center">
  <img src="static/images/cvpr2026-accepted-badge.svg" width="320" alt="CVPR 2026 Accepted" />
</p>

<p align="center">
  <a href="https://scholar.google.com/citations?user=W_YDucAAAAAJ&hl=zh-CN">Yongjie Bai</a>,
  <a href="https://wzhouxiff.github.io/" target="_blank">Zhouxia Wang</a>,
  <a href="https://yangliu9208.github.io/" target="_blank">Yang Liu</a>,
  Kaijun Luo,
  Yifan Wen,
  <a href="https://github.com/CCCalcifer/" target="_blank">Mingtong Dai</a>,
  <a href="https://wissingchen.github.io/" target="_blank">Weixing Chen</a>,
  <a href="https://scholar.google.com/citations?user=RC-LN4QAAAAJ&hl=en" target="_blank">Ziliang Chen</a>,
  <a href="https://lingboliu.com/" target="_blank">Lingbo Liu</a>,
  <a href="https://guanbinli.com/" target="_blank">Guanbin Li</a>,
  <a href="http://www.linliang.net/" target="_blank">Liang Lin</a>
</p>

<p align="center">
  <a href="https://hcplab-sysu.github.io/TAVP/">
    <img src="https://img.shields.io/badge/Website-1f6feb?logo=google-chrome&logoColor=white&style=flat-square" alt="Website" />
  </a>
  <a href="https://arxiv.org/pdf/2508.05186">
    <img src="https://img.shields.io/badge/arXiv-2508.05186-b31b1b?logo=arxiv&logoColor=white&style=flat-square" alt="arXiv" />
  </a>
  <a href="https://huggingface.co/datasets/baiyu858/RLBench-OG">
    <img src="https://img.shields.io/badge/HuggingFace-RLBench--OG-facc15?logo=huggingface&logoColor=black&style=flat-square" alt="RLBench-OG" />
  </a>
</p>

<p align="center">
  <img src="static/images/model/overview_v1.png" alt="TVVE overview" width="100%" />
</p>

TVVE learns to actively explore task-relevant virtual viewpoints for robotic manipulation. It combines a Multi-Viewpoint Exploration Policy (MVEP) with a Task-aware Mixture-of-Experts visual encoder (TaskMoE), producing stronger 3D perception, more discriminative visual features, and better generalization on RLBench, RLBench-OG, and real-world robot settings.

## <img src="https://raw.githubusercontent.com/microsoft/fluentui-emoji/main/assets/Sparkles/3D/sparkles_3d.png" width="28" alt="Sparkles" /> Highlights

- Accepted by **CVPR 2026**.
- Official repository for **TVVE**.
- Includes the training scripts, and evaluation entry points.

## <img src="https://raw.githubusercontent.com/microsoft/fluentui-emoji/main/assets/Gear/3D/gear_3d.png" width="28" alt="Gear" /> Environment Setup

### 1. Create a Python 3.8 conda environment

```bash
conda create -n tvve python=3.8 -y
conda activate tvve
pip install pip==21 setuptools==65.5.0 wheel==0.38.0
```

### 2. Clone this repository

```bash
git clone https://github.com/HCPLab-SYSU/TAVP.git
cd TAVP
```

### 3. Install the Python packages from `requirements.txt`

```bash
pip install -r requirements.txt
```

### 4. Install CUDA 12 if needed

Skip this step if CUDA 12 is already installed and `CUDA_HOME` is set correctly.

```bash
bash ./cuda_12.3.2_545.23.08_linux.run --silent --toolkit --toolkitpath=$HOME/cuda-12.3
export CUDA_HOME=$HOME/cuda-12.3
```

### 5. Install extra dependencies used by TVVE

#### 5.1 Install pytorch3d

```bash
export NVCC_FLAGS="--generate-code arch=compute_80,code=sm_80 --generate-code arch=compute_86,code=sm_86 --generate-code arch=compute_87,code=sm_87 --generate-code arch=compute_89,code=sm_89"
pip install git+https://github.com/facebookresearch/pytorch3d.git@stable
```
#### 5.2 Install xformers

```bash
pip install ninja
export MAX_JOBS=1
export TORCH_CUDA_ARCH_LIST="8.0;8.6;8.7;8.9"
pip install -v -U git+https://github.com/facebookresearch/xformers.git@main#egg=xformers
```

Adjust the CUDA architectures above to match your GPU.

### 6. Install CoppeliaSim, PyRep, and RLBench

TVVE depends on the RLBench stack.

```bash
cd ..
wget https://downloads.coppeliarobotics.com/V4_1_0/CoppeliaSim_Edu_V4_1_0_Ubuntu20_04.tar.xz
tar -xf CoppeliaSim_Edu_V4_1_0_Ubuntu20_04.tar.xz
mv CoppeliaSim_Edu_V4_1_0_Ubuntu20_04 CoppeliaSim
export COPPELIASIM_ROOT=$PWD/CoppeliaSim
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:$COPPELIASIM_ROOT
export QT_QPA_PLATFORM_PLUGIN_PATH=$COPPELIASIM_ROOT
```

```bash
git clone https://github.com/stepjam/PyRep.git
cd PyRep
pip install -r requirements.txt
pip install -e .
cd ..
```

```bash
git clone https://github.com/mlzxy/RLBench.arp.git
cd RLBench.arp
pip install -r requirements.txt
python setup.py develop
cd ../TAVP
```

If you are running on a headless server, also check the headless rendering notes in RLBench / PyRep before evaluation.

### 7. Install Faster Point-Renderer

This is the recommended renderer used by TVVE.

```bash
cd ..
git clone https://github.com/NVlabs/RVT.git
cp -rf RVT/rvt/libs/point-renderer ./point-renderer
rm -rf RVT
cd point-renderer
pip install -e .
cd ../TAVP
```

Then open `point-renderer/point_renderer/rvt_renderer.py` and remove the line:

```python
from mvt.utils import ForkedPdb
```

If you do not want to use the Faster Point-Renderer, set `render_with_cpp=False` in the TVVE config files.

## <img src="https://raw.githubusercontent.com/microsoft/fluentui-emoji/main/assets/Package/3D/package_3d.png" width="28" alt="Package" /> Dataset Download

### RLBench train / test demonstrations

The training scripts expect the compact RLBench dataset format released with ARP.

```bash
mkdir -p data
cd data
```

Download `datasets/RLBench.tar` from the ARP Box folder:

- https://rutgers.box.com/s/uzozemx67kje58ycy3lyzf1zgddz8tyq

Then extract it:

```bash
tar xvf RLBench.tar
rm -f RLBench.tar
cd ..
```

After extraction, you should have paths such as:

```text
data/train
data/test
```

### RLBench-OG Benchmark

The OOD benchmark released for this project is available at:

- [RLBench-OG Datasets](https://huggingface.co/datasets/baiyu858/RLBench-OG)
- [RLBench-OG Code](https://github.com/baiyu858/rlbench-og.git)

### Important path note

The provided shell scripts currently use hard-coded local paths for datasets and checkpoints. Before running training or evaluation, update the following fields to match your machine:

- `train.demo_folder` in `start_tvve_stage1.sh`
- `train.demo_folder` in `start_tvve_stage23.sh`
- `eval.datafolder` in `start_tvve_stage23.sh`
- `weights` / `ckpt` in `start_tvve_stage23.sh`

## <img src="https://raw.githubusercontent.com/microsoft/fluentui-emoji/main/assets/Rocket/3D/rocket_3d.png" width="28" alt="Rocket" /> Quick Start

### Stage 1 training

```bash
./start_tvve_stage1.sh
```

### Stage 2 training: PPO-only branch

In the script logic, this corresponds to `BRANCH=a` and `only_il_or_only_ppo=op`.

```bash
./start_tvve_stage23.sh a 1 op
```

### Stage 3 training: IL-only branch

In the script logic, this corresponds to `BRANCH=a` and `only_il_or_only_ppo=oi`.

```bash
./start_tvve_stage23.sh a 1 oi
```

### Evaluation

In the script logic, this corresponds to `BRANCH=b`.

```bash
./start_tvve_stage23.sh b 1 oi
```

### Notes before launching

- `start_tvve_stage23.sh` uses fixed checkpoint paths. Replace them with the checkpoints generated on your machine.
- Update the default RLBench data path in the scripts before training.
- Evaluation is launched with `xvfb-run`, so make sure the relevant headless rendering dependencies are installed.

## <img src="https://raw.githubusercontent.com/microsoft/fluentui-emoji/main/assets/Clipboard/3D/clipboard_3d.png" width="28" alt="Clipboard" /> Project Status

- [x] Show more simulation results on RLBench and RLBench-OG
- [x] Show more real-world robot results on Dobot and Franka
- [x] Release the model and code
- [x] Release the RLBench-OG benchmark
- [ ] Release the RLBench-OG test code

## <img src="https://raw.githubusercontent.com/microsoft/fluentui-emoji/main/assets/Books/3D/books_3d.png" width="28" alt="Books" /> Citation

If you find TVVE useful for your research, please cite:

```bibtex
@InProceedings{Bai_2026_CVPR,
  author    = {Bai, Yongjie and Wang, Zhouxia and Liu, Yang and Luo, Kaijun and Wen, Yifan and Dai, Mingtong and Chen, Weixing and Chen, Ziliang and Liu, Lingbo and Li, Guanbin and Lin, Liang},
  title     = {Learning to See and Act: Task-Aware Virtual View Exploration for Robotic Manipulation},
  booktitle = {Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR)},
  month     = {June},
  year      = {2026}
}
```
