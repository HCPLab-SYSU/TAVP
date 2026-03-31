<h1 align="center">Learning to See and Act: Task-Aware Virtual View Exploration for Robotic Manipulation</h1>

<p align="center"><strong>Official implementation of TVVE</strong></p>

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
    <img src="https://img.shields.io/badge/Project_Page-1f6feb?logo=google-chrome&logoColor=white&style=flat-square" alt="Project Page" />
  </a>
  <a href="https://arxiv.org/pdf/2508.05186">
    <img src="https://img.shields.io/badge/arXiv-2508.05186-b31b1b?logo=arxiv&logoColor=white&style=flat-square" alt="arXiv" />
  </a>
  <a href="https://huggingface.co/datasets/baiyu858/RLBench-OG">
    <img src="https://img.shields.io/badge/HuggingFace-RLBench--OG-f59e0b?logo=huggingface&logoColor=white&style=flat-square" alt="RLBench-OG Dataset" />
  </a>
  <a href="https://github.com/baiyu858/rlbench-og.git">
    <img src="https://img.shields.io/badge/Benchmark_Code-181717?logo=github&logoColor=white&style=flat-square" alt="Benchmark Code" />
  </a>
</p>

<p align="center">
  <img src="static/images/cvpr2026-accepted-badge.svg" width="320" alt="CVPR 2026 Accepted" />
</p>

<p align="center">
  <img src="static/images/model/overview_v1.png" alt="TVVE overview" width="100%" />
</p>

**TVVE** learns task-aware virtual viewpoints for robotic manipulation. The method combines a **Multi-Viewpoint Exploration Policy (MVEP)** with a **Task-aware Mixture-of-Experts** visual encoder (**TaskMoE**), improving 3D perception, feature discrimination, and cross-domain generalization on **RLBench**, **RLBench-OG**, and **real-world robot setups**.

## <img src="https://raw.githubusercontent.com/microsoft/fluentui-emoji/main/assets/Clipboard/3D/clipboard_3d.png" width="28" alt="Clipboard" /> Table of Contents

- [Highlights](#highlights)
- [Project Status](#project-status)
- [Installation](#installation)
- [Data Preparation](#data-preparation)
- [Training and Evaluation on RLBench](#training-and-evaluation-on-rlbench)
- [RLBench-OG Benchmark](#rlbench-og-benchmark)
- [Repository Layout](#repository-layout)
- [Citation](#citation)
- [Acknowledgements](#acknowledgements)

## <img src="https://raw.githubusercontent.com/microsoft/fluentui-emoji/main/assets/Sparkles/3D/sparkles_3d.png" width="28" alt="Sparkles" /> Highlights

- Accepted by 🔥**CVPR 2026**🔥.
- Official code release for **TVVE**.
- Training scripts for stage 1 and stage 2/3 optimization.
- Evaluation entry points for RLBench and RLBench-OG.
- Public release of the RLBench-OG benchmark dataset and codebase.

## <img src="https://raw.githubusercontent.com/microsoft/fluentui-emoji/main/assets/Clipboard/3D/clipboard_3d.png" width="28" alt="Clipboard" /> Project Status

- [x] Show more simulation results on RLBench and RLBench-OG
- [x] Show more real-world robot results on Dobot and Franka
- [x] Release the model and code
- [x] Release the RLBench-OG benchmark
- [x] Release the RLBench-OG test code

## <img src="https://raw.githubusercontent.com/microsoft/fluentui-emoji/main/assets/Gear/3D/gear_3d.png" width="28" alt="Gear" /> Installation

### 1. Create a Python environment

```bash
conda create -n tvve python=3.8 -y
conda activate tvve
pip install pip==21 setuptools==65.5.0 wheel==0.38.0
```

### 2. Clone the repository

```bash
git clone https://github.com/HCPLab-SYSU/TAVP.git
cd TAVP
```

### 3. Install Python dependencies

```bash
pip install -r requirements.txt
```

### 4. Install CUDA-dependent extras

Skip this step if your CUDA environment is already configured and compatible.

```bash
bash ./cuda_12.3.2_545.23.08_linux.run --silent --toolkit --toolkitpath=$HOME/cuda-12.3
export CUDA_HOME=$HOME/cuda-12.3
```

#### Install PyTorch3D

```bash
export NVCC_FLAGS="--generate-code arch=compute_80,code=sm_80 --generate-code arch=compute_86,code=sm_86 --generate-code arch=compute_87,code=sm_87 --generate-code arch=compute_89,code=sm_89"
pip install git+https://github.com/facebookresearch/pytorch3d.git@stable
```

#### Install xFormers

```bash
pip install ninja
export MAX_JOBS=1
export TORCH_CUDA_ARCH_LIST="8.0;8.6;8.7;8.9"
pip install -v -U git+https://github.com/facebookresearch/xformers.git@main#egg=xformers
```

Adjust the CUDA architectures above to match your GPU.

### 5. Install CoppeliaSim, PyRep, and RLBench

**TVVE** depends on the **RLBench simulation stack**.

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

If you evaluate on a **headless server**, make sure the corresponding **RLBench** and **PyRep** headless rendering requirements are satisfied.

### 6. Install the faster point renderer

This is the recommended renderer used by **TVVE**.

```bash
cd ..
git clone https://github.com/NVlabs/RVT.git
cp -rf RVT/rvt/libs/point-renderer ./point-renderer
rm -rf RVT
cd point-renderer
pip install -e .
cd ../TAVP
```

Then remove the following import from `point-renderer/point_renderer/rvt_renderer.py`:

```python
from mvt.utils import ForkedPdb
```

If you do not want to use the **C++ renderer**, set `render_with_cpp=false` in the config files.

## <img src="https://raw.githubusercontent.com/microsoft/fluentui-emoji/main/assets/Package/3D/package_3d.png" width="28" alt="Package" /> Data Preparation

### RLBench demonstrations

The training scripts expect the compact **RLBench** dataset format released with **ARP**.

```bash
mkdir -p data
cd data
```

Download `datasets/RLBench.tar` from:

- https://rutgers.box.com/s/uzozemx67kje58ycy3lyzf1zgddz8tyq

Then extract it:

```bash
tar xvf RLBench.tar
rm -f RLBench.tar
cd ..
```

After extraction, the default layout is expected to look like:

```text
data/
  train/
  test/
```

### RLBench-OG benchmark data

The **benchmark resources** are released here:

- **Dataset**: https://huggingface.co/datasets/baiyu858/RLBench-OG
- **Code**: https://github.com/baiyu858/rlbench-og.git

## <img src="https://raw.githubusercontent.com/microsoft/fluentui-emoji/main/assets/Rocket/3D/rocket_3d.png" width="28" alt="Rocket" /> Training and Evaluation on RLBench

The provided shell scripts read paths from **environment variables**. If a variable is not set, the scripts fall back to `/path/to/...` placeholders.

### 1. Set runtime paths

```bash
export TRAIN_DEMO_FOLDER=/path/to/rlbench/data/train
export EVAL_DATA_FOLDER=/path/to/rlbench/data/test
export STAGE1_INIT_WEIGHTS=/path/to/stage1_checkpoint.pth
export STAGE23_ONLY_IL_WEIGHTS=/path/to/stage23_only_il_checkpoint.pth
export STAGE23_EVAL_WEIGHTS=/path/to/stage23_eval_checkpoint.pth
```

Depending on your workflow, you may also want to set:

```bash
export STAGE23_JOINT_INIT_WEIGHTS=/path/to/stage23_joint_checkpoint.pth
export STAGE23_PPO_IL_WEIGHTS=/path/to/stage23_ppo_il_checkpoint.pth
```

### 2. Run training or evaluation

| Command | Purpose |
| --- | --- |
| `./start_tvve_stage1.sh` | Stage 1 training |
| `./start_tvve_stage23.sh a 1 op` | Stage 2 training with PPO-only updates |
| `./start_tvve_stage23.sh a 1 oi` | Stage 3 training with IL-only updates |
| `./start_tvve_stage23.sh b 1 oi` | RLBench evaluation |

### 3. Notes

- `a` starts **training** and `b` starts **evaluation** in `start_tvve_stage23.sh`.
- Evaluation uses **`xvfb-run`**; install the required **headless rendering dependencies** first.
- Logs are written to `logs/`, and **Hydra outputs** are written to `outputs/`.

## <img src="https://raw.githubusercontent.com/microsoft/fluentui-emoji/main/assets/Rocket/3D/rocket_3d.png" width="28" alt="Rocket" /> RLBench-OG Benchmark

### 1. Install the RLBench-OG environment

```bash
mkdir -p env
cd env
git clone https://github.com/baiyu858/rlbench-og.git
cd rlbench-og
pip install -r requirements.txt
pip install -e .
cd ../..
```

### 2. Download and extract the dataset

```bash
huggingface-cli download baiyu858/RLBench-OG \
    --repo-type dataset \
    --local-dir ./data/RLBench-OG

cd data/RLBench-OG
tar -xvf Occlusion.tar.xz
rm -f Occlusion.tar.xz

cd Generalization
tar -xvf train.tar.xz
rm -f train.tar.xz
tar -xvf test.tar.xz
rm -f test.tar.xz

cd ../../..
```

### 3. Occlusion Suite

Use the same **training** and **evaluation** commands as **RLBench** after switching the dataset paths and task lists.

<details>
<summary>Occlusion1 split</summary>

Update `env.tasks` in both `configs/tvve_stage1.yaml` and `configs/tvve_stage23.yaml`:

```yaml
[
  "basketball_in_hoop_occlusion",
  "scoop_with_spatula_occlusion",
  "take_plate_off_colored_dish_rack_occlusion",
  "water_plants_occlusion",
  "block_pyramid_occlusion",
  "solve_puzzle_occlusion",
  "take_usb_out_of_computer_occlusion",
  "close_drawer_occlusion",
  "straighten_rope_occlusion",
  "toilet_seat_down_occlusion"
]
```

Set:

```yaml
train.demo_folder: ./data/RLBench-OG/Occlusion/train
eval.datafolder: ./data/RLBench-OG/Occlusion/test
```

</details>

<details>
<summary>Occlusion2 split</summary>

Update `env.tasks` in both `configs/tvve_stage1.yaml` and `configs/tvve_stage23.yaml`:

```yaml
[
  "basketball_in_hoop",
  "scoop_with_spatula",
  "take_plate_off_colored_dish_rack",
  "water_plants",
  "block_pyramid",
  "solve_puzzle",
  "take_usb_out_of_computer",
  "close_drawer_occlusion",
  "straighten_rope",
  "toilet_seat_down"
]
```

Set:

```yaml
train.demo_folder: ./data/RLBench-OG/Generalization/train
eval.datafolder: ./data/RLBench-OG/Occlusion/test
```

</details>

Then run the same commands from [Training and Evaluation on RLBench](#training-and-evaluation-on-rlbench).

### 4. Generalization Suite

You can directly evaluate with the **`STAGE23_EVAL_WEIGHTS`** checkpoint trained for **Occlusion2**.

First install the local **YARR** and **PerAct** packages:

```bash
cd env/YARR
pip install -e .

cd ../peract
pip install -e .

cd ../..
```

Then edit the placeholders in **`eval_og.sh`**:

- `epoch=<select_which_epoch_to_eval>`
- `model_folder=<path_to_STAGE23_EVAL_WEIGHTS_directory>`

Finally run:

```bash
bash ./eval_og.sh
```

## <img src="https://raw.githubusercontent.com/microsoft/fluentui-emoji/main/assets/Package/3D/package_3d.png" width="28" alt="Package" /> Repository Layout

```text
TAVP/
  configs/                experiment configuration files
  env/                    RLBench-OG, PerAct, and YARR integrations
  static/                 website images and videos
  train_tvve_stage1.py    stage 1 training entry
  train_tvve_stage23.py   stage 2/3 training entry
  eval.py                 RLBench evaluation entry
  eval_og.py              RLBench-OG evaluation entry
  start_tvve_stage1.sh    stage 1 launch script
  start_tvve_stage23.sh   stage 2/3 launch script
  eval_og.sh              RLBench-OG batch evaluation script
```

## <img src="https://raw.githubusercontent.com/microsoft/fluentui-emoji/main/assets/Books/3D/books_3d.png" width="28" alt="Books" /> Citation

If you find **TVVE** useful in your research, please cite:

```bibtex
@InProceedings{Bai_2026_CVPR,
  author    = {Bai, Yongjie and Wang, Zhouxia and Liu, Yang and Luo, Kaijun and Wen, Yifan and Dai, Mingtong and Chen, Weixing and Chen, Ziliang and Liu, Lingbo and Li, Guanbin and Lin, Liang},
  title     = {Learning to See and Act: Task-Aware Virtual View Exploration for Robotic Manipulation},
  booktitle = {Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR)},
  month     = {June},
  year      = {2026}
}
```

## <img src="https://raw.githubusercontent.com/microsoft/fluentui-emoji/main/assets/Sparkles/3D/sparkles_3d.png" width="28" alt="Sparkles" /> Acknowledgements

This repository builds on several excellent open-source projects, including RLBench, PyRep, RVT, PerAct, and YARR. Please also cite their original work if you use this codebase in your research.
