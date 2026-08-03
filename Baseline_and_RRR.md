# BrainCog BDM-SNN baseline reproduction

## Source and environment

- Source: `https://github.com/BrainCog-X/Brain-Cog.git`
- Checkout: `f9b879f75da2247a9f0c31864a947f0e5d2f3dab` (2025-11-06)
- Conda environment: `lph`, Python 3.10.4
- GPU run: NVIDIA GeForce RTX 4090 on physical CUDA device 1
- PyTorch: `2.1.2+cu121`, CUDA runtime 12.1

The repository was installed editable in `lph`. Its package-level imports load
more than the BDM-SNN example itself needs, so the requirements listed by the
repository were installed in `lph` as well. `python -m pip check` completed
without broken requirements.

## Baseline task

The repository's BDM-SNN README identifies `BDM-SNN.py` as an LIF decision SNN
for Flappy Bird and describes the UAV entry point as a hardware application
whose reinforcement-learning task must be supplied by the user. The Flappy
Bird task is therefore the reproducible software-only baseline.

The BDM code originally stored modules in Python lists, preventing PyTorch's
`model.to(device)` from moving BDM connections and nodes. This checkout adds
device registration for those modules and structural masks, then provides an
optional GPU selection in the Flappy Bird entry point. The BDM topology, LIF
dynamics, STDP rules, reward rule, and game logic are unchanged.

`BDMSNN_VERBOSE=1` restores the original per-step activity/action logging.
`BDMSNN_MAX_FRAMES` defaults to the original 30000-frame cap; it is only a
test-duration override. `BDMSNN_SEED` makes Python, NumPy, and PyTorch initial
randomness explicit.

## Verified CUDA run

Run from `Brain-Cog/examples/decision_making/BDM-SNN`:

```sh
source /opt/anaconda3/etc/profile.d/conda.sh
conda activate lph
CUDA_VISIBLE_DEVICES=1 SDL_VIDEODRIVER=dummy SDL_AUDIODRIVER=dummy \
BDMSNN_DEVICE=cuda:0 BDMSNN_SEED=0 BDMSNN_MAX_FRAMES=600 \
python BDM-SNN.py
```

`CUDA_VISIBLE_DEVICES=1` exposes physical GPU 1 as process-local `cuda:0`.
The dummy SDL driver suppresses the window only; it does not alter the game
state or neural computation.

Observed result on 2026-07-19:

| Metric | Result |
| --- | ---: |
| Simulated game frames | 600 |
| Reward updates | 300 |
| Final score / maximum score | 4 / 4 |
| Cumulative reward | 1662 |
| Reward histogram | `+6: 267`, `+3: 28`, `-3: 2`, `-5: 2`, `-8: 1` |
| Wall-clock runtime | 37.66 s |

The saved outputs are `Brain-Cog/examples/decision_making/BDM-SNN/lif_reward_l.npy`
and `Brain-Cog/examples/decision_making/BDM-SNN/lif_score_l.npy`.

This validates basic GPU execution, state-to-action decisions, reward-modulated
STDP updates, and progress through four pipes. It does not independently
establish the README's stronger claim that the agent stably passes the game on
its first try; that would require several full-length fixed-seed evaluations.

## UAV status

`BDM-SNN-UAV.py` instantiates and takes off a physical DJI/RoboMaster drone,
uses placeholder state and reward definitions, and defines two actions while
retaining branches for actions 0 through 3. There is no checked-in simulator or
complete RL task. It was intentionally not run, so this baseline makes no UAV
reproduction claim.

---

# 中文补充：BDM-SNN 实验与算法详解

## 1. 文档范围与证据边界

本节说明三个不同层次的内容，三者不能混为一谈：

1. **论文模型**：Zhao、Zeng 和 Xu 在 2018 年论文中提出的脑启发决策脉冲神经网络（BDM-SNN）。论文在 DJI MATRICE 100 实机上讨论了穿窗与避障任务。
2. **BrainCog 工程实现**：本目录的 `Brain-Cog/examples/decision_making/BDM-SNN/`。该实现包含用于 Flappy Bird 的 LIF/IF 版本、简化 HH 版本，以及一个不完整的真实无人机控制脚本；仓库 README 明确说明其奖励调制 STDP 相比论文有所改进。
3. **本次复现实验**：当前目录运行的 `BDM-SNN.py`，任务是经过几何状态离散化后的 Flappy Bird，不是论文中的相机输入 DJI 飞行器实验。

因此，下面对论文结果均使用“论文报告”表述；对已执行的内容仅使用“本次复现观察到”表述。不能据此声称已经复现论文的实机 UAV 结果。

## 2. 问题定义与总体闭环

BDM-SNN 将在线决策写成如下闭环：

```text
环境观测 -> 状态编码 -> DLPFC/皮层-基底节-丘脑环路 -> PM 动作选择
    ^                                                        |
    |                                                        v
下一状态与环境反馈 <- 执行动作 <- 奖励/预测误差 <- 在线突触更新
```

设离散状态数为 `Ns`、动作数为 `Na`。状态由 DLPFC 的 `Ns` 个神经元表示；每个动作对应 PM（premotor cortex，前运动皮层）的一个神经元。纹状体直接通路 StrD1 与间接通路 StrD2 各自拥有 `Ns * Na` 个状态-动作神经元，从而把“当前状态 + 候选动作”作为一个可塑的表征单元。

该模型不是端到端视觉深度强化学习：论文先将相机图像预处理并分类为离散状态；当前 Flappy Bird 代码更直接地从游戏对象的相对几何关系计算状态。图像资源只用于渲染与像素级碰撞检测，网络没有读取原始 RGB 图像，也没有卷积视觉编码器。

## 3. 论文中的脑启发 BDM-SNN

### 3.1 模块、通路与动作选择

论文的完整概念模型有 11 个模块：DLPFC、PM、丘脑、StrD1、StrD2、STN、GPe、GPi/SNr、SNc/VTA、内侧眶额皮层 MOFC、外侧眶额皮层 LOFC。

| 生物启发模块 | 论文中的抽象功能 |
| --- | --- |
| DLPFC | 表示当前离散状态，并向基底节、丘脑和 PM 提供兴奋性输入 |
| StrD1（直接通路，“Go”） | 抑制 GPi，使丘脑去抑制，倾向于放行对应动作 |
| StrD2（间接通路，“No Go”） | 经 GPe 增强对 GPi/丘脑的抑制效果，压制对应动作 |
| STN（超直接通路） | 接收 DLPFC 输入并兴奋 GPe、GPi，参与快速竞争/抑制调节 |
| GPe、GPi/SNr | 基底节中继与输出核；GPi/SNr 对丘脑施加抑制 |
| 丘脑 | 接收 GPi/SNr 抑制、DLPFC 兴奋后投射至 PM |
| PM | 动作执行层；论文用横向抑制实现竞争性 winner-takes-all（WTA） |
| SNc/VTA、MOFC、LOFC | 分别抽象正/负奖惩相关多巴胺及其眶额皮层调节作用 |

论文使用三条路径共同决定动作：直接通路 `DLPFC -> StrD1 -> GPi -> thalamus -> PM`，间接通路 `DLPFC -> StrD2 -> GPe -> GPi -> thalamus -> PM`，以及超直接通路 `DLPFC -> STN -> {GPe, GPi}`。PM 中最早发放脉冲的动作神经元胜出，属于首脉冲时间/延迟编码式的竞争选择。

### 3.2 论文的神经元与 STDP

论文使用规则放电（RS）参数的 Izhikevich 神经元，而不是当前默认脚本的 IF 神经元。其基本形式为：

\[
\dot v = 0.04v^2 + 5v + 140 - u + I,\qquad
\dot u = a(bv-u).
\]

当膜电位达到峰值后重置为 `v <- c, u <- u + d`；论文使用 `a=0.02, b=0.2, c=-65, d=8`。突触输入为加权的前突触脉冲和。STDP 根据前、后神经元的相对放电时间改变权重：通常当前突触前脉冲先于突触后脉冲时加强，反向时间顺序时减弱。论文给出的 STDP 参数为 `A+=0.925`、`A-=0.9`、`tau+=tau-=20`。

### 3.3 连续奖励、工作记忆与多巴胺调制

论文并非直接把环境奖励当作固定的权重增量。它先为状态定义连续评价：

\[
r_t=R_b(s_t)+\alpha Eva(s_t),
\]

其中 `Rb` 是状态类别的基础奖励，`Eva` 是随几何位置连续变化的评价项。随后把相邻时刻评价差视为实际学习反馈：

\[
r_{end}=r_{t+1}-r_t.
\]

这一项被解释为与 TD 误差近似同构：它比较当前/下一状态的评价，既包含环境即时变化，也包含状态价值变化。正反馈时，论文设置 `DAD1=2, DAD2=0.5`；非正反馈时设置 `DAD1=0.5, DAD2=2`，并乘性调制 DLPFC 到两类纹状体的连接：

\[
W_{\mathrm{DLPFC}\to\mathrm{StrD1}} \leftarrow
W_{\mathrm{DLPFC}\to\mathrm{StrD1}}DAD1,
\]

\[
W_{\mathrm{DLPFC}\to\mathrm{StrD2}} \leftarrow
W_{\mathrm{DLPFC}\to\mathrm{StrD2}}DAD2.
\]

直观上，正反馈使直接通路更容易支持刚刚选择的动作，同时相对抑制间接通路；负反馈则相反。论文还将 MOFC/LOFC 分别用于正/负价值调制，并报告它们会放大两条竞争路径的差异。

## 4. 论文实验场景与论文报告的结果

论文在 DJI MATRICE 100、2.4 GHz 无线数字相机条件下报告两类实机任务：

1. **穿窗任务**：把无人机相对窗口的位置划分为 14 个状态，动作为左、上、右、下四个方向；目标是移向窗口中心。
2. **障碍物规避任务**：观测中障碍物位于左侧或右侧，共两个状态；动作为向左或向右飞行。

论文报告，在穿窗任务中，模型对一个新状态通常可在三次试验内形成正确动作偏好；并在四个起点各做 15 次实验，将所需步数与此前 PFC-BG 模型比较。论文还报告，在 100 个决策步的动作生成迭代统计中，包含 MOFC/LOFC 时平均需要 51.423 次内部迭代，不含它们时为 70.613 次。这些数值来自论文，不是本次复现得到的测量结果。

论文也明确指出其局限：状态需事先定义，自动状态分类尚待研究；仅抽象了相位性 DA 奖惩变化，未完整建模张力性 DA 或更复杂的神经调制机制。这些限制对于后续将模型接入视觉、通信或硬件时仍然成立。

## 5. 当前 BrainCog `BDM-SNN.py` 的实际模型

### 5.1 与论文的对应关系和差异

当前运行版本是用于快速控制实验的工程化近似，而不是论文 11 模块模型的逐项复刻。

| 项目 | 2018 论文 | 当前 `BDM-SNN.py` 默认路径 |
| --- | --- | --- |
| 神经元 | Izhikevich RS | `node_type="lif"` 实际实例化 `IFNode`：无泄漏积分、阈值 0.5、发放后清零 |
| 显式模块 | 11 个，含 MOFC、LOFC、SNc/VTA | 8 个：DLPFC、StrD1、StrD2、STN、GPe、GPi、丘脑、PM |
| DA/OFC | 连续评价、`r_end`、DAD1/DAD2 与 MOFC/LOFC | 不显式模拟 SNc/VTA、MOFC、LOFC；环境奖励在外部直接调制两条 DLPFC-纹状体连接 |
| 动作竞争 | PM 首脉冲获胜的延迟编码 WTA | 离散内部步中 PM 首次非零输出时取 `argmax`；同一时步并列时 PyTorch 默认选索引较小者 |
| 感知 | 相机输入经外部状态识别 | 游戏几何关系直接离散为状态，不学习视觉表征 |
| 任务 | 实机 UAV 穿窗/避障 | Pygame Flappy Bird，两个动作 |

仓库将这一版本称为 LIF 实现，但代码在 `node_type="lif"` 下调用的是 `IFNode`，其膜电位更新为 `mem <- mem + input * dt`，没有 LIF 的泄漏项。另有 `BDM-SNN-hh.py`，它选择简化 HH 神经元变体。后续报告中应按代码实际称为 IF 版本，避免把神经元动力学误记为论文 Izhikevich 或严格 LIF。

### 5.2 本次 Flappy Bird 的规模与连接图

本次设置为 `Ns=9`、`Na=2`、兴奋权重基值 `+1`、抑制权重基值 `-0.5`。八个群体合计 55 个神经元：DLPFC 9，StrD1 18，StrD2 18，STN 2，GPe 2，GPi 2，丘脑 2，PM 2。

```text
状态 one-hot(幅值 2)
       |
       v
    DLPFC(9) ----> StrD1(18) --| GPi(2) --| Thalamus(2) -> PM(2) -> 动作
       |              ^              ^                           |
       |              |              |                           | 横向抑制
       |              +-- StrD2(18) -| GPe(2) -------------------+
       +----------------------------> STN(2) -> {GPe, GPi}
       +------------------------------------> Thalamus
```

代码中的连接矩阵、符号与初值如下；“特异”表示状态 `s` 只连向状态-动作索引 `s*Na+a`，而非所有状态-动作单元：

| 连接 | 拓扑 | 初始权重 |
| --- | --- | ---: |
| 输入 -> DLPFC | 一对一 | `+1` |
| DLPFC -> StrD1、StrD2 | 特异，`9 x 18` | `+1` |
| DLPFC -> STN | 全连接，`9 x 2` | `+1` |
| StrD1 -> GPi | 特异，`18 x 2` | `-0.5` |
| StrD2 -> GPe | 特异，`18 x 2` | `-0.5` |
| GPe -> GPi | 一对一 | `-0.5` |
| STN -> GPi、GPe | 全连接 | `+0.5` |
| GPe -> STN | 全连接 | `-0.25` |
| GPi -> 丘脑 | 一对一 | `-0.5` |
| 丘脑 -> PM | 一对一 | `+1` |
| DLPFC -> 丘脑 | 全连接，`9 x 2` | `+0.2` |
| PM -> PM | 去掉对角线的横向抑制 | `-2.5` |

一次前向传播按 DLPFC、StrD1/StrD2、STN、GPe、GPi、丘脑、PM 的次序更新。STN 使用上一内部时间步的 GPe 输出，PM 也使用上一内部时间步的 PM 输出，因此两者包含离散时间的递归状态；每次环境反馈后调用 `DM.reset()` 清除神经元膜电位与 STDP 内部痕迹，但不清除已学权重。

## 6. 当前代码的动作选择与在线学习

### 6.1 内部脉冲仿真与动作输出

对于当前状态 `s`，环境构造长度为 9 的 one-hot 输入 `x`，仅 `x[s]=2`。`chooseAct` 最多执行 500 个内部脉冲时间步：

\[
o_t,\;dw_t=\operatorname{BDMSNN}(x),\qquad
a=\arg\max(o_t)\ \text{当}\ \max(o_t)>0.
\]

其中 `o_t` 是 PM 的两个输出脉冲。若 500 步内 PM 从未发放，函数没有定义回退动作；这是当前实现的失败边界。实际本次运行中网络会产生 PM 脉冲。动作 `0` 表示不爬升（小鸟下沉），动作 `1` 表示爬升；这不是 UAV 的四向动作空间。

### 6.2 局部 STDP 痕迹和奖励调制更新

`STDP`/`MutliInputSTDP` 在每次内部步中维护输入痕迹，基础衰减为 `0.99`。实现利用局部连接的 Autograd 张量构造 `dw`，它承担局部 STDP/资格迹信号的角色，并非由全局任务损失反向传播得到的标准深度学习梯度。

动作选择窗口中，DLPFC 到 StrD1、StrD2 的局部痕迹分别累积为：

\[
E^{D1}\leftarrow0.8E^{D1}+dw^{D1},\qquad
E^{D2}\leftarrow0.8E^{D2}+dw^{D2}.
\]

行动后，环境给出当前转移的标量奖励 `r`。代码构造一个 `9 x 18` 奖励矩阵 `R`：默认元素为 1，而刚刚执行的 `(state, action)` 元素改为 `r`。随后只更新当前状态对应的 DLPFC 行：

\[
\Delta W_{D1}=R\odot E^{D1},\qquad
\Delta W_{D2}=-R\odot E^{D2}.
\]

更新前，该状态的两个候选动作增量会做零均值、单位标准差归一化；再乘以结构掩码、加到权重，并将这两个权重除以其中最大值。最后 DLPFC->StrD1/StrD2 权重都截断为非负。正奖励的设计意图是相对增强直接通路中已选状态-动作，并相对抑制间接通路；负奖励则相反。

这与论文的 `DAD1/DAD2` 乘性 DA 方程不是同一个数值实现。特别地，当前奖励幅度会先进入所选动作与另一候选动作的相对差异，随后又被二动作归一化，因此不能把累计奖励数值直接解释为线性权重增量，也不能将此版本称为逐项复现论文的连续 DA 模型。

## 7. Flappy Bird 环境、状态、奖励与终止条件

### 7.1 环境动力学

- 窗口大小为 `568 x 512`，小鸟大小为 `32 x 32`，初始横坐标为 250。
- 游戏以 60 Hz `clock.tick(60)` 更新；管道每 2 秒生成一对，宽度 80 像素，横向速度为每帧 3 像素。
- 在动作 `1` 时小鸟上升；动作 `0` 时小鸟下沉。状态 2--5 中上升/下沉速度会加倍。
- 小鸟触碰管道、越过上下边界即碰撞；管道完全经过小鸟后得 1 分。
- 当前脚本的 `msec_to_climb` 在创建时为 2，且代码中未递减，所以动作 1 在整个回合中持续表现为上升；这是实现细节，不应假定为标准 Flappy Bird 的有限时长拍翅模型。

### 7.2 九个状态

状态不是直接由像素喂入网络，而是由小鸟中心相对最近管道空隙中心的位置、是否已靠近管道边缘和是否碰撞计算。

| 状态 | 代码含义 |
| --- | --- |
| 0、1 | 位于目标空隙中心上方/下方的安全带内，或接近空隙中心 |
| 2、3 | 仍在空隙范围内但离中心较远的上方/下方区域 |
| 4、5 | 明显偏离空隙中心的过高/过低区域 |
| 6、7 | 管道已接近且小鸟靠近上/下管道边缘的危险区域 |
| 8 | 已发生碰撞 |

当尚未有可交互管道时，代码以游戏窗口中线为临时目标中心；当新管道刚成为目标时，`isNewPipe=True`，奖励函数会避免把状态切换误判为控制错误。

### 7.3 奖励表

奖励并不是论文的 `r_end=r_(t+1)-r_t` 连续评价，而是当前 Flappy Bird 的手工规则：

| 后继状态及条件 | 奖励 |
| --- | ---: |
| 0 或 1 | `+6` |
| 2 或 3，状态不变、非新管道且距离中心变小 | `+3` |
| 2 或 3，状态不变但距离未变小 | `-5` |
| 2 或 3，状态切换或新管道 | `-3` |
| 4 或 5，状态不变、非新管道且距离中心变小 | `+3` |
| 4 或 5，状态不变但距离未变小 | `-8` |
| 4 或 5，状态切换或新管道 | `-5` |
| 6 或 7，状态不变、非新管道且距离中心变小 | `+3` |
| 6 或 7，其余情形 | `-3` |
| 8（碰撞） | `-100` |

环境大约每两帧进行一次“状态 -> 动作 -> 后继状态 -> 奖励 -> 权重更新”循环。`lif_reward_l.npy` 保存每次更新的奖励，`lif_score_l.npy` 保存每个游戏帧的累计得分。

## 8. 本次已完成运行的解读

本次固定 `BDMSNN_SEED=0`、物理 GPU 1、headless SDL，使用 `BDMSNN_MAX_FRAMES=600`。限制为 600 帧后是以测试上限退出，不等同于碰撞终止，因此不应把本轮结果理解为“存活至失败”的完整 episode 统计。

实际输出为：600 个游戏帧、300 次在线更新、最终/最高得分均为 4、累计奖励 1662；奖励计数为 `+6:267`、`+3:28`、`-3:2`、`-5:2`、`-8:1`。得分提升发生在帧 132、252、372、492 附近。它证明了以下最小闭环均已在 CUDA 上运行：状态编码、PM 动作发放、游戏动力学、手工奖励、两条 DLPFC-纹状体连接的在线更新及结果落盘。

但该结果尚不能证明“稳定通关”或统计显著性。合理的下一层基线应至少包含多个固定随机种子、完整 30,000 帧上限或自然碰撞终止、每回合分数/碰撞率/奖励/内部动作迭代数，以及 DLPFC、StrD1、StrD2、STN、GPe、GPi、丘脑、PM 的脉冲计数与权重轨迹。

## 9. 对后续低维通信与 RRAM 研究的含义

当前 BDM-SNN 是单个 PyTorch 进程中的软件网络，并没有定义多核划分、AER 包、链路时延/FIFO、RRAM 交叉阵列读写、模拟神经元失配或能耗模型。因此它只能作为“未压缩的软件决策基线”，不能直接用于宣称通信量或硬件能耗收益。

后续若将其拆为多核系统，应先固定本节的状态、动作、奖励、种子和完整对照路径，再明确哪些连接跨核。例如可先把 DLPFC/纹状体与 STN-GPe-GPi-丘脑-PM 拆为两个逻辑核。对每个跨核边界分别记录逻辑脉冲数 `N_logic`、物理包数 `N_packet`、链路比特 `B_link`、峰值速率、突发度、目标端阵列激活次数和任务得分。只有在全连接基线、低秩通信和低秩加事件控制三组使用相同任务与种子比较后，才能判断低维表示是否真的降低了通信或硬件代价。

## 10. 参考资料

1. Zhao, F., Zeng, Y., and Xu, B. (2018). *A Brain-Inspired Decision-Making Spiking Neural Network and Its Application in Unmanned Aerial Vehicle*. Frontiers in Neurorobotics, 12:56. DOI: `10.3389/fnbot.2018.00056`。
2. BrainCog 当前检出版本的 `examples/decision_making/BDM-SNN/README.md`、`BDM-SNN.py`、`braincog/model_zoo/bdmsnn.py`、`braincog/base/brainarea/basalganglia.py`、`braincog/base/learningrule/STDP.py` 与 `braincog/base/node/node.py`。

## 11. RRR 低维通信的第一轮原型（2026-07-19）

当前 BDM-SNN 以人工结构初始化、局部 STDP 和在线奖励调制学习 DLPFC 到 StrD1/StrD2 的 `9 x 18` 状态-动作权重。它没有可供反向传播优化的端到端任务损失，因此不适合直接套用“训练时对该权重施加低秩正则”的常见深度网络做法：这会把多个本应独立的状态-动作记忆强制共享表征，并改变在线可塑性本身。

本次首先以全通信基线的 300 次在线更新采集 DLPFC 源活动、纹状体活动、目标电流和权重轨迹。对已访问的 5 个状态，DLPFC 源活动的中心化秩为 4；但训练后 DLPFC->StrD1 和 DLPFC->StrD2 权重均为满秩 9，前 4 个奇异值各只解释约 60% 的权重能量。因此，`DLPFC(9) -> Striatum(18)` 的 rank-4 并非无损近似，不适合作为第一个低秩传输结论。

第一轮 RRR 原型改为跨越更合理的边界：将完整 STDP 学习留在本地 `DLPFC -> StrD1/StrD2` 权重上，而把 `StrD1(18) -> GPi(2)` 与 `StrD2(18) -> GPe(2)` 作为两个候选跨核链路。仅这两条链路的前向电流会在拟合后被 RRR 重建电流替换；`STN -> GPe`、`GPe -> GPi`、`STN -> GPi` 以及其余路径仍为全维原路径。实现仍计算这两条固定结构连接的完整电流，作为 RRR 拟合目标和原网络局部 STDP 导数计算的参考；不过当前游戏循环实际只调用 `UpdateWeight(0)` 和 `UpdateWeight(1)`，即只对两条 `DLPFC -> StrD1/StrD2` 连接施加奖励调制的在线更新，`StrD1 -> GPi`、`StrD2 -> GPe` 本身并未更新权重。从因果历史窗口拟合仿射 RRR，并在之后的前向传播中实施：

\[
z=(x-\bar{x})P,\qquad \hat y=zD+\bar y.
\]

这里 `x` 是 18 维纹状体脉冲向量，`y` 是 2 维完整目标电流，`z` 是 `k` 维 latent 通信表示。这里有两个不同的计数器，不能把它们混为一谈：

1. `BDMSNN_RRR_WARMUP=200` 和 `BDMSNN_RRR_REFIT_INTERVAL=200` 都按一次“环境动作 -> 奖励 -> `updateNet` -> `UpdateWeight(1)`”计数；通常约两游戏帧一次。600 帧实验有 300 次此类更新，因此第 200 次更新结束时首次拟合，随后才开始采用 RRR 前向电流；下一次重拟合本应在第 400 次更新，但本轮没有达到。
2. `BDMSNN_RRR_WINDOW=200` 则按 `BDMSNN.forward()` 调用计数。一次环境决策在 `chooseAct` 内会执行 1--500 个内部 SNN 步直到 PM 发放动作，因此第一次拟合实际使用的是最近 200 个内部步的样本，并不是“恰好前 200 次环境更新各一个样本”。这是当前原型的时间尺度不一致，下一版应改为每个决策窗口的因果脉冲计数/统计量后再拟合和传输。

固定 seed 0、600 帧的结果如下：

| 跨核通信 | 得分 | 累计奖励 | D1->GPi 运行时窗口保真度 | D2->GPe 运行时窗口保真度 |
| --- | ---: | ---: | ---: | ---: |
| 全通信 | 4 | 1662 | 不适用 | 不适用 |
| RRR `k=1` | 4 | 1662 | 1.000 | 0.711 |
| RRR `k=2` | 4 | 1662 | 1.000 | 1.000 |

`k=1` 的单条链路可写成 `18 -> 1 -> 2`；两条链路相互独立，系统共有两个标量 latent。`k=2` 则是 `18 -> 2 -> 2`：由于目标电流本身只有 2 个分量，它在目标电流空间达到满秩、可作为无损保真对照，并不压缩目标电流的维数。从“18 个源地址/源脉冲”角度，它仍是到两个 latent 分量的表示降维；但当前 latent 是连续标量而非 AER 脉冲，故没有测得或声明任何逻辑事件数、包数或比特数的降低。`k=1` 才把每条 18 维纹状体输出压缩为 1 个连续 latent 通道；它在这个单一 seed 和短窗口中没有降低分数，但 D2 通路约 29% 的运行时窗口目标电流方差没有被保留，尚不能认为稳健。

补充的 paired 多 seed 结果验证了这个风险：在 seed 0、1、2 上，全通信均完成 600 帧、得分 4；rank-1 RRR 在 seed 0、2 也完成 600 帧且得分 4，但在 seed 1 于第 520 帧碰撞退出（仍得 4 分）。分数在一个管对完全越过小鸟后才加 1；本轮四次加分位于约第 133、253、373、493 帧，第五个管对即使不碰撞也要约第 613 帧后才能加分。因此 seed 1 是先获得四分、再在第五个管对前碰撞。当前代码在发生碰撞时没有把 `state=8` 的 `-100` 奖励写入 `num_reward`，所以旧结果 JSON 中 `collision_count=0` 不能用来判定是否碰撞，应以实际退出帧/环境状态为准。这表明短时间的最终分数不足以评估闭环稳定性，rank-1 目前只能被视为有损候选，而非已验证的性能保持压缩。

结果文件：`rrr_striatum_to_output_seed0_600.json`、`rrr_striatum_to_output_seed0_600.png`、`rrr_rank1_multiseed_600.json` 与 `rrr_rank1_multiseed_600.png`；实现位于 `braincog/model_zoo/communication_subspace.py`，通过 `BDMSNN_RRR_RANK`、`BDMSNN_RRR_MODE=striatum_to_output`、`BDMSNN_RRR_WINDOW`、`BDMSNN_RRR_WARMUP` 与 `BDMSNN_RRR_REFIT_INTERVAL` 控制。

### 11.1 留出集目标电流方差图（2026-07-20）

“目标电流”不是完整 GPi/GPe 神经元收到的总电流，而是被压缩的单条结构性分量：对每个决策样本 \(t\)，以纹状体二值脉冲 \(x_t\in\mathbb{R}^{18}\) 乘该固定的 `18 x 2` 抑制性连接矩阵 \(W\)，得到 \(y_t=x_tW\in\mathbb{R}^{2}\)。`D1 -> GPi` 的总输入还包含 `GPe -> GPi` 和 `STN -> GPi`；`D2 -> GPe` 的总输入还包含 `STN -> GPe`，这些未纳入本图。该游戏的固定映射使单个分量通常取 `0` 或 `-0.5`，所以图中的阶跃是正常的。

为了避免只报拟合窗口内部的保真度，新增 `examples/decision_making/BDM-SNN/plot_rrr_target_current.py`：它用 seed 0 基线的前 200 个“环境决策/奖励更新”样本拟合，在未参与拟合的后 100 个样本计算

\[
\operatorname{EV}=1-\frac{\sum_{t,j}(y_{t,j}-\hat y_{t,j})^2}{\sum_{t,j}(y_{t,j}-\bar y_{\mathrm{test},j})^2}.
\]

其中分母 `SST` 是留出时间段中目标电流相对其留出集均值的总变化，分子 `SSE` 是重建误差。因此 `EV=0.738` 的含义是 RRR `k=1` 解释了 D2->GPe 两个目标电流分量随时间变化的约 73.8%，仍有约 26.2% 的变化成为残差；它既不是“保留了 73.8% 的源脉冲”，也不是“任务成功率为 73.8%”。留出集汇总如下：

| 链路 | `k=1` 通道 0 | `k=1` 通道 1 | `k=1` 合并 EV | `k=2` 合并 EV |
| --- | ---: | ---: | ---: | ---: |
| StrD1 -> GPi | 0.997 | 0.997 | 0.997 | 1.000 |
| StrD2 -> GPe | 0.777 | 0.697 | 0.738 | 1.000 |

图 `examples/decision_making/BDM-SNN/rrr_target_current_heldout_seed0.png` 的左列将留出集真实电流与 `k=1` 重建电流叠加，右列给出每个目标通道和合并统计的 EV；相应的机器可读结果在 `rrr_target_current_heldout_seed0.json`。它是离线、决策级的验证，不能与上表运行时内部步窗口的 `0.711` 直接混用。

### 11.2 下一阶段：从“低维数值”变为“低事件通信”

当前 RRR 仅证明了一个连续值重建的有损/保真关系，尚未产生可跨核传输的脉冲，更没有证明链路节能。下一阶段按以下顺序进行：

1. **先修正测量时间尺度。** 每个动作决策窗口记录源端 18 维脉冲计数、真实目标电流和动作时间，并用这些因果、等时间宽度的样本做 RRR；以多 seed、自然终止 episode 比较全通信和 `k=1/2`。这样才能把重建误差、碰撞率和得分可靠对应。
2. **实现真正的 latent 脉冲通道。** 源核用 \(P\) 对本地纹状体脉冲/计数做投影，`k` 个 LIF/IF latent 神经元积分后才向目标核发送事件；目标核用 \(D\) 将事件解码为 GPi/GPe 电流。\(P,D\) 往往有正负值，RRAM/SNN 实现需用差分电导或正、负两个事件通道，不能把连续负 latent 直接当普通单极性 AER 脉冲。
3. **报告真实通信指标。** 对全通信、latent-only、latent+事件控制三组分别记录源端逻辑脉冲数 \(N_{logic}\)、物理包数 \(N_{packet}\)、链路比特 \(B_{link}\)、峰值/突发速率、目标端阵列激活次数、时延和任务指标；维数下降本身不等于这些量下降。
4. **再加入残差（delta）编码。** 发送端和接收端以相同的预测器维护“本应有的 latent” \(z_{pred,t}\)。发送端计算 \(e_t=z_t-z_{pred,t}\)，仅当 \(\lvert e_t\rvert>\theta\) 时发送带符号的 delta/脉冲；接收端收到后作同样的状态修正，未收到事件时双方均沿预测状态前进。若 latent 随时间缓慢变化，这会把稳定区间的连续传输变为零事件，从而可能降低 AER 流量。代价是阈值带来的失真，以及链路丢包、时钟偏差、神经元漏电和 RRAM 噪声会使两端预测漂移；因此必须有定期刷新/检查点并测量漂移对动作的影响。该机制不是无损保证，只有在时间语义验证后才可计入节省。

## 12. 扩展到多脑区跨核通信的 RRR 消融（2026-07-20）

### 12.1 为什么第一轮只选两条纹状体输出链路

`StrD1 -> GPi` 和 `StrD2 -> GPe` 首先被选中，不是因为其他脑区不能压缩，而是因为它们是第一版最适合暴露问题的候选：源群体为 18 个纹状体神经元、目标仅有 2 个输出神经元，存在明显的 `18 -> k -> 2` 表示降维空间；两条连接是固定结构权重，不会直接改写正在在线 STDP 学习的 `DLPFC -> StrD1/StrD2` 状态-动作记忆；且它们位于直接/间接通路的分叉处，对闭环决策有足够影响力。反之，直接把 RRR 写入塑性记忆权重，会混淆“通信近似误差”和“在线学习规则已被改变”这两个因素。

这并不构成“只允许压缩这两条”的结论。RRR 的通用对象是任意跨核源活动到目标电流映射；但是否应启用压缩取决于实际核划分、源/目标维度、时变性、拟合窗口是否覆盖行为状态，以及误差是否会穿过递归环路被放大。尤其对 `2 -> 2` 小链路，rank-1 的地址数优势有限；对控制环路，哪怕每个窗口的电流 EV 为 1，也不自动保证跨状态外推或闭环稳定。

### 12.2 本次逻辑四核划分与候选链路

为测试更通用的跨脑区方案，新增 `all_cross_core` 模式，按以下逻辑四核划分，而不是宣称它已是最终物理芯片布局：

| 逻辑核 | 脑区 | 本地、不压缩的连接 |
| --- | --- | --- |
| Core A | DLPFC、StrD1、StrD2 | `DLPFC -> StrD1`、`DLPFC -> StrD2` |
| Core B | STN | 无 |
| Core C | GPe、GPi | `GPe -> GPi` |
| Core D | 丘脑、PM | `Thalamus -> PM`、`PM -> PM` |

在此划分下，共有八条候选跨核连接：`DLPFC -> STN (9x2)`、`StrD2 -> GPe (18x2)`、`STN -> GPe (2x2)`、`StrD1 -> GPi (18x2)`、`STN -> GPi (2x2)`、`GPe -> STN (2x2)`、`GPi -> Thalamus (2x2)`、`DLPFC -> Thalamus (9x2)`。代码允许用 `BDMSNN_RRR_LINKS` 选择其中任意子集；未选链路严格保持全通信，因此可逐条消融，而不是把“全脑区压缩”当作不可解释的黑盒。

### 12.3 延长窗口实验设置

所有新实验使用 CUDA、headless SDL、`BDMSNN_FRAME_RATE=0`、最大 3000 游戏帧（1500 次奖励更新）、`RRR_WINDOW=400` 个内部 SNN 步、`WARMUP=200` 次外部更新、`REFIT_INTERVAL=200` 次外部更新。全通信基线也采用相同逻辑核边界并记录八条全源脉冲的逻辑事件计数。该上限足以通过约 24 个管对，明显长于前述 600 帧/4 分的演示窗口。

| 方法（seed 0） | 被替换的跨核链路 | 完成帧数 | 得分 | 累计奖励 |
| --- | --- | ---: | ---: | ---: |
| 全通信 | 无 | 3001（上限） | 24 | 8513 |
| 全八条 RRR `k=1` | 全部八条 | 625（碰撞） | 5 | 1657 |
| 全八条 RRR `k=2` | 全部八条 | 625（碰撞） | 5 | 1657 |
| 仅纹状体 `k=1` | `StrD1 -> GPi`、`StrD2 -> GPe` | 625（碰撞） | 5 | 1657 |
| 小环路 `k=1` | `STN -> GPe`、`STN -> GPi`、`GPe -> STN` | 3001（上限） | 24 | 8513 |
| 小环路+丘脑 `k=1` | 前三条加 `GPi -> Thalamus` | 3001（上限） | 24 | 8513 |

seed 1 的独立延长验证中，全通信和“小环路+丘脑 `k=1`”也都完成 3001 帧、得分 24、累计奖励 8311。故当前证据支持“有选择地压缩该小环路在两个 seed、3000 帧上未降低任务表现”，但仍不足以证明所有随机种子、长自然 episode 或硬件扰动下均无损。

`k=2` 的全链路结果与 `k=1` 一样提前退出，是一个重要警告：`k=2` 只代表 2 维目标电流在**当前拟合窗口内**可以满秩表示；它不是跨时间、跨未见状态或跨递归闭环的全局无损保证。前 200 次更新的内部步样本覆盖不足、之后在线 STDP 改变活动分布，都会造成 RRR 外推偏差。更频繁地每 20 次更新重拟合亦未改善全链路 `k=1`（539 帧、4 分），表明问题不只来自重拟合频率。

### 12.4 这次记录了什么通信量，以及不能如何解读

新增的全通信监视器记录每条逻辑跨核边界的源端脉冲数；RRR 路径记录预热后原本会发送的源脉冲数，以及实际产生的连续 latent 标量数。以 seed 0 的“小环路+丘脑”组为例，预热后的四条被替换链路共有 135124 个源逻辑脉冲，而 RRR 输出为 98396 个连续 scalar 值，地址/数值槽位代理下降约 27.2%。仅小环路三条时，代理下降约 37.5%。

这不是 AER 事件、包数、比特数、阵列读数或能耗的实测节省：当前 latent 是连续浮点值，尚无阈值神经元、正负事件通道、量化、编码包或链路模型。这个代理仅说明如果每个 latent scalar 最终能以一个可比较的事件/数值槽位传输，存在进一步事件化和硬件映射的候选空间。完整结果图为 `examples/decision_making/BDM-SNN/all_cross_core_extended_seed0_summary.png`，汇总 JSON 为 `all_cross_core_extended_seed0_summary.json`。

### 12.5 当前可用的结论与下一步

1. RRR 可以在代码层面对任意选定的跨核连接建立独立的 `source -> target current` 模型；第一轮只选纹状体输出是保守的研究起点，而不是方法限制。
2. “全边统一 rank-1/2”在本网络中失败，说明跨核压缩必须按链路、状态覆盖和控制敏感性选择，不能将局部 EV 当作部署许可。
3. 当前安全候选是 `STN -> GPe`、`STN -> GPi`、`GPe -> STN`，并可在本测试中扩展到 `GPi -> Thalamus`；应保留两条纹状体输出和 DLPFC 驱动边为全通信，直至使用决策级、跨状态训练样本与显式保守门控重新验证。
4. 下一步应把每个决策窗口的源脉冲计数、目标电流、真实动作延迟做成因果 RRR 样本；再加入“仅当留出窗口误差低于阈值才启用”的链路级门控，以及真正 LIF latent/AER 事件、正负差分编码、残差编码和 RRAM 噪声/时延模型。

## 13. RoboSuite Panda Lift：BDM-SNN 全通信基线与算法流程（2026-07-23）

本节是与 Flappy Bird 独立的第一阶段机械臂实验。入口为
`Brain-Cog/examples/decision_making/BDM-SNN-Robosuite/lift_bdm_snn.py`；它保留
BDM-SNN 的 DLPFC、StrD1、StrD2、STN、GPe、GPi、丘脑和 PM 拓扑，以及在线
reward-modulated STDP（R-STDP），但**没有启用 RRR 或其他通信压缩**。因此它的
作用是先验证“机器人任务—SNN—奖励学习—跨核流量测量”的全通信闭环；不能把它的
逻辑 spike 计数称为 AER 包数、比特数或能耗。

### 13.1 整体闭环：感知—记忆—决策—整合—行动—学习

每个决策时刻执行如下闭环：

\[
o_t \xrightarrow{\text{编码}} s_t \xrightarrow{\mathrm{DLPFC/BG}} a_t
\xrightarrow{\text{Panda 原语}} o_{t+1},r_t
\xrightarrow{\text{eligibility + R-STDP}} W_{t+1}.
\]

| 大类 | 本实验对应模块 | 做什么 |
| --- | --- | --- |
| 感知 | `LiftStateEncoder` | 从 robosuite 低维真值读取方块位置、末端位置和夹爪状态；不使用 RGB 或 CNN。相对位置的 8 个方位、4 个距离档、2 个夹爪闭合/抓取模式组成 \(S=8\times4\times2=64\) 个离散状态。 |
| 记忆 | DLPFC \(64\) 与 StrD1/StrD2 \(64\times8\) 的可塑连接 | DLPFC 对当前状态发放；通往直接/间接纹状体通路的状态—动作突触权重保存长期经验。每个 episode 清空膜电位与 eligibility trace，但不清空这些长期权重。 |
| 决策 | 直接通路 StrD1→GPi 与间接通路 StrD2→GPe→STN→GPi | 两条基底节通路将状态—动作脉冲转换为对 8 个候选动作的竞争/抑制。STN、GPe、GPi 保留原 BDM-SNN 的递归调节结构。 |
| 整合 | GPi→丘脑、DLPFC→丘脑、丘脑→PM、PM 侧抑制 | 丘脑整合基底节门控与 DLPFC 驱动；PM 的 8 个输出神经元经竞争产生动作。PM 全静默时使用随机回退，并记录该事件；最大脉冲相同时随机打破并列，避免初始对称权重永久偏向动作 0。 |
| 行动 | `LiftActionPrimitives` + Panda OSC pose 控制器 | 将 PM 的离散动作映射为 \(+x,-x,+y,-y,+z,-z\) 与 `gripper_open/close` 共 8 个短时笛卡尔原语；每个高层决策执行固定个数的 MuJoCo 底层控制步，而非直接输出连续力矩。 |
| 学习 | 进度奖励、eligibility trace、`UpdateWeight` | 根据距离缩短、首次抓取、成功抬起、停滞/超时/不安全状态形成 R-STDP 调制奖励。每个内部 SNN 步累积 D1/D2 eligibility；奖励到来后仅更新当前状态的 8 个动作突触切片 \([sA:(s+1)A]\)，再归一化，因此实现不再依赖 Flappy Bird 的 2 动作硬编码。 |

在逻辑多核边界上，当前实现把 \{DLPFC, StrD1, StrD2\}、STN、\{GPe, GPi\}、
\{丘脑, PM\} 视为四个核，并监测 8 条跨核连接。`communication_rank=0` 时网络
严格使用完整突触电流；监测器只累计源端 logical spike events，不改变前向传播。

### 13.2 先验证任务可达性，再验证学习

在学习之前先运行 robosuite 集成检查：Panda Lift 能 reset、返回 `cube_pos`、
`robot0_eef_pos` 和夹爪观测，执行随机 7 维底层 OSC 动作并产生奖励；可选
`frontview_image` 尺寸为 \(64\times64\times3\)。实际 BDM-SNN 的动作空间仍是上述
8 个离散原语，故“底层 action dimension=7”并不和 \(A=8\) 冲突。

Lift 的接触对初始姿态敏感，故加入了只用于课程/接口验证的离散教师：它使用**同一组**
8 个动作依次完成“末端移至方块侧方接触偏置—下降—夹爪关闭—上抬”。课程固定方块
XY、姿态和 Panda 初始关节噪声，以减少与算法无关的 reset 方差；教师只替代实际执行的
动作，SNN 仍进行前向、累计 eligibility 并由真实环境奖励执行同一套 R-STDP 更新。
它不是最终评估策略，也不代表 PM 自主决策。

完整命令示例（使用 CUDA 0 和 EGL 无窗口渲染）：

```bash
cd /home/lph/Brain_SNN
MUJOCO_GL=egl /home/lph/.conda/envs/lph/bin/python \
  Brain-Cog/examples/decision_making/BDM-SNN-Robosuite/lift_bdm_snn.py \
  --episodes 30 --max-decisions 180 --control-steps 1 --internal-steps 3 \
  --teacher-episodes 20 --teacher-start 1 --teacher-end 0 \
  --fixed-cube --deterministic-robot-start --seed 0 --device cuda:0 \
  --output-dir results/lift_bdm_snn_curriculum20_autonomous10
```

### 13.3 当前结果与正确解读

1. **全通信 BDM-SNN 基线已经在 Lift 上端到端运行。** 状态编码、8 类动作映射、
   BDM-SNN 前向、64 状态/8 动作的泛化 R-STDP 更新、奖励、成功检测、脑区 spike、
   权重变化、动作分布和跨核 logical spike 统计都已写入结果文件。该过程在 CUDA 上运行。
2. **任务动作接口可达，但接触仍有物理随机性。** 在固定课程条件下、教师全程执行的
   10 个 episode 中，7 个达到 Lift 稀疏成功（方块高度超过桌面 \(0.04\) m），成功轨迹
   通常约 110 次高层决策；这是“8 个离散原语能够完成抓取/抬起”的验证，而非学习成绩。
3. **自主学习尚未调通，结论必须保守。** 运行 30 个 episode、前 20 个教师概率由 1
   线性退火到 0、最后 10 个无教师 episode 后，最后 10 个纯 PM 自主 episode 的稀疏
   成功率为 **0/10**。全部 30 个 episode 的稀疏成功率为 4/30，且这些成功均发生在仍有
   教师动作的课程期。因此当前基线可作为可复现的集成与测量平台，但不能声称 BDM-SNN
   已学会自主 Lift，也不应在此之前加入 RRR 比较性能。

该次运行的完整指标位于
`results/lift_bdm_snn_curriculum20_autonomous10/lift_bdm_snn_metrics.json`，图位于同目录
的 `lift_bdm_snn_baseline.png`。全程共 15,681 个内部 SNN 步；例如
StrD1→GPi 的源端 logical spikes 为 35,295，StrD2→GPe 为 41,544。这些数值仅是后续
通信压缩对照的 \(N_{logic}\)，尚未实现 latent 脉冲、AER 封包、链路 bit 或 FIFO 模型。

### 13.4 下一步（仍保持全通信）

当前失败的主要原因是长时序抓取的 credit assignment：64 状态编码没有显式携带“接近—
下降—闭合—抬起”的阶段记忆，而单步标量奖励把教师实际执行动作与 PM 自身 spike
eligibility 混合。下一步应先在全通信条件下比较两项可解释消融：(1) 教师阶段对 PM/
纹状体相应动作切片进行 action-clamp eligibility，使奖励确实归因到教师动作；(2) 增加
不改变原始几何观测的有限阶段/近期动作记忆，或使用决策窗口 eligibility，使“抓到后上抬”
能获得稳定正向 credit。只有自主成功率在多个 seed 和随机初始条件下稳定后，才将视觉
编码和 RRR latent 通信作为第二阶段变量逐一加入。

### 13.5 自主学习诊断与消融（2026-07-23）

针对“教师能完成、教师撤除后 PM 不能完成”的现象，继续保持 CUDA 全通信、固定方块
课程和相同的 20 个教师预训练 episode + 10 个无教师测试 episode，依次测试了三个
不含 RRR 的解释性修改。所有表中的“自主成功”仅统计最后 10 个 `teacher_decisions=0`
的 episode。

| 组别 | 改动 | 自主成功 | 结论 |
| --- | --- | ---: | --- |
| 原课程 | 自然 STDP eligibility，教师概率线性由 1 降为 0 | 0/10 | 教师执行与 SNN 自然 eligibility 不完全对应。 |
| action-clamp | 教师执行动作 \(a_T\) 时，仅令 \((s,a_T)\) 突触 eligibility 为 1；其他状态/动作置零 | 0/10 | 修正局部 reward credit 仍不足以形成完整抓取策略。 |
| action-clamp 预训练 | 前 20 个 episode 教师概率固定为 1，随后完全撤除教师 | 0/10 | 失败不是因为线性退火过快或课程样本过少。 |
| 可观测阶段上下文 | 将几何/夹爪导出的“接近、下降、接触关闭、抬起”4 档与原 64 状态组合为 256 状态 | 0/10 | 仅拆分粗粒度状态别名仍未解决动作策略。 |

这里的 action-clamp 并非让教师直接写入网络权重：BDM-SNN 仍前向发放、仍由环境奖励
和原有 D1/D2 R-STDP 更新；它只在教师接管时将更新 credit 明确归给实际执行的离散动作。
实现和每 episode 的 `action_clamp_decisions`、教师与 PM 的动作一致率均记录在
`lift_bdm_snn.py` 的 JSON 输出中。

进一步检查发现，原实现每次高层决策仅积分 3 个内部 SNN 步，自主期 1,800 个决策中
约有 1,150 次 PM 完全静默，因而主要由随机回退动作控制。故新增“最少 3 步、PM 首次
发放即决策、最多 30 步”的可选决策窗口。它在教师 smoke 中将静默降至约 12--16/110
决策；完整自主测试中也降至每 episode 2--28/180。尽管如此，该组自主成功仍为 0/10：
问题已从“PM 未发放”缩小为“PM 已发放但尚未输出正确的序列化策略”。完整结果分别在：

- `results/lift_bdm_snn_action_clamp_curriculum20_autonomous10/`
- `results/lift_bdm_snn_action_clamp_pretrain20_autonomous10/`
- `results/lift_bdm_snn_phase_context_pretrain20_autonomous10/`
- `results/lift_bdm_snn_adaptive_window_pretrain20_autonomous10/`

因此当前最稳妥的结论是：全通信 Lift 平台和三个学习诊断均已可运行并给出负结果，但
还不具备可比较 RRR 的自主控制基线。后续应先把“老师给出轨迹”从逐步替代动作改成明确
的行为克隆/动作读出训练，或重新设计能表示连续阶段和动作持续时间的 SNN 状态；并以
多 seed 的无教师成功率作为进入压缩阶段的门槛。

### 13.6 教师轨迹的显式行为克隆诊断（2026-07-23）

为区分“BDM-SNN 的 PM 读出失败”与“教师轨迹本身不能从当前观测学习”，增加了与
BDM-SNN **分开报告**的监督读出诊断。教师期仍运行完整 BDM-SNN、全通信监测与 R-STDP；
诊断读出只在无教师测试期替代物理动作，绝不计作 BDM-SNN 的自主成功。

1. `behavior_clone`：教师动作发生时，将当前状态行的 D1 教师动作权重置为最大、同一
   动作的 D2 权重置弱，其他动作反向处理。该方法把教师期 PM 与教师动作一致率提升到
   约 0.70--0.75，但 20 个教师 episode 后的 PM 测试仍为 0/10；这说明 D1/D2 权重偏好
   并未在现有基底节动态中稳定转化为正确 PM 序列。
2. `clone_table`：对离散状态记录教师动作多数票。4 episode smoke 曾出现 1/4 Lift
   成功，但完整 20+10 复验为 0/10；课程最终仅覆盖 15 个状态，其中 3 个有多个教师动作
   标签，且测试早期可出现大量未见状态。静态查表不能处理接触扰动后的状态分布漂移。
3. 连续 `clone_mlp`：用 \(\Delta x,\Delta y,\Delta z,q_{g1},q_{g2}\) 的 5 维连续观测
   训练小型 MLP。教师样本训练准确率可达约 90%，但测试只重复横移/下降，未稳定产生
   `gripper_close` 与上抬，0/4 成功。
4. 循环 `clone_gru`：同一 5 维输入进入 episode 内保留隐状态的 GRU，训练准确率约
   91.2%，仍是 0/4 成功。这表明仅添加短期循环记忆不足以克服接触发生微小偏差后进入
   未覆盖观测区域的闭环累积误差。

代码中的 `--autonomous-readout clone_table|clone_mlp|clone_gru` 明确标记为诊断，不应
作为 BDM-SNN 或 RRAM 通信结果。相应 smoke 结果位于：

- `results/lift_bdm_snn_behavior_clone_phase_context_pretrain20_autonomous10/`
- `results/lift_bdm_snn_clone_table_phase_context_pretrain20_autonomous10/`
- `results/lift_bdm_snn_clone_mlp_smoke/`
- `results/lift_bdm_snn_clone_gru_smoke/`

因此下一阶段应先改善**闭环数据覆盖与恢复机制**：在教师执行时有目的地注入小幅末端
扰动，并为“接触失败、横向偏移、未抓住方块”采集恢复动作；用这些多模态轨迹训练策略，
并以未见扰动的无教师成功率衡量。只有这种恢复策略本身稳定后，才能将它可靠映射回
BDM-SNN 的状态/记忆与 PM 读出；在此前加入 RRR 只会把控制误差误判为通信误差。

### 13.7 教师恢复、动作持续时间与第一性原理审查（2026-07-23）

本轮继续保持全通信，首先将教师从单次“接近--下降--闭合--上抬”扩展为可观测抓取
失败后的 `recover_open -> approach` 闭环恢复：若上抬 4 个高层决策仍未抓住方块，则打开
夹爪两次后重新对齐。与此同时，状态从原来的 64 个几何/夹爪离散状态扩展为
\(64\times9\times3=1728\)：附加“前一动作（8 个动作加初始符号）”及其持续时间的
`1`、`2--5`、`6+` 三档。20 个教师预训练 episode 与 10 个固定初始条件、无教师 PM
自主 episode 的结果如下：

| 组别 | 教师期 Lift 成功 | 自主 Lift 成功 | 自主抓取 | PM 静默/180 决策 | 结论 |
| --- | ---: | ---: | ---: | ---: | --- |
| 恢复教师 + 动作持续状态 | 16/20 | 0/10 | 0/10 | 1.2 | 时间状态和自适应积分解决了“PM 不发放”，没有产生正确动作序列。 |
| 同上，修复夹爪保持控制 | 16/20 | 0/10 | 0/10 | 1.2 | 修复动作语义失配后结果仍为 0/10；失配不是唯一根因。 |

结果位于
`results/lift_bdm_snn_temporal_recovery_pretrain20_autonomous10/` 与
`results/lift_bdm_snn_temporal_recovery_holdfix_pretrain20_autonomous10/`。第一组总计
26,942 个内部 SNN 积分步。这里的 1728 状态方案仅是软件诊断，不是 RRAM-ready：单条
DLPFC\(\to\)纹状体连接的形状已为 \(1728\times13824\)，远超单个 \(256\times256\)
阵列，也没有跨状态泛化能力。

本轮还发现并修复了一个必须明确区分的控制接口问题。教师将“上抬”标成动作 `+z`，但
会额外持续发送闭夹爪命令；旧代码只在教师接管时保留这个命令，导致自主 PM 即使选中同一
`+z`，其物理执行却是零夹爪命令。现已改为只要夹爪已闭合且当前动作不是 `gripper_open`，
所有策略都会保持闭夹爪。修复后仍未抓取，说明后续失败不能归因于该 bug；但此前任何
教师--自主直接比较均应以此语义差异为保留条件。

独立第一性原理审查给出以下优先级，而非继续扩大状态表：

1. **P0：统一动作语义。** 把夹爪模式作为持久控制状态，或使用“笛卡尔原语 \(\times\)
   夹爪开/保持闭合”的因子化动作；先用非 SNN 的教师重放/查表在同一低层命令接口验证
   示范可闭环复现。
2. **P1：以可观测 option 取代隐藏教师阶段。** 采用对齐、下降、闭合确认、持夹上抬、
   失败恢复等有限状态机；转移只依赖抓取/接触、相对高度、稳定时间和失败计数。采集
   随机初态及刻意扰动后的恢复示范，并在未见扰动上评估。
3. **P2：重做奖励调制更新。** 当前 `UpdateWeight` 每一步先对同一状态的 8 个动作更新做
   z-score，再把最大权重归一到 1；在 action-clamp 的 one-hot credit 下，这会把奖励绝对
   幅值基本抹掉，并同时改变未执行动作。应先用小状态任务验证
   \(\Delta w_{s,a}=\eta\,\delta_t e_{s,a}\)：只更新实际执行的 \((s,a)\)，episode 内
   衰减 eligibility，\(\delta_t\) 使用 reward baseline 或 critic 的 TD error；随后才重新
   接回 D1/D2 双通路。
4. **P3：固定神经--物理时间尺度。** 一个控制决策应对应固定数量的 SNN 步和固定动作
   保持时间，而不应因 PM 首次发放而在 3--30 步之间变化；输出可用固定窗口 spike count
   加 WTA，而非“首次单脉冲即停止积分”。

因此当前结论仍是：Lift 的全通信集成、教师基线、活动/逻辑 spike 测量和多个失败诊断均
已建立，但尚无稳定 PM 自主成功率，不能进入 RRR 或通信收益比较阶段。下一轮应先按 P0--P2
建立可分别验收的动作接口、option/恢复数据和三因子 actor--critic 学习基线。

### 13.8 局部三因子 TD 更新的首个对照（2026-07-23）

按 13.7 的 P2 建议，在原始 64 状态、全通信、相同 20 个教师预训练 + 10 个无教师 PM
测试协议下，新增 `--plasticity-rule three_factor`。它保留 BDM-SNN 的前向脉冲和 D1/D2
通路，但不用原 `UpdateWeight` 的行内 z-score 与最大值归一化；改为表格 critic 的 TD 误差
\(\delta_t=r_t+\gamma V(s_{t+1})-V(s_t)\)，且只改变环境实际执行的一个突触：
\(\Delta w^{D1}_{s,a}=\eta\delta_t e^{D1}_{s,a}\)，D2 使用相反调制。

最小 smoke 在教师全接管下 4/4 成功，无 NaN；完整对照的教师期为 16/20 成功，最后 10
个无教师 episode 仍是 **0/10 Lift、0/10 抓取**。不过自主平均学习回报为 -6.26，优于
动作持续状态旧更新对照的 -15.31，说明保留 TD 幅值确实改变了动作/回报分布；这不是任务
成功证据。代价是 PM 静默升至平均 37.6/180 决策，且动作显著偏向 `-z`，说明单独替换
突触更新尚未解决 PM 读出动力学和长时程操作阶段问题。

完整指标位于 `results/lift_bdm_snn_three_factor_pretrain20_autonomous10/`。此实现是一个
**局部三因子更新诊断**，不是已调优的 actor--critic BDM-SNN：下一步应先在 P1 的小型
可观测 option 状态机中固定动作保持时间、限制每阶段合法动作，再系统扫描 TD/eligibility
时间常数和 D1/D2 调制符号；只有获得多 seed 的自主 Lift 后，才能讨论 RRR。

### 13.9 可观测 option 状态机：教师仅训练期存在（2026-07-24）

为验证“教师离线传授、部署时只运行 SNN 与在线学习”的路径，新增 `--option-context`。
部署期的 `LiftOptionController` 不是教师：它只用末端--方块相对位置、抓取检测与有界计数，
在 FPGA/控制逻辑可实现的有限状态机中维护 `align`、`descend`、`close`、`lift`、`recover`
五个 option，并把 option 编入 SNN 状态（\(64\times5=320\)）。它只执行两类轻量职责：

1. 根据可观测前后条件切换阶段，并在每阶段屏蔽物理上不合理的动作；例如下降阶段只允许
   `-z`，闭夹阶段只允许 `gripper_close`，抬起阶段只允许 `+z` 并保持夹爪闭合。
2. 教师概率为零后绝不提供动作标签；PM 仍在当前 option 的允许动作中读出，D1/D2 与 TD
   critic 仍持续在线更新。

教师只在前 20 个 episode 产生示范动作。在训练期使用 `behavior_clone` 将示范动作显式写入
D1/D2 读出（这不是纯 R-STDP，必须单独标注），并同时运行局部三因子 TD 更新。4 episode
教师 smoke 为 4/4 Lift 成功，且每 episode 的冲突示范状态仅 1 个，说明 option 上下文确实
降低了隐藏阶段造成的标签冲突。

正式测试为每个 seed 前 20 回合教师、后 10 回合 `teacher_decisions=0` 的 PM 自主执行；
三 seed 结果如下：

| seed | 教师期成功 | 自主成功 | 自主抓取 | 自主阶段覆盖 |
| ---: | ---: | ---: | ---: | --- |
| 0 | 15/20 | 1/10 | 1/10 | 唯一成功回合经历 align 81、descend 56、close 2、lift 8 决策。 |
| 1 | 15/20 | 0/10 | 0/10 | 1,800 个自主决策均停在 align。 |
| 2 | 15/20 | 0/10 | 0/10 | 1,800 个自主决策均停在 align。 |
| 合计 | 45/60 | **1/30 (3.3%)** | **1/30** | 尚不稳定。 |

目录为：

- `results/lift_bdm_snn_option_clone_threefactor_pretrain20_autonomous10/`
- `results/lift_bdm_snn_option_clone_threefactor_seed1_pretrain20_autonomous10/`
- `results/lift_bdm_snn_option_clone_threefactor_seed2_pretrain20_autonomous10/`

这组实验给出一个重要但有限的正向证据：无需部署教师，`SNN PM + 持久 option 逻辑 + 在线
三因子更新` 曾完成一次完整 Lift 闭环；但 3.3% 远不能称为调通或稳定复现。主要失败模式是
PM 在 `align` option 的四个方向间未形成足够稳定偏好，导致无法进入下降，而非后续闭夹/上抬
阶段失败。下一步应只针对对齐 option 进行可验证改造：提高其几何状态分辨率或采用连续
population 编码，固定 PM 的 spike-count 决策窗口并测量动作一致率；完成多 seed 稳定自主
成功前，继续禁止加入 RRR、视觉和通信收益声明。

### 13.10 对齐阶段增强：xy 网格有效，固定 PM 窗口无增益（2026-07-24）

针对 13.9 中“几乎全部回合停在 align”的失败模式，保持 320 个 SNN 输入状态和相同的
教师训练/部署无教师协议，进行了两项可解释消融。

1. **对齐 xy 网格。** 仅在 `align` option，用相对目标点的 \(5\times5\) xy 误差网格
   取代原来的八象限/距离/夹爪组合码；其他四个 option 仍使用原状态码。因此没有增加输入
   神经元数或 DLPFC--纹状体阵列尺寸，只提高了最困难对齐阶段的横向方向和误差幅度分辨率。
2. **固定 PM 窗口。** 可选地将每个控制决策固定积分 10 个 SNN 步，并在整个窗口上累计
   PM spike count 后 WTA；这与原先“至少 3 步、首次 PM 发放则停止、最多 30 步”的自适应
   窗口构成单独对照。

在 seed 0 上，`xy 网格 + 固定 10 步窗口` 自主仍为 1/10，平均 PM 静默从 66.2 上升至
109.3/180，且每回合约 1,759 个内部 SNN 步；固定窗口增加了控制/计算延迟但未提高成功。
因此不将它作为后续默认。相反，仅启用 xy 网格并保留原自适应 PM 窗口后，三个 seed 的
无教师自主结果如下：

| 版本 | seed 0 | seed 1 | seed 2 | 合计自主成功 | 合计下降 option 决策 |
| --- | ---: | ---: | ---: | ---: | ---: |
| 原 option + 教师示范 + TD | 1/10 | 0/10 | 0/10 | 1/30 (3.3%) | 30 |
| xy 网格 + 自适应 PM 窗口 | 2/10 | 2/10 | 0/10 | **4/30 (13.3%)** | **302** |

网格版本自主平均回报为 -3.42（原 option 为 -6.86），每个自主成功均经过实际的
`align -> descend -> close -> lift` 选项序列。结果目录为：

- `results/lift_bdm_snn_option_grid_seed0_pretrain20_autonomous10/`
- `results/lift_bdm_snn_option_grid_seed1_pretrain20_autonomous10/`
- `results/lift_bdm_snn_option_grid_seed2_pretrain20_autonomous10/`
- `results/lift_bdm_snn_option_grid_window_seed0_pretrain20_autonomous10/`（固定窗口负对照）

结论应限于：提升 align 的几何编码分辨率，是目前唯一在多 seed 上同时提高阶段覆盖和稀疏
Lift 成功的改动；它不是对 RNN/CNN 或通信压缩的替代，也没有使策略稳定。下一步应在不扩大
成稀疏大查表的前提下，将 5x5 单热网格替换为紧凑连续 population 编码，或让 SNN 对四个
对齐方向输出显式的动作置信度/重复持续时间；每一项仍需与当前 13.3% baseline 作多 seed
对照。

### 13.11 对齐动作保持：仅在 PM 不确定时保持会恶化控制（2026-07-24）

继续针对 align 阶段，测试了 `--align-action-persistence`：部署时不引入教师，仅当 PM
静默或多个合法方向并列时，轻量控制逻辑在**同一 xy 网格**内最多保持上一横向动作 4 个
决策；一旦 PM 有唯一赢家或网格改变，立即重新接受 PM 选择。其本意是避免单步随机
tie-break 造成左右抖动，且 JSON 单独记录 `align_persistence_decisions`，避免把保持误记为
PM 独立选择。

seed 0 的 20 个教师 + 10 个无教师测试结果为：教师期 15/20 成功；自主期 **0/10**、
0 次抓取、0 次进入 descend，平均回报 -11.63。自主 1,800 个 align 决策中，保持逻辑实际
介入 234 次，却使原 xy 网格版本的 2/10 成功退化为 0/10。

这说明“PM 不确定”不等于“上一动作正确”：PM 静默/并列时首个回退动作本身常是随机方向，
将它连续执行会使末端离开目标并减少下一次纠错机会。故不采用该规则，也不进行多 seed
复验。后续若要学习动作持续时间，必须以可验证的证据驱动，例如 PM 对某方向的显式
置信度差值、或执行后 xy 误差确实降低；不能仅用不确定性作为保持信号。完整结果位于
`results/lift_bdm_snn_option_grid_hold_seed0_pretrain20_autonomous10/`。

### 13.12 进度验证的短期方向记忆：跨 seed 提升至 30%（2026-07-24）

13.11 表明仅因 PM 不确定就重复动作会放大随机错误。为保留“连续动作”但不引入教师，新增
`--align-progress-persistence`：部署期每次实际执行横向动作后，轻量逻辑只比较执行前后的
可观测 xy 误差。若误差至少下降 \(10^{-4}\)，将该方向登记为最多可复用 4 个决策的短期
记忆；以后 PM 静默或并列时才允许复用该**已由物理进度验证**的方向。若该方向不再降低
误差，记忆立即失效。它不读取教师动作、不使用目标方向标签，也不覆盖有唯一赢家的 PM 输出；
每个实际复用动作在 JSON 中记录为 `align_progress_persistence_decisions`。

在相同的前 20 个教师 episode、后 10 个完全无教师 PM 自主 episode、320 状态全通信条件下：

| 版本 | seed 0 | seed 1 | seed 2 | 合计 Lift | 合计抓取 | 下降 option 决策 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| xy 网格（无方向记忆） | 2/10 | 2/10 | 0/10 | 4/30 (13.3%) | 4/30 | 302 |
| 随机/并列动作保持 | 0/10 | 未复验 | 未复验 | 0/10 | 0/10 | 0 |
| xy 网格 + 进度验证方向记忆 | **3/10** | **1/10** | **5/10** | **9/30 (30.0%)** | **10/30** | **635** |

该版本的 30 个自主 episode 平均学习回报为 -0.095（网格无方向记忆为 -3.42），共发生 802
次进度验证后的动作复用。结果位于：

- `results/lift_bdm_snn_option_grid_progresshold_seed0_pretrain20_autonomous10/`
- `results/lift_bdm_snn_option_grid_progresshold_seed1_pretrain20_autonomous10/`
- `results/lift_bdm_snn_option_grid_progresshold_seed2_pretrain20_autonomous10/`

因此目前有跨 seed 的证据表明：将“已产生可观测进度的动作方向”作为短期状态/记忆，比
把 PM 静默简单视为保持理由有效得多。它仍是部署期需要实现的轻量反馈状态，不是离线教师；
而且 30% 成功和 seed 1 的 1/10 都表明策略尚不稳定，不能开始 RRR。下一阶段应将该一位
外部短期记忆显式表示为小型 SNN 状态/神经元群，并在固定与随机初态、多 seed 条件下将
成功率提高到可接受门槛，再独立评估压缩通信。

### 13.13 将进度记忆直接编码为 SNN 状态：离散组合爆炸的负对照（2026-07-24）

为检验 13.12 的短期方向记忆能否不再覆盖 PM 动作、而是由 SNN 自行利用，新增
`--progress-memory-context`。每次横向动作确实降低 xy 误差后，环境只维护一个可观测 token：
`none/+x/-x/+y/-y`。该 token 与 5x5 align 网格组合为 \(25\times5\) 个 align 状态并输入
DLPFC；其他四个 option 保持原状态，输入总规模由 320 增至 640。自主期**不启用**
`--align-progress-persistence`，即 token 不会直接重写 PM 动作，物理动作完全由 PM 加 option
合法动作门控决定。

教师 smoke 4/4 成功，说明接口无数值或控制错误；但完整 seed 0 的 20 教师 + 10 无教师
测试为 **0/10 Lift、0/10 抓取、0 次进入 descend**，平均回报 -7.02。此结果不是“进度
token 无用”的充分证据，而是当前 one-hot 查表式 BDM 输入缺少组合泛化：教师时绝大多数
样本是 `memory=none`，自主第一次取得进度后输入变为未充分示范的 `(xy_grid, memory_action)`
组合；其 D1/D2 权重未被克隆/TD 覆盖，PM 重新变为静默或并列，最终停留在 align。结果位于
`results/lift_bdm_snn_option_memorycontext_seed0_pretrain20_autonomous10/`。

这个负对照与 13.12 的 9/30 共同给出下一步推导：短期记忆确有控制价值，但不能通过把所有
几何--记忆组合扩成稀疏 DLPFC 查表来“内化”。后续应构造**因子化共享表示**：保持 25 个
xy 感知神经元和 5 个进度 token 神经元为并列群体，分别投向同一小型动作/纹状体群，使相同
方向 token 能跨 xy 格共享连接；或使用连续 population code。先以该因子化全通信 baseline
验证多 seed 自主率，再考虑把 token/感知群作为可压缩跨核通信变量。

### 13.14 因子化并列神经元：当前克隆规则下仍失败（2026-07-24）

在 13.13 后实现了更紧凑的 `--factorized-progress-memory`，避免 \(25\times5\) 组合查表：
align 时 DLPFC 同时激活一个 xy 网格神经元（25 个之一）和一个进度 token 神经元（
`none/+x/-x/+y/-y`，5 个之一）；其余四个 option 使用 256 个基础神经元，合计 286 个输入
神经元，低于原 option 的 320 和组合查表的 640。教师监督、action-clamp eligibility 与
三因子更新会分别作用到两个活跃输入的状态--动作行；自主期不允许该 token 覆盖 PM 输出。

4 episode 教师 smoke 为 4/4 成功，输入/更新接口无异常；但完整 seed 0 运行再次得到
**0/10 Lift、0/10 抓取、0 次 descend**，平均回报 -10.12。该负结果指出，单纯“共享
token 行”并不自动获得组合泛化，当前结构至少有两个具体问题：

1. 现有 `behavior_clone` 对每个活跃 DLPFC 行都覆写完整 8 动作偏好。xy 行应编码“此位置
   选什么动作”，但共享 token 行只应表达“最近哪个方向曾带来进度”；当同一 token 出现在
   不同 xy 格、教师下一步需要改轴或反向时，token 行收到冲突的完整动作标签，互相覆盖。
2. 两个感知神经元以相同电流同时激活，改变了原 one-hot BDM-SNN 的纹状体/STN 前向电流
   尺度和竞争平衡。因子化不是无代价的状态重排；其 logical DLPFC source events 在 smoke
   中也从单一状态的每步 1 个变为 align 时 2 个，不能据“286 小于 640”声称通信减少。

结果目录为 `results/lift_bdm_snn_option_factorizedmemory_seed0_pretrain20_autonomous10/`。
结合 13.12--13.14，失败案例的主因已更明确：多数自主失败首先因 PM 静默/并列与未充分
泛化的对齐方向偏好而无法越过 align 阈值；即使 token 已产生，当前查表/克隆规则也不能将
它作为条件性辅助信息稳定利用。其次，少数已抓取失败是闭夹接触和上抬确认不足。

因此下一步不应继续增加离散组合状态，也不应让记忆 token 直接预测完整动作。合理的下一版
应是**条件性偏置路径**：空间 xy 行保留完整动作学习；token 仅在“其方向与当前 PM 候选
相同”时增加受限、可学习的 D1 偏置，或通过小型门控突触调制该方向，且其更新只来自实际
xy 进度而非教师完整动作标签。还需归一化多输入总电流。先在固定初态、多 seed 下比较
“无 token / 外部动作复用 / token 条件偏置”三种全通信版本，确认 PM 自主成功率和 PM
覆盖稳定后，才能把这个小 token 群讨论为多核通信与 RRR 的候选变量。

### 13.15 条件性进度方向偏置：内部化尝试未超过外部短期记忆（2026-07-24）

为避免 13.12 中短期记忆在 PM 不确定时直接替换物理动作，新增
`--progress-direction-bias`，并保持 320 个状态的 `option + 5x5 xy` 全通信基线。它由两条
彼此独立的轻量状态组成：

1. 每次真实执行横向原语 \(a\in\{+x,-x,+y,-y\}\) 后，计算可观测横向误差
   \(d_{xy}=\lVert(p_{cube}-p_{eef})_{xy}-(-0.02,0)\rVert\)。局部标量
   \(b_a\) 按 \(b_a\leftarrow\mathrm{clip}[b_a+\eta\,\mathrm{clip}((d_{before}-d_{after})/0.01,-1,1)]\)
   更新，范围限制为 \([-0.5,0.5]\)。因此其学习信号是**执行后的物理进度**，不是教师的
   动作标签；教师只在前 20 个训练 episode 存在。
2. `AlignProgressMemory` 仍只保存最近一次确实减小 \(d_{xy}\) 的方向 token 及其有限
   有效期。自主期先用**未修改的** PM spike count 判断 PM 是否静默或并列；只有这一原始
   PM 读出不确定时，才把 token 对应的正 \(b_a\) 加到该方向的读出分数以打破平局。若 PM
   已有唯一赢家，偏置绝不推翻该 SNN 决策。JSON 的 `progress_bias_decisions` 只统计这种
   实际介入，而不把“最终动作恰好同向”误记为介入。

这相当于一个受限的、进度调制的读出门控：几何 xy 输入仍负责学习完整的空间--动作对应，
记忆 token 只提供“刚才哪个方向在当前物理闭环中有效”的小幅证据。它不是 RRR、不是跨核
通信压缩，也没有把 token 扩张为组合 DLPFC 查表状态。

在固定方块、确定性 Panda 初态、seed 0、前 20 个教师 episode、后 10 个
`teacher_decisions=0` 的协议下，结果为：

| 版本 | 自主 Lift | 自主抓取 | 平均学习回报 | 平均 PM 静默/180 | 偏置实际介入 | align/下降/闭夹/抬升决策 |
| --- | ---: | ---: | ---: | ---: | ---: | --- |
| xy 网格 + 进度验证动作复用（13.12，seed 0） | 3/10 | 未单独汇总 | -- | -- | 动作复用规则 | -- |
| xy 网格 + 条件性方向偏置 | **2/10** | **2/10** | **-2.418** | **59.8** | **194** | **1575 / 139 / 4 / 14** |

两个成功回合均经历 `align -> descend -> close -> lift`，说明一旦对齐阈值被跨越，后续
option 与低层控制接口可以完成闭环；但 8/10 失败回合仍完全停留在 align。尤其 episode 21
虽进入下降 29 次却未抓取，表明第二类失败是下探时的接触/夹取时机仍不鲁棒。完整结果位于
`results/lift_bdm_snn_option_progressbias_ambiguousonly_seed0_pretrain20_autonomous10/`。

此前一次未严格限制“仅 PM 不确定时介入”的原型也得到 2/10，但因它可能改写 PM 唯一赢家，
不作为可比较证据；上表仅使用语义修正后的结果。该修正同时说明本轮负结果不是由“偏置过度
覆盖 SNN”造成的表述混淆。

本次失败的主因不是数值崩溃，而是**偏置的证据形式仍不足以解决连续闭环控制**：它只携带一
个方向和一个全局小标量，不包含当前位置到目标尚差多少、该方向还应持续多久、以及何时应
切换横纵轴；而 PM 仍频繁静默/并列。把这种弱证据从“直接复用已验证动作”改成温和 tie-break
后，纠偏能力反而下降，因而没有超过 13.12 的 seed 0 的 3/10 结果。其次，进入 descend 的
少数失败说明闭夹前的高度/接触确认也需要显式状态，而不能只依赖固定次数的下降动作。

由此推导的下一步不应继续堆叠偏置或立即测试 RRR，而应先构造一个**可观测的连续对齐--接触
记忆状态**：以相对 xy 误差的连续 population code（或误差大小档 + 方向）表示“尚余距离”，
再以 PM 的方向置信度和误差单调下降共同决定动作持续时间；同时在 descend 前加入相对高度、
接触/抓取确认与失败恢复状态。该状态应通过独立输入群或受限门控连接到 PM，而不是对每个
token 写入完整行为克隆标签。验收顺序是先用同一低层接口验证该状态机/教师重放在随机初态
可稳定成功，再让 SNN 在多 seed、无教师条件下学习；只有这一全通信自主 baseline 稳定后，
才将其输入群作为 RRR 与事件通信实验的候选跨核变量。

### 13.16 奥卡姆审查：两个最小物理/状态改动均未优于原始均匀网格（2026-07-24）

本轮没有继续叠加 token、偏置、记忆网络或新的学习规则，而是回到当前最有效且最简单的
全通信版本：**5x5 均匀 xy 网格 + option 合法动作门控 + 仅在真实 xy 误差下降后才短暂复用
方向**（13.12）。对 seed 0 的原始基线而言，20 个教师 episode 后的 10 个无教师自主
episode 为 3/10 Lift。两项逐一、其余参数不变的最小对照如下：

| 仅改变的一项 | 第一性原理动机 | 自主 Lift / 抓取 | 主要现象 | 结论 |
| --- | --- | ---: | --- | --- |
| 将 5x5 网格中心格收窄为 option 的 2 mm 下探阈值 | 原均匀中心格宽 20 mm，可能混淆“已可下探”和“仍需微调” | 1/10 / 2/10 | `align` 1,555 次、下降 159 次 | 退化；中心变窄会使相邻格扩至约 48 mm，损失更常见接近过程的几何分辨率。该开关已从代码移除。 |
| 横向/竖向原语幅度由 0.25 降为 0.10 | 实测 0.25 命令首步移动约 1.2--1.3 mm，2 mm 阈值附近或有过冲 | 0/10 / 0/10 | 虽有 255 次下降，但 0 次闭夹 | 退化；同一个缩小幅度也使单步下降不足，教师期已由 15/20 成功降至 0/20，说明不能不分阶段地缩小全部动作。 |

第二项还给出一个关键的控制接口结论：物理原语不只是“探索步长”。`descend` 和 `lift` option
各只允许一个固定方向动作，故它们的成功依赖该原语在有限决策预算内达到接触/抬升位移；把
xy 微调所需的更小位移盲目施加给 z 轴，会直接破坏原本可行的教师轨迹。

因此本轮按奥卡姆原则**保留**：原始均匀 5x5 网格、0.25 的统一原语、option 门控、局部
三因子更新与进度验证方向复用；**去除/不采用**：中心接触格、全局缩小原语、组合 token、
因子化 token、条件偏置和固定 PM 窗口。这里“保留”表示当前固定初态基线中证据最强，不表示
这些机制已经足够稳定。

失败主因也因而收敛为一个更具体的问题：在 `align` 时，方向记忆只能告诉控制器“上一次哪
个方向带来了进度”，但不表示剩余误差大小，也不能区分粗接近与 2 mm 内的微调；而动作幅度
又必须按阶段区分，不能简单全局缩小。下一步最小且有针对性的改动应是**保持 8 个离散动作和
320 个 SNN 状态不变，仅让同一 xy 动作在接近阈值时使用较小的物理幅度**，z 轴与夹爪动作仍
保持 0.25。这是控制原语的状态依赖执行参数，不增加 SNN 输出维度、教师标签、记忆 token 或
跨核通信；先以教师重放验证其不损害 20/20 左右的可行轨迹，再做相同三 seed 的无教师 PM
对照。若仍无提升，应停止微调 SNN 读出，转而单独验证低层 controller 的可控性和接触条件，
而非进入 RRR。

### 13.17 分阶段细步长与学习/示范诊断：均未形成跨 seed 提升（2026-07-24）

按照 13.16 的推导，本轮首先只在 `align` 且 xy 误差不超过 12 mm 时，将横向命令幅度从
0.25 切为 0.10；z 轴、夹爪、8 个 SNN 动作、320 个状态、option、在线三因子更新和进度
方向记忆均不变。独立教师检查达到 **20/20 Lift**，说明该版本没有像“全局 0.10”一样破坏
下降/抬升接口。无教师自主结果为：

| 版本 | seed 0 | seed 1 | seed 2 | 合计 |
| --- | ---: | ---: | ---: | ---: |
| 原始 0.25 + 进度方向记忆 | 3/10 | 1/10 | 5/10 | **9/30 (30.0%)** |
| 近目标 xy=0.10，其余仍 0.25 | 4/10 | 3/10 | 1/10 | **8/30 (26.7%)** |

细步长在 seed 0/1 改善，却使 seed 2 从 5/10 降为 1/10；总体没有提升，因此不能据单 seed
将其设为新基线，相关开关已从主入口移除。结果位于：

- `results/lift_bdm_snn_option_grid_progresshold_finecontrol_teacher20_seed0/`
- `results/lift_bdm_snn_option_grid_progresshold_finecontrol_seed0_pretrain20_autonomous10/`
- `results/lift_bdm_snn_option_grid_progresshold_finecontrol_seed1_pretrain20_autonomous10/`
- `results/lift_bdm_snn_option_grid_progresshold_finecontrol_seed2_pretrain20_autonomous10/`

随后用三个单变量诊断检查“为什么教师期之后仍不稳定”：

| 诊断（seed 0） | 教师 Lift | 自主 Lift / 抓取 | 解释 |
| --- | ---: | ---: | --- |
| 正常在线三因子更新 | 15/20 | 3/10 / 3/10 | 当前最简对照。 |
| 自主期冻结全部突触更新 | 15/20 | **0/10 / 0/10** | 在线 TD 不是单纯冲掉教师知识；相反，它对补齐未示范轨迹是必要的。 |
| 教师期方块位置均匀抖动 ±15 mm，测试仍固定 | 13/20 | **2/10 / 3/10** | 示范覆盖增加，但同一粗网格内的动作/接触差异也增加；教师本身更不稳定。 |
| 同状态教师动作采用历史频率软投票，替代最后标签覆盖 | 15/20 | **1/10 / 3/10** | 保留冲突动作却没有提供选择冲突动作所需的连续几何信息，PM 偏好反而变弱。 |

对应目录为：

- `results/lift_bdm_snn_option_grid_progresshold_frozenautonomous_seed0_pretrain20_autonomous10/`
- `results/lift_bdm_snn_option_grid_progresshold_teacherjitter015_seed0_pretrain20_autonomous10/`
- `results/lift_bdm_snn_option_grid_progresshold_clonevotes_seed0_pretrain20_autonomous10/`

这些负对照共同否定了三个简单解释：失败不主要来自统一动作幅度、在线更新遗忘或教师样本
数量不足。更根本的瓶颈是**表示不可辨识**：一个 20 mm 宽的 xy 网格单元可包含不同误差符号、
不同主轴和不同剩余距离，教师在同一状态内自然会给出不同动作；冻结、增加示范或对冲突标签
取平均都无法恢复输入中没有的信息。进度方向记忆能把成功率提高到 30%，是因为它利用了动作
执行后的真实误差变化，临时补充了一位闭环信息，但仍不能完整表示当前位置。

因此继续扫描学习率、步长或保持次数已经不符合奥卡姆原则。下一步应只替换 `align` 的输入
表示，保持其余已验证模块不动：用少量重叠的 x/y 误差 population 神经元分别编码符号与幅值，
并将两个群的总输入电流归一化；监督只更新对应轴的方向动作，避免共享 token 被完整 8 动作
标签覆盖。它的首要验收不是通信维数，而是：教师标签冲突下降、PM 唯一赢家比例上升、三 seed
无教师成功率超过 9/30。达到这些条件前仍维持全通信，不进入 RRR 或硬件收益比较。

### 13.18 x/y 对齐 population 编码：轴分解本身不能替代位置绑定（2026-07-24）

为直接检验 13.17 的“表示不可辨识”判断，新增并测试了一版只替换 `align` 输入的全通信
population 编码。原 25 个 xy 组合单热格替换为 10 个可复用的感知细胞：x 和 y 各有 2 个
符号细胞（正/负）与 3 个绝对误差幅值细胞（\(\leq10\) mm、10--40 mm、\(>40\) mm）。
每个对齐决策激活 x 的“符号+幅值”两格和 y 的“符号+幅值”两格；输入电流归一化为总和 2.0，
与单热基线一致。教师的 `+x/-x` 监督和三因子更新只写入 x 群，`+y/-y` 只写入 y 群，避免
一个轴完整覆写另一个轴的 8 动作偏好。其余动作、option、教师训练期、在线 TD、进度方向
记忆和物理控制均未改变。

教师 smoke 为 4/4 Lift、无 NaN；完整 seed 0 的 20 个教师 + 10 个无教师结果如下：

| align 编码 | 自主 Lift / 抓取 | 平均学习回报 | PM 静默/180 | 结论 |
| --- | ---: | ---: | ---: | --- |
| 原 5x5 组合单热 + 进度方向记忆 | 3/10 / 3/10 | 0.171 | 62.2 | 当前最佳 seed 0 对照。 |
| x/y 四细胞均分电流 | 2/10 / 3/10 | -10.149 | 24.1 | 两轴即使误差不等仍提供等强证据，动作优先级不明确。 |
| x/y 电流按 \(|e_x|:|e_y|\) 分配 | 1/10 / 1/10 | -8.113 | 17.8 | 降低并列/静默不等于正确动作；共享单轴行仍缺少 x--y 组合绑定。 |

对应结果目录：

- `results/lift_bdm_snn_option_population_teacher_smoke_seed0/`
- `results/lift_bdm_snn_option_population_progresshold_seed0_pretrain20_autonomous10/`
- `results/lift_bdm_snn_option_population_weighted_teacher_smoke_seed0/`
- `results/lift_bdm_snn_option_population_weighted_progresshold_seed0_pretrain20_autonomous10/`

这里的 266 个输入状态数低于 320，且 population 的 DLPFC logical 活动方式与单热不同；没有
测量实际 AER 包、bit 或 FIFO，因此**不能**据此主张通信成本下降。该实验的价值仅在于排除
一个算法假设：仅将 x/y 误差拆成共享行、甚至按误差大小重加权，仍无法让目前“输入行直接到
动作偏好”的 BDM-SNN学习正确组合动作。第二版虽把平均 PM 并列数由 1.73 降至 1.25，却
同时把成功降为 1/10，说明“更确定”可能只是更确定地选择了错误方向。

因此本轮按奥卡姆原则不保留 population 开关；主入口恢复为 5x5 网格全通信 baseline。与其再
微调编码边界或电流比例，更合理的下一步是承认结构限制：若要利用 x/y 可组合信息，需要一个
显式的**双线性交互/局部 conjunctive 层**，使“x 符号--幅值”和“y 符号--幅值”在进入动作
读出前形成有限的结合单元；或保持 25 个组合状态但采用局部连续特征的邻域泛化。两者都会改变
网络结构，必须先在单独的小型二维到目标任务中验证其是否确实减少标签冲突和提升无教师闭环，
再接入 Lift；在此之前继续把复杂结构直接塞入 Lift 不利于归因，也仍不进入 RRR。

### 13.19 独立二维结合与邻域探针：定位为表示--学习规则联合限制（2026-07-24）

为避免继续在 Lift 的接触、option、夹爪与连续控制误差中混淆原因，新增两个独立诊断入口：

- `planar_binding_probe.py`：5x5 离散二维点走向原点，四个方向动作；教师先选择绝对误差较大
  的轴。比较组合查表、共享 x/y 轴行、以及显式 x-y conjunction 行。
- `planar_bilinear_probe.py`：教师与测试点均为连续二维位置；比较最近硬分箱与最多四个相邻
  conjunction 单元的双线性激活，测试邻域泛化是否能替代硬分箱。

二者都保留 BDM-SNN 前向、PM 读出和训练期逐行 D1/D2 行为克隆；它们不是 RRR、硬件或通信
实验。离散探针以完整遍历 25 个教师起点两轮、三 seed 测试全部 25 个起点，结果为：

| 表示 | 输入行数 | 教师冲突更新 | 自主到达原点（25 起点） | 解释 |
| --- | ---: | ---: | ---: | --- |
| 25 个组合查表 | 25 | 0 | 16 / 17 / 14 | 每个位置--动作绑定明确，但 PM 本身仍有脉冲随机性。 |
| 10 个共享 x/y 轴行 | 10 | 20 | 3 / 3 / 3 | 同一轴特征在不同另一轴位置对应不同下一动作，完整动作标签必然冲突。 |
| 10 个轴行 + 25 个显式 conjunction 行 | 35 | 0 | 16 / 17 / 14 | conjunction 行恢复了结合信息，结果与原 25 个组合查表等价，没有额外性能收益。 |

连续探针的三 seed 结果更严格地否定了“直接加邻域激活即可泛化”：最近硬分箱每 seed 都只成功
1/25（起点就是原点），教师冲突为 7--9；双线性激活同样 1/25，冲突反而为 27。双线性降低了
PM 静默/并列，但没有提高行为正确性。这是因为现有克隆会将**每个**被激活的邻居行覆写成当前
完整教师动作；连续决策边界两侧的样本共享邻居后，冲突从状态表示转移到权重写入过程。

结果位于 `results/planar_binding_probe/`，典型文件包括：

- `lookup_sweep2_seed0.json`、`axis_sweep2_seed0.json`、`conjunctive_sweep2_seed0.json`
- `lookup_bilinearprobe_seed0.json`、`bilinear_bilinearprobe_seed0.json`

这轮给出了一个比“再加神经元”更具体的结论：问题是**表示与学习规则的联合限制**。组合状态能
消除标签冲突，但缺少连续泛化；把邻近/轴特征共享给多个状态后，当前“整行覆写成一个动作”的
监督规则又会破坏共享。下一步若要继续，必须先在这个二维探针中把克隆改成真正的、按输入激活
强度分配的局部增量三因子更新（而非行覆写或简单频率平均），并以连续二维成功率超过硬分箱为
验收；若该探针仍不能提升，应停止将此类结构引入 Lift，转而将当前 30% 全通信 baseline 作为
已知下界，并重新定义更适合 BDM-SNN 结构的 embodied 任务。无论哪种情况，在 baseline 稳定
之前都不进入 RRR 或通信收益宣称。

### 13.20 二维局部增量监督：降低 PM 不确定性仍未恢复正确闭环（2026-07-24）

为验证 13.19 中“行覆写破坏邻域共享”的假设，仅在连续二维探针中新增
`--learning-mode incremental`：不再将每个活跃行重置为单一教师动作，而按其当前输入激活强度
以小步长向教师 D1/D2 目标靠近。双线性邻居的权重越大，获得的局部教师增量越大；Lift 主线
未改动。连续二维教师与测试、120 条教师轨迹、三 seed 的结果如下：

| 表示 + 学习规则 | 连续测试成功（25 起点） | 教师冲突 | 平均 PM 静默 | 平均 PM 并列 |
| --- | --- | --- | ---: | ---: |
| 最近硬分箱 + 行覆写 | 1 / 1 / 1 | 9 / 7 / 8 | 6.44 / 6.48 / 6.04 | 2.86 / 2.81 / 2.69 |
| 双线性邻域 + 行覆写 | 1 / 1 / 1 | 27 / 27 / 27 | 3.24 / 3.72 / 3.40 | 1.95 / 2.08 / 2.00 |
| 最近硬分箱 + 局部增量 | 1 / 1 / 1 | 7 / 8 / 9 | 5.80 / 6.60 / 5.52 | 2.95 / 3.23 / 2.79 |
| 双线性邻域 + 局部增量 | 1 / 1 / 1 | 27 / 27 / 27 | 0.44 / 0.36 / 0.60 | 1.22 / 1.17 / 1.31 |

双线性局部增量确实把 PM 静默和并列显著压低，却没有提高正确到达率；这排除了“只要不用整行
覆写，邻域泛化就会自动解决”的假设。现有实现中，BDM-SNN 的多阶段脉冲竞争、有限积分窗口和
随机 tie-break 仍可把较确定的读出变成较确定的错误动作。换言之，**PM 不确定性是指标而非任务
目标**；不能用它代替闭环成功率。

该结果位于 `results/planar_binding_probe/lookup_incremental_seed*.json` 与
`results/planar_binding_probe/bilinear_incremental_seed*.json`。本轮到此停止将新的表示/学习规则
接入 Lift：二维最简任务未给出正向验证，继续扩展只会混淆结论。当前可复现的 Lift 结论仍是
13.12 的固定初态、全通信、进度验证方向记忆 `9/30`；它可以作为工程 baseline，但不能称作稳定
自主操作，更不能用于 RRR、AER 或 RRAM 通信收益结论。

若继续研究，合理的分叉应是：

1. 把 PM 脉冲读出本身替换为固定时间窗的概率/价值读出，并先在二维任务验证正确动作率；这改变
   BDM-SNN 决策机制，需与当前 PM readout 单独比较。
2. 保持当前 BDM-SNN，不再强行匹配连续机器人控制，转向离散状态--动作更匹配的具身导航/网格
   任务，先建立稳定全通信决策基础后再研究跨核通信。

这两个方向均应首先明确研究目标；在未选择前，不再继续无依据地堆叠编码或辅助规则。

### 13.21 统一统计口径与 PM--丘脑读出消融（2026-07-25）

本节先澄清前面不同分母的含义，避免把它们误作同一种成功率：

| 指标 | 分母是什么 | 用途 | 不能推出什么 |
| --- | --- | --- | --- |
| Lift 的 `x/10` | 一个 seed 下固定初态的 10 个完整无教师 episode | 评估同一训练权重在重复机器人闭环中的稳定性 | 不能说明已覆盖全部连续几何状态。 |
| Lift 的 `x/30` | 三个独立 seed 各 10 episode 的合计 | 当前工程 baseline 的多 seed 汇总 | 仍不是随机初态的泛化成功率。 |
| 二维离散的 `x/25` | 枚举 (5\times5=25) 个离散起点各一次 | 检验**全部已训练的状态--动作绑定**能否无教师闭环执行 | 不是 25 个随机 episode，也不能直接和 Lift 百分比横比。 |

因此，二维任务要求接近 `25/25` 是一个刻意严格的**单元测试验收**：教师已经完整遍历过这
25 个位置且组合查表没有标签冲突；若仍到不了原点，原因必定在 SNN 动力学/读出，而非未见状态、
接触或连续控制。它不是声称 Lift 必须达到 25 次成功的门槛。

在该最小任务中新增 `planar_pm_readout_probe.py`，固定两轮完整教师遍历（120 个训练决策），
部署阶段 `deployment_teacher_decisions=0`，保持 DLPFC--StrD1/StrD2 权重、基底节通路、STDP
前向与全通信监测不变，仅比较从哪一脑区的**已产生脉冲**累计读出动作：

| 读出 | 时间窗 | 覆盖 | 无教师闭环 | 结论 |
| --- | ---: | ---: | ---: | --- |
| PM spike count | 3 / 10 / 30 步 | 各 3 seed | 11--16 / 25、14--17 / 25、12--17 / 25 | 加长 PM 积分并未恢复正确闭环。 |
| D1--D2 已写权值差（诊断上界，非 SNN 部署） | -- | 3 seed | 25 / 25 | 教师将离散状态--动作表写入正确；问题不在该表。 |
| 丘脑 spike count | 1 步 | 6 seed | 6--15 / 25 | 窗口不足，脉冲证据尚不完整。 |
| 丘脑 spike count | 3 步 | 6 seed | 19--23 / 25 | 明显改善但仍有未完成轨迹。 |
| 丘脑 spike count | 10 步 | 10 seed | **25 / 25** | 平均逐决策正确率约 0.91--0.97。 |
| 丘脑 spike count | 30 步 | 6 seed | **25 / 25** | 逐决策正确率为 1.00，但控制延迟更大。 |

这给出一个比“PM 本身随机”更精确的机制定位。丘脑已经汇集了 GPi 抑制与 DLPFC 兴奋的动作
相关证据；而 PM 还要经过 `thalamus -> PM` 的阈值发放以及 PM--PM 侧向抑制。当前参数下，这个
最后阶段会把有用的时窗脉冲变成静默或并列。故本节的丘脑读出是**候选 thalamo-cortical action
decoder 的结构消融**，不是把教师、权值查表或外部策略偷偷带回部署，也不能再称为“纯 PM 输出”。

为检验二维结论能否迁移，在 Lift 入口加入 `--decision-readout pm|thalamus`：两种设置均运行
完整 BDM-SNN、相同 D1/D2 三因子更新、相同 option 和进度验证方向记忆；仅用固定 10 个内部
SNN 步累计的 PM 或丘脑脉冲选择物理原语。20 个教师 episode 后，后 10 个 episode 都满足
`teacher_decisions=0`。丘脑读出的三 seed 结果为：

| 动作读出 | seed 0 | seed 1 | seed 2 | 合计 | 无教师丘脑静默率 | 无教师 PM 静默率 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| 丘脑、固定 10 步 | 2/10 | 0/10 | 2/10 | **4/30** | 41.2% / 36.7% / 41.3% | 62.9% / 56.9% / 66.7% |
| 原 PM、自适应窗口 + 进度验证方向记忆（13.12） | 3/10 | 1/10 | 5/10 | **9/30** | 不适用 | 另一个协议，不能与固定窗的静默率直接横比 |

结果目录为 `results/planar_pm_readout_probe/` 与
`results/lift_bdm_snn_thalamus_window10_seed{0,1,2}_pretrain20_autonomous10/`。二维任务证明丘脑
读出能隔离并绕开最后 PM 动力学限制；但在 Lift，它只比历史上同为固定 PM 窗口的 seed-0 对照
略好（2/10 对 1/10），并且总体仍低于当前 9/30 工程 baseline。因此不以它替换 baseline，也不
进入 RRR：机器人残余失败还来自粗 5x5 状态下的动作标签冲突、横向动作持续时间、下降--接触--
夹取时机及在线更新漂移。

后续若继续，最小且可证伪的方向不是再调 PM 阈值，而是选择其一：(1) 将丘脑累计读出正式建模
为独立的 thalamo-cortical 输出层，并在**离散导航**中先建立多 seed 稳定 baseline；或 (2) 保留
PM 生物解释，但重新设计 `thalamus -> PM` 增益、阈值和侧抑制，使 PM 复现二维 25/25，再返回
机器人。两者都必须先完成全通信稳定性验证，之后才能研究 RRR/事件通信。

### 13.22 PM 脉冲读出动力学重构与教师覆盖审查（2026-07-25）

#### PM 的最小、可逆重构

13.21 表明丘脑在 10 步窗口已经具有正确的动作证据、而 PM 读出仍失败。因此没有新增教师、
外部分类器或权值查表，而是仅重构 PM 内部的两项 IF 动力学参数。原网络 PM 电流为

\[
I_{PM,j}(t)=s_{TH,j}(t)-2.5\sum_{i\ne j}s_{PM,i}(t),\qquad
s_{PM,j}(t)=H(v_{PM,j}(t)-0.5).
\]

这里的 `-2.5` 来自原始 `5 * weight_inh`；它是每个竞争 PM 神经元对其余三个神经元的强侧向
抑制。由于四个 PM 神经元起始近似对称、丘脑证据又是时间分散的，首个共同脉冲会同时对所有
候选注入较大负电流，结果往往是整个 PM 群静默或并列。重构版本保持 DLPFC--基底节--丘脑、
所有 STDP/三因子更新、PM 阈值与动作读出不变，仅把 PM 非对角抑制改为较弱的可配参数
`pm_lateral_gain`：

\[
I_{PM,j}(t)=s_{TH,j}(t)+g_{lat}\sum_{i\ne j}s_{PM,i}(t),\qquad g_{lat}\in\{-2.5,-1.25,-0.5,-0.25,-0.1,0\}.
\]

实现通过 `BDMSNN(..., pm_threshold=..., pm_lateral_gain=...)` 完成，默认仍为原始阈值 0.5 和
原始抑制 -2.5；Lift 入口也以 `--pm-threshold`、`--pm-lateral-gain` 显式记录配置。因此它是软件
SNN 的动力学消融，不是已验证的生物定量参数或硬件电路实现。

二维完整查表、两轮教师遍历、部署无教师、PM 自身 10 步 spike-count 读出的结果为：

| PM 阈值 | 非对角侧抑制 | seed 覆盖 | 无教师成功（25 起点） | 平均逐决策正确率 | PM 静默率 | 结论 |
| ---: | ---: | ---: | --- | ---: | ---: | --- |
| 0.15 / 0.25 / 0.35 | -2.5 | 各 3 | 14--18 / 25 | 0.456 | 0.435 | 单降阈值没有稳定改善，主因不是发放门槛。 |
| 0.5 | -1.25 | 3 | 16--21 / 25 | 0.534 | 0.312 | 减弱抑制开始减少静默。 |
| 0.5 | -0.5 | 10 | 22--24 / 25 | 0.654 | 0.069 | PM 已显著恢复，但仍有错误。 |
| 0.5 | -0.25 | 10 | 22--25 / 25 | 0.748 | 0.039 | 接近完整，但个别 seed 仍失败。 |
| 0.5 | -0.1 | 10 | **25 / 25** | 0.882 | 0.000 | PM 自身达到二维验收。 |
| 0.5 | 0.0 | 10 | **25 / 25** | 0.939 | 0.000 | 无 PM 横向抑制也可完成；说明原始抑制过强。 |

这不是“侧抑制没有作用”的结论：本任务只有四个已正确写入的离散动作，较弱或零抑制足以让
丘脑差异累计；在更大动作空间、噪声或时间竞争下仍可能需要校准的抑制。`-0.1` 而非 `0` 被
选作后续最小版本，因为它保留了竞争的符号，同时在二维 10 seed 中稳定 25/25。结果位于
`results/planar_pm_dynamics_probe/`。

把该参数直接带回 Lift，保持前 20 个教师 episode、后 10 个 `teacher_decisions=0`、固定方块、
确定性初态、相同 `option + 5x5 grid + progress persistence + three_factor`，并为与 13.21 严格
对齐而使用固定 10 内部步 PM 计数读出：

| PM 读出版本 | seed 0 | seed 1 | seed 2 | 合计 Lift | 无教师 PM 静默率 |
| --- | ---: | ---: | ---: | ---: | ---: |
| 原始侧抑制 -2.5，丘脑读出（13.21） | 2/10 | 0/10 | 2/10 | 4/30 | 62.9% / 56.9% / 66.7%（PM 监测值） |
| 重构 PM 侧抑制 -0.1，PM 读出 | 2/10 | 0/10 | 3/10 | **5/30** | **39.6% / 44.3% / 35.9%** |
| 当前最佳工程基线：原 PM 自适应窗 + 进度方向复用（13.12） | 3/10 | 1/10 | 5/10 | **9/30** | 另一个决策窗口协议，不能直接横比 |

PM 重构确实让机器人中的静默大幅下降，并在固定窗协议下给出有限的成功增益；但不能越过当前
最佳基线。原因是 PM 已不再是唯一瓶颈：即使每次动作选择更确定，连续机器人仍可能在错误的
离散格/阶段中执行确定的错误动作。因此此版本保留为可复现的 PM 动力学正向证据，不替换主
baseline，也不据此启动 RRR。结果为
`results/lift_bdm_snn_pm_reconstructed_gm01_window10_seed{0,1,2}_pretrain20_autonomous10/`。

#### 独立教师--学生审查

另行对当前代码、轨迹协议与结果做了只读审查，结论是：**教师覆盖不足是重要原因，但不能单独
解释低成功率。**

1. 当前机制不是严格的“离线教师蒸馏”。教师期由脚本教师实际接管动作，同时进行环境奖励、
TD 三因子更新与 `behavior_clone`；无教师期 `teacher_decisions=0`，但权重仍继续在线更新。
   因此 9/30 的准确表述应为“教师预训练后的在线自适应闭环”，不是冻结学生的纯泛化分数。
2. 教师仅在固定方块、确定性 Panda 初态的近名义轨迹中演示。学生一旦在 `align` 静默/并列而
   随机走偏，就会到达教师很少访问的横向偏移、错误下探、接触失败和恢复状态；这正是行为克隆
   的 covariate shift / compounding error。连续状态无限，20 条窄轨迹不可能覆盖所有 case。
3. 更根本的是状态聚合而不仅是示范条数：学生的 `align` 把连续 xy 误差压为 5x5 格（每格约
   20 mm 且边界截断），教师却按连续误差的符号和较大轴选择动作；同一学生输入可对应不同教师
   动作。实测每个 seed 有约 30 次同离散状态的硬克隆标签冲突；方块抖动演示由 3/10 降为 2/10，
   动作频率投票仅 1/10。这说明在相同粗编码上简单增加数据会累积相互冲突的标签。
4. 覆盖不是唯一项的反证来自二维查表：那里所有 25 个状态都已遍历、无标签冲突，但原 PM
   仍只有 14--17/25；必须先修复 PM 动力学才达到 25/25。
5. 当前 `option` 是部署可用的安全门控而非教师，但 `descend/close/lift/recover` 各只允许一个
   原语，SNN 主要承担 `align` 的四方向选择。因此应将当前系统诚实称为“FSM 安全门控 + SNN
   对齐”的混合控制 baseline，并分开报告对齐、进入下降、抓取和 Lift 成功率，而非端到端 8 动作
   SNN 操纵。

SOTA 机器人方法处理的不是“神奇外推”，而是明确扩展训练分布与优化目标：行为克隆同样会有
分布偏移；**DAgger** 让当前学生 rollout，把真实偏移状态交给专家标注并聚合数据，最接近本任务；
**SAC/PPO** 等在线 actor--critic 在大量仿真交互、replay、探索奖励和课程中学习恢复；目标条件
RL 加 **HER** 可把失败抓取重标为中间可达目标；domain randomization / 扰动课程扩大可训练支持集。
GAIL 仍需足够专家覆盖，CQL/IQL 等离线 RL 也只能在已有数据支持内保守优化，均不能从 20 条
窄轨迹凭空解决 OOD 恢复。

由此推荐顺序为：

1. 先增加**只记录、不改策略**的覆盖审计：教师/学生每个 option 的离散状态集合、连续 xy/z
   误差直方图、同状态标签熵、学生落在演示支持外的比例、读出与教师一致率；并把“冻结权重、
   无教师”评测与“允许在线适应”严格分为两组。
2. 然后才做 DAgger 式聚合：让当前学生真实 rollout 暴露错误状态，训练期教师只标注这些恢复
   动作；前提是先补充能够消除冲突的局部连续残差或更细且不截断的 xy、z/接触信息。首要验收
   是标签熵/冲突下降，而不是多跑一些 episode。
3. 在未见初态与扰动的冻结权重评测中验证，成功后再讨论在线 TD 的适应收益。
4. 若仍不稳定，再将任务视作新算法方向：以目标条件、分层 option actor--critic，或
   SAC/PPO+课程+HER 建立数字控制上界，然后研究如何蒸馏/映射至 SNN；这不是原 BDM-SNN 的
   小修补。全通信 baseline 稳定之前仍不进入 RRR。

### 13.23 覆盖审计：证实分布偏移，但暂不直接运行 DAgger（2026-07-25）

按 13.22 的优先级，新增 `--coverage-audit`。它是只读观察器：不调用随机数、不覆盖动作、
不写突触、不改变教师概率或在线 TD 更新。对每个教师执行决策，记录 `(option, DLPFC state)`、
教师动作与连续特征

\[
f=[e_x,e_y,e_z,\mathrm{grasped}],\qquad
[e_x,e_y]=(p_{cube}-p_{eef})_{xy}-(-0.02,0).
\]

无教师时同样由部署可见的 option 控制器计算**反事实教师动作**，但仅用于日志，绝不用于实际
控制。审计报告：(a) 教师离散状态数、冲突状态数、标签熵；(b) 学生状态是否出现于教师支持集；
(c) 在相同 option / grasp 模式下，学生连续 ((e_x,e_y,e_z)) 到最近教师样本是否超过 6 mm；
(d) 网络动作、最终执行动作分别和反事实教师的符合率；(e) 各 option 的 xy 二维直方图。

为避免把不同控制时程混为证据，正式运行严格采用历史 13.12 协议：`max_decisions=180`、每决策
1 个 MuJoCo 控制步、PM 最少 3 最多 30 步自适应积分、20 个全教师 episode（`teacher_end=1`）、
`action_clamp + behavior_clone + three_factor`，随后 10 个 `teacher_decisions=0` episode。三 seed
审计如下：

| seed | 无教师 Lift | 教师离散状态 / 冲突状态 / 平均标签熵(bit) | 离散支持外 | 连续支持外(6 mm) | 同时支持外 | 最近教师距离 | 网络--教师一致率 | 最终动作--教师一致率 | 无教师 option 决策（align/descend/close/lift/recover） |
| ---: | ---: | --- | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| 0 | 3/10 | 32 / 2 / 0.083 | 32.2% | 50.7% | 19.6% | 14.0 mm | 61.1% | 68.0% | 1294 / 310 / 8 / 29 / 0 |
| 1 | 2/10 | 32 / 2 / 0.083 | 25.9% | 58.2% | 18.5% | 19.0 mm | 50.4% | 56.5% | 1489 / 192 / 8 / 23 / 2 |
| 2 | 0/10 | 32 / 2 / 0.083 | 25.5% | 66.8% | 24.6% | 18.3 mm | 39.0% | 48.2% | 1800 / 0 / 0 / 0 / 0 |

其中“离散支持内”只表示学生落在某个已经见过的 5x5 格/option，不表示连续几何相同；因此连续
支持外比例明显更高是预期且更有解释力。seed 2 全程无法越过 align，正对应最低的教师动作一致率。
审计加入后该次重新运行的成功数为 3/10、2/10、0/10；它与历史 9/30 在个别 seed 上存在仿真/
GPU 轨迹差异，且审计不修改策略，故这次的 5/30 **不替换**历史 baseline，也不能解释为审计导致
性能下降。本节只使用与成功率无关的覆盖指标作判断依据。

**是否现在使用 DAgger？结论：需要 DAgger 的数据分布思想，但不应立刻把现有离散编码直接接入
DAgger。** 证据支持其必要性：50--67% 无教师状态在连续几何上超过教师支持，且网络--教师动作
一致率仅 39--61%，学生确实经常到达名义示范之外。但审计也显示现有教师状态已有冲突；旧实验的
方块抖动示范已经证明，在 5x5 粗格上增加数据会增加冲突而非提高成功。若现在聚合 DAgger 标签，
会把更多不同的恢复几何压到相同 DLPFC 行，再由 `behavior_clone` 逐行覆写，不能保证改善。

因此下一步应是一个更受限的前提实验，而非马上跑 DAgger：仅在 `align` 为现有 5x5 组合状态追加
少量**连续残差**（每轴格内正/负半格或 2--4 个重叠人口神经元），并取消边界饱和；z 高度、抓取/
接触也以独立可观测上下文加入。保持 8 个动作、option、PM 重构开关、全通信和所有学习规则不变。
先用教师数据验收“冲突状态数/标签熵下降”，再启动训练期 DAgger：学生实际 rollout，脚本教师仅为
当前偏移状态打恢复标签，聚合后训练；评测则冻结权重且无教师，另设在线 TD 自适应组。这样 DAgger
解决的是已经可区分的 OOD 状态，而不是用更多互相矛盾的标签掩盖表示问题。

审计代码在 `lift_bdm_snn.py` 的 `CoverageAudit`，结果位于
`results/lift_bdm_snn_option_grid_progresshold_coverageaudit_matched_seed{0,1,2}_pretrain20_autonomous10/`。

### 13.24 最小充分 align 表示与严格 DAgger 消融（2026-07-25）

覆盖审计证明存在 OOD，但 13.23 不允许立刻增加 DAgger 标签：必须先让学生状态能够表达教师选
横向还是纵向动作的依据。教师的 align 规则是

\[
a^*=\begin{cases}
\mathrm{sign}(e_x), & |e_x|\ge |e_y|,\\
\mathrm{sign}(e_y), & |e_y|>|e_x|.
\end{cases}
\]

因此单纯把原 20 mm 的 5x5 网格细化为 10 mm 的 12x12 残差网格（两端各一个 overflow bin）仍会
让同一格跨越 \(|e_x|=|e_y|\) 边界。20 个教师 episode、约 2,559--2,562 个标签的三 seed
审计确实显示：细网格有 43 个已访问状态、**5 个冲突状态**、平均熵 **0.102 bit**，没有通过
“冲突下降”的验收。

第二版只增加一个完全由部署几何计算、与教师隐藏 phase 无关的最小变量
\(b=\mathbb{1}[|e_y|>|e_x|]\)：align 输入为 `12 x 12 x 2 = 288` 个状态，另四个 option 保持原
64 状态，总 DLPFC 状态数为 **544**。它不是把教师动作作为状态输入，而是把教师规则实际需要的
“哪一轴误差较大”显式可观测化。相同三 seed 教师审计为：48 个已访问状态、**0 冲突状态、0 bit
熵**，且教师可达性为 15/20。由此该表示通过前提验收。

在不使用 DAgger、仍允许历史协议中的在线 TD 更新时，20 教师 + 10 无教师的对照达到：

| 表示 | seed 0 | seed 1 | seed 2 | 无教师 Lift | 无教师抓取 | 标签冲突 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| 原 5x5 + 进度记忆（13.12） | 3/10 | 1/10 | 5/10 | 9/30 | 10/30 | 有 |
| 12x12 残差 + 主误差轴 + 进度记忆 | 5/10 | 6/10 | 4/10 | **15/30** | **17/30** | **0** |

这说明此前 5x5 表示的冲突确实是主限制之一；但该 `15/30` 是**允许在线 TD 适应**的闭环数字，
不能和冻结部署分数混用，也仍不足以称为稳定控制。

在通过表示验收后，实现严格 DAgger 三阶段协议：

1. 前 20 回合为全教师执行，写初始行为克隆和在线 TD；
2. 接着 10 回合为 DAgger：**环境只执行 SNN/option 的学生动作**，`teacher_decisions=0`；脚本教师
   仅在学生实际到达的状态计算反事实动作标签，用 `behavior_clone` 聚合到 D1/D2 行，学生执行动作
   的 TD 更新仍保留；
3. 最后 10 回合为严格冻结评测：无教师、`epsilon=0`、不执行 D1/D2、critic 或 DAgger 标签更新。

为避免把“已经被 DAgger 追加的数据”错误地当成初始示范覆盖，代码在 DAgger 开始前冻结初始支持
集，单独报告 DAgger rollout 相对该支持的 OOD。三个 seed 均追加 1,800 个学生状态标签，最终保持
**零标签冲突/零熵**。但冻结评测只有：

| 训练方案 | seed 0 | seed 1 | seed 2 | 冻结无教师 Lift | 冻结抓取 |
| --- | ---: | ---: | ---: | ---: | ---: |
| 20 教师后直接冻结（无 DAgger） | 1/10 | 0/10 | 1/10 | **2/30** | 3/30 |
| 20 教师 + 10 学生执行 DAgger 后冻结 | 2/10 | 1/10 | 0/10 | **3/30** | 3/30 |

DAgger 使冻结评测时相对聚合支持集的连续支持外比例约降至 9.5%，而无 DAgger 对照为 45--66%；
这验证了它确实覆盖学生诱发状态。不过成功只增加 1 次，且三 seed 仍不稳定，不能称作有效的
Lift 解决方案。合理解释是：状态可区分性与分布支持虽已改善，但 PM/在线突触写入、连续动作时程、
接触--闭夹--抬升的稀疏成功链仍限制冻结策略。

因此本轮在此停止增加 DAgger 回合或改动其标签权重，以免用更多数据掩盖机制问题。当前可靠结论是：

- **表示修正**是有力正向结果（允许适应的 15/30）；
- **DAgger** 的分布校正方向正确但当前小样本、逐行硬克隆版本只给出很弱增益（冻结 3/30 对 2/30）；
- 下一轮应优先把学生动作持续时间/接触阶段的状态与更新单独诊断，并将“冻结策略”和“在线适应”
  始终分别报告，而非立刻进入 RRR。

新开关为 `--align-residual-grid-context`、`--align-residual-axis-context`、`--dagger-episodes` 与
`--freeze-evaluation`。结果目录分别为
`results/lift_bdm_snn_alignresidual_teacher20_coverage_seed*/`、
`results/lift_bdm_snn_alignaxis_progresshold_coverage_seed*_pretrain20_autonomous10/`、
`results/lift_bdm_snn_alignaxis_frozen_eval10_seed*/` 和
`results/lift_bdm_snn_alignaxis_dagger10_frozen_eval10_seed*/`。

### 13.25 失败链路里程碑诊断：当前首先卡在 align，在线适应可作为独立部署协议（2026-07-26）

为避免再由 `option_decisions` 的粗略计数倒推阶段，在 `lift_bdm_snn.py` 加入**只记录、不改变控制或
学习**的 `stage_milestones`：每回合记录首次进入 `align/descend/close/lift/recover` 的决策号、align
初始/最小/退出 xy 误差、首次抓住/成功的时刻，并按 option 记录 SNN 原始读出及最终执行动作与只读
反事实教师的符合率。反事实教师仅供离线统计，绝不参与无教师回合的动作或写权。

使用与 13.24 相同的固定方块、确定性 Panda 初态、20 个全教师回合和后三个 seed 各 10 个无教师
回合，重新运行两种唯一差异在于是否允许 TD 三因子写入的协议：

| 无教师协议 | 进入 descend（align 成功） | 进入 close / lift | 曾抓住 | Lift 成功 | 首个失败为 align | 平均 D1 / D2 L1 写入 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| 冻结：无教师、`epsilon=0`、不更新 critic/D1/D2 | 5/30 (16.7%) | 3/30 / 3/30 | 3/30 (10.0%) | **2/30 (6.7%)** | 25/30 (83.3%) | 0 / 0 |
| 受限在线适应：无教师、`epsilon=0`，仅真实奖励 + TD + 局部资格迹更新 | 11/30 (36.7%) | 11/30 / 9/30 | 11/30 (36.7%) | **7/30 (23.3%)** | 19/30 (63.3%) | 11.41 / 12.13 |

这里“对齐成功”严格定义为 FSM 已由部署可见的 `xy_error <= 2 mm` 条件进入 `descend`，不是仅看
xy 误差短暂变小。在线组进入 descend 的平均首次时刻为第 87.8 个决策；冻结组为第 114.8 个决策。
后段由 FSM 规定单一的下探、闭夹和上抬原语：11 个在线回合进入 close，其中 9 个进入 lift，但
11/11 都曾抓住方块；冻结对应为 3/3、3/3、3/3。在线组抓住后 7/11 抬升成功。因此当前最主要、可量化
的瓶颈仍是横向 align 的 SNN 读出/更新，而不是已经进入后段后的动作种类选择；也与 align 阶段原始
SNN 读出对反事实教师的一致率仅 42.7%（冻结 36.6%）相符。后段仍有少量问题：
在线组 2/11 曾抓住但未 Lift，说明接触保持/剩余时程是第二优先级，不能据此宣称已解决。

这次重新运行的受限在线组为 7/30，而 13.24 曾记录 15/30。插桩不调用随机数、不覆盖行为也不写
权，不能把差异解释为日志导致；这表明现有 15/30 是一个历史样本而非已确立的稳定成功率。今后只可
将它报告为“某次三 seed 运行的 15/30”，不可简化为“自主 Lift 已稳定达到 50%”。新诊断结果目录为
`results/lift_bdm_snn_alignaxis_stage_online_seed*/` 与
`results/lift_bdm_snn_alignaxis_stage_frozen_seed*/`。

#### 在线 STDP 是否属于最终部署？

可以，但必须将它与冻结泛化严格分开。BDM-SNN 的原始闭环思想本来就是动作后由奖励调制可塑性；仓库
`BDM-SNN-UAV.py` 也在每个动作后 `updateNet`，随后只清神经元状态/资格迹而保留长期权重。不过 UAV
脚本中的状态和奖励为占位实现，不能视作已验证的 UAV 实验。

因此本项目后续预注册两类无教师结果：

1. **冻结泛化**：训练结束后，无教师、`epsilon=0`，不写突触、critic 或通信投影；衡量预训练策略
   本身的控制能力。
2. **受限在线适应**：同样无教师和无特权状态/反事实教师/DAgger 标签，只允许部署可得传感器、真实
   奖励、局部资格迹和广播 TD 标量更新指定 DLPFC--StrD1/StrD2 突触；衡量硬件面对偏移的自适应能力。

后者不是放宽标准：必须限制学习率、权重范围、每回合/总写入次数和累计 L1 漂移，按回合报告成功率
与写入量；安全模块只能中止动作，不能输出教师动作。硬件映射上，资格迹与 RRAM 写脉冲应留在本地
神经核，低频奖励/TD 标量可广播；还要独立统计写脉冲数、写能耗、写延迟、耐久、非线性、漂移和噪声。
DAgger 始终属于训练期教师数据聚合，不能冒充部署在线学习；RRR 的重拟合也须保持冻结或另设慢时标
消融，不能混入 STDP 收益。

当时提出过受限在线适应预算的后续消融；但随后研究目标调整为先调通无限制的动作--奖励--更新闭环，
该消融暂不执行，具体闭环实验与后续判断见 13.26。

### 13.26 动作--奖励--更新闭环对齐：无教师在线学习的长窗口验证（2026-07-26）

本轮按“先调通闭环、暂不做冻结/限写消融”的原则，只检查一个因果一致性问题。原自主路径在 PM
静默/并列时会经随机决胜或进度验证短期记忆选出实际 Cartesian 动作，但 TD 三因子更新沿用的自然
STDP eligibility 可能来自另一个 PM 候选；即“真实执行 `+x` 后距离变小”的奖励不一定写回
`state -> +x` 通道。新增 `--executed-action-credit` 后，**仅在无教师回合**，用实际发给机器人
的原语在当前 DLPFC 状态的 D1/D2 通道留下局部 one-hot eligibility，再使用原有真实进度奖励、TD
critic、三因子正负调制和权重范围。它不读取教师、不给标签、不新增动作或状态，也不改变 PM 前向；
自然 STDP trace 继续累积给下一决策。

以 20 个全教师预训练回合后，连续 50 个 `teacher_decisions=0` 自主在线回合运行三 seed；全程
`epsilon=0`、固定方块/初态、全通信、主误差轴编码、进度验证方向记忆，且**不使用**
`--freeze-evaluation`。因此这是目前要验证的“无限制在线 BDM-SNN 闭环”，不是冻结泛化。结果为：

| 自主回合窗口 | align -> descend | 曾抓取 | Lift 成功 | 说明 |
| --- | ---: | ---: | ---: | --- |
| 20--29 | 4/30 (13.3%) | 4/30 | 4/30 (13.3%) | 初始教师权重单独不足。 |
| 30--39 | 14/30 (46.7%) | 10/30 | 9/30 (30.0%) | 在线动作--奖励配对后出现改善。 |
| 40--49 | 7/30 (23.3%) | 5/30 | 2/30 (6.7%) | 仍有明显策略漂移。 |
| 50--59 | 19/30 (63.3%) | 15/30 | **14/30 (46.7%)** | 当前最强的无教师在线窗口。 |
| 60--69 | 13/30 (43.3%) | 8/30 | 7/30 (23.3%) | 改善没有单调保持。 |
| 20--69 合计 | 57/150 (38.0%) | 42/150 (28.0%) | **36/150 (24.0%)** | seed 0/1/2 分别为 15/50、8/50、13/50。 |

这是一个正向但有限的闭环证据：无教师期确有多次成功，且在第 50--59 回合达到 46.7%，说明环境
可测进度奖励可经实际动作通道、局部资格迹和 TD 调制改善后续控制；并非必须持续调用教师。不过
窗口成功率由 13.3% 升至 46.7% 后又回到 23.3%，所以不能把峰值称作稳定自主率。当前问题不是
“完全没有学习”，而是高学习率下的持续在线更新、PM 静默/并列随机决胜、有限样本 TD critic 共同
造成非平稳权重漂移；平均每自主回合 D1/D2 L1 更新约 0.68/0.70（逐回合已记录），而成功回合仍有
17--53 个 PM 静默决策。结果目录为
`results/lift_bdm_snn_alignaxis_online_credit_seed*/`。

下一步仍遵循当前目标：不转入冻结、DAgger 或 RRR，而是在相同无限制在线协议下解决“能改善但
不能保持”的问题。最小的下一项应是把**真实 xy 进度**从仅用于 shaping reward 扩展为明确的局部
动作价值：对 align 中每个已执行方向维护跨回合的进度统计，并只在 PM 静默/并列时用该统计打破
平局；TD/STDP 仍无限制执行。这样不替代 SNN 的唯一赢家，也不提供教师动作，而是减少随机错误
动作持续污染在线更新。验收应是后三个 10 回合窗口不低于前两个窗口，并同时提升 `align -> descend`
到达率；若仍失败，才说明需要重构更长时程的 critic/状态，而非继续简单延长训练。

### 13.27 去除偶然进度偏置：三次实测后，在线闭环提升至 38%（2026-07-26）

按 13.26 的推导，实现 `--align-progress-value`。在每个实际可观测 align DLPFC 状态 (s) 和横向
动作 (a\in\{+x,-x,+y,-y\}) 上保留跨 episode 的局部值 (q_{s,a})，每次实际执行后以归一化 xy
误差下降 (p=\mathrm{clip}((d_{before}-d_{after})/0.01,-1,1)) 更新：

\[
q_{s,a}\leftarrow\mathrm{clip}[(1-\eta)q_{s,a}+\eta p,0,q_{max}].
\]

它完全由真实物理进度更新，不读取教师标签；只有 PM 读出静默/并列时才把 (q_{s,a}) 加到四个
横向读出分数，PM 唯一赢家不受影响，原有 TD/STDP 仍对每个实际动作无限制写入。

第一版只需一次观测即可参与平局裁决，三 seed、20 教师 + 50 无教师在线回合得到 `39/150` Lift，
仅略高于 13.26 的 `36/150`；但成功从早期窗口 `23/30` 逐步降至最后两个窗口各 `4/30`。原因是
某次偶然误差下降会留下长期正值，随后在不确定 PM 中反复自我确认。故不保留该一次观测版本。

最小修正是要求同一 `(state, action)` 获得至少 **3 次**真实物理进度样本才允许参与平局裁决；这
不是限制 STDP 写入，只是拒绝以一次偶然观测替代随机 tie-break。相同三 seed、相同 20 + 50 协议为：

| 自主窗口 | 无最小样本进度表 | 三次样本后进度表：Lift | 三次样本后 align -> descend |
| --- | ---: | ---: | ---: |
| 20--29 | 13/30 | 13/30 | 17/30 |
| 30--39 | 10/30 | **14/30** | **18/30** |
| 40--49 | 8/30 | **14/30** | **15/30** |
| 50--59 | 4/30 | 7/30 | 9/30 |
| 60--69 | 4/30 | 9/30 | 10/30 |
| 20--69 | 39/150 (26.0%) | **57/150 (38.0%)** | **69/150 (46.0%)** |

三次样本版本三个 seed 分别为 `23/50`、`17/50`、`17/50`；后三个窗口合计 `30/90`，不低于前两个
窗口的 `27/60`，因此通过了“避免后期整体坍塌”的最低验收，并比没有进度表的动作信用对齐版本
`36/150 (24.0%)` 增加 21 次成功。它仍不是稳定控制：最后两个窗口为 23.3% 和 30.0%，远低于可
部署门槛，且进度表实际介入了约 4,756 个 PM 不确定决策，说明 PM 自身读出仍是主要问题。

这版应诚实称为**全通信 SNN + 部署可得的局部在线进度价值辅助**：辅助只在读出不确定时工作，
不是教师、不是 DAgger，也未替换 SNN 的唯一赢家；但它目前是软件表而非已实现的脉冲脑区。下一步
不应马上把它包装成 SNN 或进入通信压缩，而应先在同一无限制在线协议中增加一条只读检查：分开统计
“PM 唯一赢家”和“进度表裁决”两类动作各自的真实进度与成功贡献。若进度表裁决长期优于随机平局，
再把其 `(state, action, progress)` 局部更新映射为小型脉冲价值/资格迹群；若两者相近，则应优先
重构 PM 读出而非继续增大表。结果位于
`results/lift_bdm_snn_alignaxis_online_progressvalue_seed*/` 与
`results/lift_bdm_snn_alignaxis_online_progressvalue3_seed*/`。

### 13.28 动作来源审计：短期物理验证记忆优于进度价值表（2026-07-26）

对 13.27 的三次样本版本加入只读 `align_action_sources` 统计，运行配置、随机数调用、动作和学习
规则均不变。每个无教师横向动作按其最终实际来源记录为：PM 唯一赢家、PM 不确定但进度价值表裁决、
PM 不确定但进度验证短期记忆复用、或无辅助的随机 fallback；分别统计执行后的 xy 误差变化及是否
**直接**让 FSM 从 align 进入 descend。三 seed、150 个自主回合合并结果：

| 最终动作来源 | 横向决策数 | 正 xy 进度比例 | 平均 xy 误差下降 | 直接进入 descend |
| --- | ---: | ---: | ---: | ---: |
| PM 唯一赢家 | 15,455 | 61.3% | 0.655 mm | 45 (0.29%) |
| 进度价值表裁决 | 1,774 | 49.3% | 0.313 mm | 0 |
| 进度验证短期记忆 | 3,061 | **77.2%** | **1.217 mm** | **29 (0.95%)** |
| 无辅助模糊 fallback | 237 | 16.0% | -1.002 mm | 0 |

不能将后续抓取/Lift 成功机械归因给单次横向动作，故表中只报告可因果对应的一步 xy 进度与
`align -> descend` 转移。结果显示两点：

1. PM 的唯一赢家仍提供最多的正常动作样本，但单步进度弱于已验证方向记忆，说明 PM 本身未能
   稳定输出最有效的连续横向方向。
2. 三次样本进度价值表虽然提升总体成功率的一部分背景条件，但作为**最终直接裁决**动作反而弱于
   PM 唯一赢家，且在当前优先级下常被后续短期记忆覆盖；没有证据支持立即把它映射成新的脉冲价值
   脑区。无辅助 fallback 明确有害。

因此当前最小、最有前景的保留机制是“实际执行后以物理误差验证方向、有限步复用”的短期闭环记忆；
进度价值表保留为消融开关，不作为下一版结构核心。下一步仍维持无限制在线 TD/STDP 与无教师部署，
但不再扩大表，而是构造只含**已验证方向、剩余 xy 误差大小及有限有效期**的显式状态/脉冲群，使
PM 不确定时可从 SNN 内部表达短期记忆而非软件动作覆盖。开始接入 Lift 前，应先在二维对齐探针验证
该小状态能提高 PM 唯一/有效读出的比例；否则优先重构 PM 读出动力学。结果为
`results/lift_bdm_snn_alignaxis_online_progressvalue3_sourceaudit_seed*/`。

### 13.29 先否定无效的脉冲上下文，再重构 PM 竞争（2026-07-26）

本轮先没有直接改复杂的 Lift，而是在 `planar_progress_context_probe.py` 中做一个可枚举的 5x5
二维对齐预检。位置群仍以电流 2.0 输入 BDM-SNN；候选上下文群额外编码“上一次**真实**使曼哈顿
距离下降的方向、剩余距离二值区间、3 步有效期”，并仅以 0.25 的小调制电流加入。教师只写位置行，
从不为上下文行提供动作标签；部署期上下文行只根据实际距离下降做局部三因子更新。这个设计是为了
检验能否用显式脉冲状态替代 13.28 的软件短期动作复用，而不把软件规则直接搬进 Lift。

三 seed、每 seed 100 个随机离散起点的结果如下：

| 版本 | 成功 | 正距离进度 | PM 唯一读出 | 结论 |
| --- | ---: | ---: | ---: | --- |
| 仅位置群 | **218/300 (72.7%)** | **67.7%** | 60.1% | 参考基线 |
| 位置群 + 脉冲进度上下文 | 197/300 (65.7%) | 64.5% | 64.1% | 唯一读出略多，但动作更常确定地错误 |

因此不把该上下文群接入 Lift；这是一条有价值的负结果：增加一个短期记忆输入，并不自动改善
BDM-SNN 的动作选择，不能仅以 PM 的唯一 winner 比例来判断机制有效。结果在
`results/planar_progress_context_probe/`。

转而采用已有二维完整查表验收（13.22）的最小正向结论：原始 PM 的非对角侧抑制
`g_lat=-2.5` 过强，`g_lat=-0.1` 在 10 个 seed 中均完成 25/25。除这一 PM 内部竞争参数外，
本次 Lift 协议与 13.27--13.28 完全相同：全通信、20 个教师期、后 50 个
`teacher_decisions=0` 的确定性自主回合、真实环境奖励、执行动作资格迹、无限制 TD/STDP 写入、
进度验证短期记忆和三次样本进度表均保留；没有 DAgger、没有教师标签、没有 RRR。

| 无教师 50 回合 x 3 seed | 原 PM：`g_lat=-2.5` | 弱侧抑制 PM：`g_lat=-0.1` |
| --- | ---: | ---: |
| Lift 成功 | 57/150 (38.0%) | **109/150 (72.7%)** |
| 各 seed 成功 | 23 / 17 / 17 | **36 / 35 / 38** |
| PM 静默决策比例 | 25.2% | **3.3%** |
| 到达 descend | 69/150 (46.0%) | **147/150 (98.0%)** |
| 曾抓取方块 | 65/150 (43.3%) | **138/150 (92.0%)** |
| 到达 lift option | 65/150 (43.3%) | **135/150 (90.0%)** |

这次提升的因果链与诊断一致：弱化 PM 竞争后，丘脑传来的动作证据不再频繁被最后一级的同步强抑制
压成静默/并列，PM 唯一动作在横向阶段的真实正 xy 进度由 61.3% 提升到 77.3%，平均单步误差下降
由 0.655 mm 提升到 1.508 mm；绝大多数 episode 因而能离开 align 并进入抓取链。所有无教师期仍
发生权重变化（每回合 D1/D2 L1 改变量约 0.665/0.663），故应称为**全通信、弱 PM 侧抑制、无限制
在线适应闭环**，不能称为冻结部署的泛化成功率，也不能归因于通信压缩或硬件实现。

仍有 41/150 次未成功，且已有 138 次曾抓取但只有 109 次 Lift 成功，剩余主要瓶颈已从“横向
对齐/PM 静默”下移到接触后的夹持保持、抬升时序和在线漂移。下一步应先用现有 `stage_milestones`
对这 29 个“已抓取未成功”案例做失败链路分型，并只针对占比最高的一种后接触状态缺失做最小可观测
状态或动作持续时间改动；不要重新引入未通过二维验证的脉冲进度群，也暂不进入 RRR。结果目录为
`results/lift_bdm_snn_alignaxis_online_pmrecon_gm01_seed*/`。

### 13.30 后接触失败分型与两个最小动作时序消融（2026-07-26）

对 13.29 的 41 个无教师失败 episode 按首次阶段里程碑分型，而不是直接继续调学习率：3 个
未离开 align、9 个已下降但从未抓取、3 个曾抓取但未进入 lift、**26 个已进入 lift 但最终未成功**。
后者是最大组；其首次进入 lift 平均在第 155.9 个决策，平均只有 24.1 个决策余额，且实际只执行
3.7 次 `+z`，而成功组首次进入 lift 平均为第 140.0 个决策、实际执行 6.6 次 `+z`。这提出两个
彼此独立、无需新教师或新动作的解释：恢复阈值可能过早，或夹爪在上抬前需要更长关闭时间。

为使这两个假设可复现，Lift 入口新增只影响 option 状态机的 `--failed-lift-steps`（默认 4）和
`--close-steps`（默认 2）；默认行为不变。它们都使用与 13.29 相同的弱 PM、全通信、20 教师 +
50 无教师在线 TD/STDP、固定 180 决策预算、三 seed 协议。

| 仅改变的 option 时序 | 三 seed 无教师 Lift | 与 `109/150` 比较 | 判定 |
| --- | ---: | ---: | --- |
| 基线：close=2，未抓取 lift=4 后恢复 | **109/150 (72.7%)** | -- | 保留 |
| 延后恢复：未抓取 lift=8 后恢复 | 104/150 (69.3%)，37/33/34 | -5 | 拒绝 |
| 延长夹爪关闭：close=3 | 105/150 (70.0%)，34/33/38 | -4 | 拒绝 |

延后恢复没有修复主要失败，反而会让已失去方块的轨迹继续耗尽决策预算；增加一次关闭也没有稳定提高
成功。这两项均不作为默认策略，但参数保留为透明的时序消融开关。由此可排除“只要多等一下”这种
过度简单的解释；下一步应明确检验固定 180 决策预算是否限制了晚到 lift 的 episode，或记录真正
抓取状态在 lift/recover 前后的连续变化，再决定是否需要一个部署可得的抓持确认状态。结果目录为
`results/lift_bdm_snn_alignaxis_online_pmrecon_liftpatience8_seed*/` 和
`results/lift_bdm_snn_alignaxis_online_pmrecon_close3_seed*/`。

### 13.31 决策预算是后接触链路的实质瓶颈（2026-07-26）

13.30 的最大失败组常在第 156 个决策才首次进入 lift，因此在不改变网络、option、动作原语、
教师期、奖励或任何学习率的条件下，只把每个 episode 的最大决策数从 180 提升至 240。三 seed 结果：

| 无教师在线协议 | 180 决策预算 | 240 决策预算 |
| --- | ---: | ---: |
| Lift 成功 | 109/150 (72.7%) | **123/150 (82.0%)** |
| 各 seed 成功 | 36 / 35 / 38 | **41 / 40 / 42** |
| 曾抓取 | 138/150 | **149/150** |
| 到达 descend / close / lift | 147 / 136 / 135 | **150 / 149 / 149** |
| 平均实际决策数 | 155.7 | 173.2 |
| PM 静默率 | 3.32% | 3.36% |
| 每回合 D1/D2 L1 写入 | 0.665 / 0.663 | 0.802 / 0.799 |

成功提升 14 次且三个 seed 一致，表明原 180 步确实让一部分晚进入 lift 的轨迹在上抬前超时；PM
静默率没有变化，因而不是 PM 动力学再次改善。然而这不是一个纯粹的“给更多物理时间”消融：在当前
无限制在线协议中，多出的决策也带来更多真实 TD/STDP 写入（L1 改变量同步增大）。因此最严谨的结论
是**延长闭环交互窗口提高了当前在线适应系统的 Lift 成功率**，还不能分离出时间预算与额外在线学习
各自的贡献。这个版本是当前最佳软件结果，但仍不是冻结部署、随机初态泛化、通信压缩或硬件能耗结果。

下一步若要将 82% 变成可解释结论，应固定同一教师预训练权重，将在线权重更新关闭后再比较 180/240
预算，或在 240 步中只记录而不写入后 60 步的 TD/STDP；但这属于用户此前明确暂缓的冻结/受限写入
消融，故当前先将 240 窗口作为无限制在线闭环候选配置，并优先记录抓取确认和恢复前后的状态轨迹。
结果目录为 `results/lift_bdm_snn_alignaxis_online_pmrecon_budget240_seed*/`。

### 13.32 面向 RRAM 的状态和 Str 群体压缩（2026-07-27）

此前最佳 (S=544,A=8) 中，`align` 的 12x12 xy 残差格加主误差轴占 288 个状态，而
`descend/close/lift/recover` 四个 option 各占 64 个状态、合计 256 个。后四个阶段实际上只有
一个 FSM 安全原语被允许：`-z`、close、`+z`、open。因此其 SNN 输出不能改变实际动作，却仍分配
DLPFC/Str 神经元并写入状态--动作突触。为验证可否删除这部分冗余，新增 `--snn-align-only`：仅
`align` 时调用 DLPFC--BG--PM、更新 eligibility 和 TD/STDP；固定阶段由同一可观测 FSM 直接执行，
不送入 SNN，也不产生突触写入。教师仍只在前 20 回合存在，后 50 回合 `teacher_decisions=0`，其余
协议均保持弱 PM 侧抑制 `-0.1`、全通信和 240 决策窗口。

同时将残差分箱改为可配 `--align-residual-bins B`。每轴保留上下 overflow 两格，其余 (B-2) 个格在
[-50, 50) mm 内等宽；再乘以主误差轴二值，故 `align` 状态数为 (S=2B^2)。三 seed 的格点筛选为：

| align 状态编码 | SNN 状态数 | 无教师 Lift 成功 | 各 seed | 判断 |
| --- | ---: | ---: | --- | --- |
| 原始：12x12x2 + 4 个固定阶段块 | 544 | 123/150 (82.0%) | 41 / 40 / 42 | 旧最佳对照 |
| 仅 align，12x12x2 | 288 | 130/150 (86.7%) | 45 / 45 / 40 | 通过 |
| 仅 align，10x10x2 | **200** | **141/150 (94.0%)** | 48 / 47 / 46 | 选择 |
| 仅 align，8x8x2 | 128 | 133/150 (88.7%) | 45 / 43 / 45 | 太粗，拒绝 |

10x10 优于 12x12 并不说明更粗编码普遍更好；在当前固定场景和有限在线样本下，它减少了状态碎片化，
使每个状态--动作权重获得更多重复访问。继续缩到 8x8 后性能回落，因此选 10x10x2 作为当前任务的
最小经验支持表示，而不再继续缩格。

在 (S=200,A=8) 的通过版本上，进一步新增 `--compact-striatum`，但**没有合并 Str 与 GP**。原始
展开 Str 为每状态复制八个 LIF：

\[
\mathrm{DLPFC}[s]\rightarrow\mathrm{StrD1}[s,a],\quad
\mathrm{DLPFC}[s]\rightarrow\mathrm{StrD2}[s,a],
\]

故 StrD1/StrD2 各有 (SA=1600) 个神经元。压缩版仍保存两张完整的
(W^{D1},W^{D2}\in\mathbb{R}^{200\times8}) 状态--动作可塑权重表，且仍只对真实执行的
\((s,a)\) 做三因子更新；变化只是将每个 Str 群压为 8 个共享的动作通道 LIF：

\[
\mathrm{DLPFC}[s]\xrightarrow{W^{D1}_{s,a}}\widetilde{\mathrm{StrD1}}[a]
\rightarrow\mathrm{GPi}[a],\qquad
\mathrm{DLPFC}[s]\xrightarrow{W^{D2}_{s,a}}\widetilde{\mathrm{StrD2}}[a]
\rightarrow\mathrm{GPe}[a].
\]

即 D1/D2 两条通路、Str LIF 阈值/复位、Str->GPi/GPe、GPe/STN/GPi、丘脑和 PM 均仍独立存在；
它是动作通道级 Str 群体近似，而非“删除 Str 后直接连 GP”。三 seed 结果：

| 版本 | 总 SNN 神经元 | StrD1 / StrD2 神经元 | 无教师 Lift 成功 | 各 seed |
| --- | ---: | ---: | ---: | --- |
| 旧 (S=544) 展开 Str | 9282 | 4352 / 4352 | 123/150 (82.0%) | 41 / 40 / 42 |
| 10x10，仅 align，展开 Str | 3434 | 1600 / 1600 | **141/150 (94.0%)** | 48 / 47 / 46 |
| 10x10，仅 align，压缩 Str | **250** | **8 / 8** | **136/150 (90.7%)** | 45 / 46 / 45 |

压缩 Str 相对同状态的展开版本下降 5/150（3.3 个百分点），但未出现大幅失效，且仍高于旧 82.0%
结果。所有 150 个自主 episode 均到达 `descend/close/lift`，PM 静默率约 1.96%，说明当前主要误差
并非对齐无法完成。该结果仅在固定方块/确定性初态与无限制在线更新下成立，不能视为冻结或随机初态
泛化结果。

硬件含义：压缩版本最大权重矩阵是 DLPFC->StrD1/D2 的 200x8；DLPFC 的所有外送矩阵
`D1(200x8)+D2(200x8)+STN(200x2)+Thalamus(200x8)` 可并列打包为 200x26，一个 256x256 阵列可容纳。
其余 8/2 通道的 BG--Thalamus--PM 小矩阵可打包至第二阵列；状态输入到 DLPFC 的 one-hot 地址/单位映射
可由路由逻辑实现，若强制使用 RRAM 则另需一阵列。因此它是约 **2 个主要 RRAM MVM 阵列**、3 个
物理神经元核（DLPFC 200；BG 含 Str/STN/GPe/GPi 34；Thalamus/PM 16）的候选，而不是原软件矩阵
逐块映射所需的大量低利用率阵列。仍需在下一阶段验证 RRAM 写入量化、变异、延迟和 AER 交通。

结果目录为：
`results/lift_bdm_snn_alignonly_res{8,10,12}_seed*/` 与
`results/lift_bdm_snn_alignonly_res10_compactstr_seed*/`。

### 13.33 将 align SNN 动作轴从 8 缩至 4（2026-07-28）

在 13.32 的 `S=200`、仅 `align` 进入 SNN、压缩 Str 而不合并 Str--GP 的候选上，继续考察
动作轴是否仍有冗余。这里不能简单地把八个物理动作删成四个，否则任务无法下降、夹紧和抬升；正确的
缩减是利用已验证的可观测 option：只有 `align` 需要在横向动作中竞争，而 `descend/close/lift/recover`
分别只有 `-z`、close、`+z`、open 一个安全原语。因此新增 `--align-action-count 4`，使 SNN 的
StrD1、StrD2、GPe、GPi、thalamus、PM 均只保留 `+x,-x,+y,-y` 四通道；后四个物理原语仍由 FSM
直接下发，既不占 SNN 神经元，也不写入 SNN 突触。

为避免把动作数变化和学习设置变化混为一谈，使用与 13.32 相同协议：固定方块和确定 Panda 初态、
每 seed 20 个教师回合加 50 个无教师评价回合、240 决策上限、教师退出后保持无限制在线 TD/STDP、
PM 侧向抑制为 `-0.1`。三 seed 结果如下：

| SNN align 动作数 | DLPFC->StrD1 / DLPFC->StrD2 | 总 SNN 神经元 | 无教师 Lift 成功 | 各 seed |
| --- | --- | ---: | ---: | --- |
| 8（13.32 候选） | `200x8` / `200x8` | 250 | 136/150 (90.7%) | 45 / 46 / 45 |
| **4（本轮）** | **`200x4` / `200x4`** | **226** | **132/150 (88.0%)** | 42 / 48 / 42 |

4 通道相对于 8 通道少 24 个神经元，成功率低 4/150（2.7 个百分点）。所有 150 个自主回合都达到
`descend`；仅 2/150 未达到 `close`，其余失败主要发生于抓取--抬升后的恢复链路。这说明当前小规模
退化不是“横向动作类别不够”，而是较少 PM/BG 动作通道改变了脉冲竞争、使部分对齐轨迹在接触前超时。
在这一固定课程下，88.0% 可作为需要更小网络时的可接受候选；若将 90.7% 作为优先目标，则保留 8 通道。
不能据此声称对随机初态、冻结部署或真实机器人同样成立。

4 通道的神经元规模为 DLPFC 200、StrD1 4、StrD2 4、STN 2、GPe 4、GPi 4、thalamus 4、PM 4，
合计 226。DLPFC 的外送矩阵可并列为
`D1(200x4)+D2(200x4)+STN(200x2)+Thalamus(200x4)=200x14`，故仍可放入一个 `256x256`
RRAM 阵列，且列占用从 8 通道时的 26 降为 14。其余 BG--thalamus--PM 的最大矩阵为 `4x4`，可打包
进第二个阵列。因此在“one-hot 状态到 DLPFC 由地址/路由逻辑完成”的前提下，仍需 **2 个主要 RRAM
MVM 阵列**，不是 1 个；缩动作轴减少的是列数、神经元和小矩阵活动，而没有消除 DLPFC->BG 与
BG/thalamus/PM 这两个顺序计算阶段。若输入 one-hot->DLPFC 也必须用 RRAM MVM，则其 `200x200`
单位映射还需第 3 个输入阵列（或独立的时分复用方案），不能与上述 200x14 权重同时常驻。

复现本轮 4 通道实验（分别取 `seed=0,1,2`）的核心命令为：

```sh
CUDA_VISIBLE_DEVICES=0 /home/lph/.conda/envs/lph/bin/python \
  Brain-Cog/examples/decision_making/BDM-SNN-Robosuite/lift_bdm_snn.py \
  --episodes 70 --teacher-episodes 20 --max-decisions 240 --control-steps 1 \
  --internal-steps 3 --max-internal-steps 30 --teacher-start 1 --teacher-end 1 \
  --teacher-credit-mode action_clamp --executed-action-credit \
  --teacher-learning-mode behavior_clone --plasticity-rule three_factor \
  --three-factor-learning-rate 0.08 --critic-learning-rate 0.1 --critic-discount 0.98 \
  --clone-off-weight 0.05 --autonomous-epsilon 0 --option-context \
  --align-residual-axis-context --align-residual-bins 10 --snn-align-only \
  --compact-striatum --align-action-count 4 --pm-lateral-gain -0.1 \
  --coverage-audit --fixed-cube --deterministic-robot-start --seed 0 --device cuda:0 \
  --output-dir results/lift_bdm_snn_alignonly_res10_compactstr_a4_seed0
```

结果目录为 `results/lift_bdm_snn_alignonly_res10_compactstr_a4_seed*/`。本轮只实现并测量软件 SNN 的
动作通道缩减与全通信逻辑 spike 计数；尚未测量 AER 包数、比特数、FIFO、真实 RRAM 写入/噪声或能耗，
所以不能仅凭 `200x4` 而宣称通信或硬件能耗已降低。

### 13.34 后续 RRR / 流形通信方案（2026-07-28，尚未执行）

**架构选择。** 两个版本都可接入同一套通信接口，但后续 RRR 的主实验选择 `S=200,A=8` 的
压缩 Str 基线（90.7%），而 `A=4`（88.0%）保留为最终小型部署复核。理由不是 8 动作性能显著更高，
而是 RRR 的可压缩维度受目标群体维度限制：4 动作版本中所有 BG/PM 群体最多只有 4 维，除去
rank-1 的公共电流后，几乎没有可探索的低秩余量；8 动作版本能先检验 rank 6、4、2 的性能--通信
权衡。任何在 8 动作上得到的方案都必须用 4 动作再复核，不能假定会自动迁移。

采用当前可装入阵列的三核分区：核心 0 为 DLPFC(200)+StrD1/StrD2(各 A)，核心 1 为 STN(2)+
GPe(A)+GPi(A)，核心 2 为 thalamus(A)+PM(A)。因此监测的跨核链路为下表的八条；两个 DLPFC->Str
突触表保留在核心 0 内，第一阶段既不跨核也不压缩。

| 跨核链路 | 8 动作维度 | 4 动作维度 | 初始处理 | 原因与候选 rank |
| --- | --- | --- | --- | --- |
| DLPFC->STN | `200x2` | `200x2` | RRR | 当前全 1 权重的目标电流严格 rank-1；用 `k=1`。 |
| DLPFC->thalamus | `200x8` | `200x4` | RRR | 当前各动作通道接收相同的 DLPFC 电流，严格 rank-1；用 `k=1`。 |
| STN->GPe、STN->GPi | `2x8` | `2x4` | RRR | 当前全连接且各目标列相同，严格 rank-1；各用 `k=1`。 |
| GPe->STN | `8x2` | `4x2` | RRR | 当前两个目标接收相同 GPe 汇聚电流，严格 rank-1；用 `k=1`。 |
| StrD1->GPi、StrD2->GPe | `8x8` | `4x4` | 暂不压缩 | 对角动作身份通路，矩阵满秩；过早压缩会直接混淆候选动作。第二阶段离线审计后，8 动作依次试 `k=6,4,2`，4 动作依次试 `k=3,2`。 |
| GPi->thalamus | `8x8` | `4x4` | 暂不压缩 | 同样接近动作身份门控；先全通信，后与上两条同样的逐 rank 阶梯测试。 |

这意味着第一阶段会压缩 5/8 条跨核边，但特意保留 3 条最影响动作选择的身份通路为全通信。这里的
rank-1 结论来自**当前代码矩阵结构**，不是仅由数据拟合得出的乐观假设；仍需用真实脉冲轨迹验证重建
误差和控制性能，因为 LIF 阈值、复位与时间顺序会放大很小的电流误差。

**在线学习边界。** 当前无限制三因子 TD/STDP 只显式更新两张
`DLPFC->StrD1/DLPFC->StrD2` 状态--动作表；它们是任务适应的核心，第一阶段保持本地完整 MVM。
若未来为了更多物理核而强制将 DLPFC 与 Str 分离，才把它作为单独的高风险实验：保留完整可写权重
shadow，使用过去决策窗口的样本对该 `200x8`（或 `200x4`）通路因果重拟合，8 动作从 `k=6` 再到
`k=4`，4 动作从 `k=3` 再到 `k=2`。但 DLPFC 当前是稀疏 one-hot 脉冲，直接发送一个状态 AER 地址
可能已经比每个窗口发送多个连续 latent 值更省链路；所以它必须与 AER 包/比特/目标阵列激活实测比较，
不能预设 RRR 一定更优。

**执行顺序与门控。**

1. 先改通信模块：支持按链路设置 rank、显式在每个决策结束后用历史窗口 refit，并把校准窗口、
留出窗口和部署窗口严格时间因果分开。当前 `three_factor_update` 直接写权重，不会触发旧
`UpdateWeight` 内的 refit 钩子；不修复此点不能进行可信的在线 RRR 实验。
2. 对全通信 8 动作轨迹做离线审计：按链路、按训练/无教师评价时段绘制 `k=1..min(N_src,N_tgt)` 的
目标电流 EV、NRMSE、PM 动作一致率和单窗口峰值误差。仅当留出窗口满足预设阈值（建议 EV >= 0.99、
NRMSE <= 5%、无静默/并列恶化）才允许该 rank 进入闭环。
3. 先闭环启用上述五条理论 rank-1 链路，三 seed 重跑 20 教师 + 50 无教师回合；接受条件为相对
同 seed 全通信成功率下降不超过 5 个百分点，且没有新的 PM 静默、超时或阶段转移失败模式。失败时
逐链路回退全通信，而不是调网络其它参数掩盖问题。
4. 再只对一条动作身份链路做 rank 阶梯：8 动作 `6 -> 4 -> 2`，4 动作 `3 -> 2`；每个 rank 均先离线
审计、再单链路闭环、最后三条身份边联合闭环。rank 等于原维度是对照，不称为压缩。
5. 线性 RRR 通过后才实现真正的 latent LIF 群体、阈值/不应期与 AER 编码；分别记录逻辑源 spike、
latent spike、AER 包/bit、峰值事件率、FIFO 占用、目标阵列读次数、控制延迟及成功率。此时再比较
连续 latent、spike latent 与预测残差事件三种传输，避免把“latent 标量数变少”误称为链路降耗。
6. 只有线性 RRR 在动作身份链路上无法同时达到误差门槛和任务门槛时，才研究非线性/递归 latent
流形；它应解释线性不足的残差动力学，而不是作为没有线性基线的额外复杂网络。

每个阶段都保留 rank=0 全通信、同一动作数、同一 seed 和同一在线学习协议。现阶段的目标是建立
“任务保持--电流重建--真实事件/阵列工作量”三者的因果证据，而不是仅使矩阵 rank 变小。

### 13.35 RRR 第一阶段：因果在线重拟合与闭环反例（2026-07-28）

本轮以 13.32 的 `S=200,A=8`、250 神经元、全通信成功率 136/150 (90.7%) 为唯一基线，开始
RRR 的最保守闭环尝试。首先完成必要工程修正：三因子 TD/STDP 会直接写
`DLPFC->StrD1/DLPFC->StrD2` 权重，原来的 RRR refit 只挂在旧 `UpdateWeight` 内，因而不会被这一
学习路径调用。现改为 `BDMSNN.refit_communication()`：每个 **SNN-active 高层决策** 在完成本次
环境反馈、本地 TD/STDP/教师克隆更新后因果 refit；下一决策才可用新投影。通信模块同时支持逐链路 rank
字典，而非所有边共享一个 rank；它记录在已拟合模型下的在线 EV、NRMSE、峰值电流误差、全通信回退
样本、连续 latent 标量数。此处的“在线 EV”是预测时与完整影子电流比较的数值诊断，不是任务性能。

第一组试验按最初方案同时对五条非身份边使用 rank-1：`DLPFC->STN`、`DLPFC->thalamus`、
`STN->GPe`、`STN->GPi`、`GPe->STN`；三条动作身份边始终全通信。DLPFC 两条广播在某些窗口是
目标恒定电流，故实现为精确仿射 DC (rank-0) 解码，而非把“零方差”误记为拟合失败。三 seed、每 seed
20 教师 + 50 无教师、其它参数与全通信基线相同，结果为：

| 版本 | RRR 链路 | 无教师 Lift 成功 | 各 seed | 结论 |
| --- | --- | ---: | --- | --- |
| 全通信基线 | 无 | 136/150 (90.7%) | 45 / 46 / 45 | 对照 |
| 五条非身份边 | DLPFC 两边 + STN/GPe 三边，`k=1` | 79/150 (52.7%) | 29 / 24 / 26 | 拒绝 |
| 仅 STN/GPe 三边 | `STN->GPe`、`STN->GPi`、`GPe->STN`，`k=1` | 77/150 (51.3%) | 29 / 27 / 21 | 拒绝 |

两组 RRR 的在线电流诊断却几乎完美：STN 两条边 EV 约
`1-2.4e-14`、NRMSE 约 `1.5e-7`，GPe->STN EV 约 `1-5.4e-14`、NRMSE 约 `2.3e-7`；DLPFC 的 DC
边 EV 为 1 或近 1。也就是说，**本轮失败不能归因于“选错 rank”或明显电流重建误差**。即使只有三条
STN/GPe rank-1 边，有限精度的连续投影/解码误差也会在 LIF 阈值、复位和 GPe--STN--GPi 递归中改变某一
内部步的 spike；该 spike 差异再经在线学习和后续闭环状态放大，最终使无教师成功率约减半。相应地，
评价期 PM 静默数从全通信三 seed 的 172/119/156 上升为保守 RRR 的 266/191/266（分母分别约
7.4k--10.6k 决策），且失败轨迹更长。这是一个有价值的反例：**EV=1 与低 NRMSE 均不足以作为脉冲
闭环控制中的部署准则。**

本轮没有把 latent 标量减少误报为 AER 节省。例如保守 RRR 的 seed 0 中，三条源群共产生
158,977 个 logical source spike，而三条链路传输了 115,587 个连续 latent 标量（每个已拟合内部步一
个标量）；两者事件语义、时间编码和物理包格式不同，尚不能比较能耗或带宽。需要真正的 latent LIF /
残差事件编码与 AER 包计数后才能作该结论。

因此当前可接受的结论是：**全通信仍是 Lift 的性能基线；当前“连续 RRR 电流替换”在这条递归 BG 回路
中不通过闭环验证。** 下一步不应继续压缩更多动作身份边。应在保持完整影子电流的前提下，先做一种
事件语义保守的通信机制：源端预测目标电流，仅当残差足以改变目标神经元在当前不应期/阈值裕量下的
发放时发送 residual event；否则保持本地预测。它需要同时记录阈值裕量、残差事件数、误触发/漏触发、
PM spike/动作一致率和任务成功率。只有该机制在单一 STN/GPe 边上通过三 seed 后，才逐步扩展；若仍不
通过，应承认该小型离散 BG 环不适合作为“连续 latent 直接替换”的正面 RRR 证据，并转向更高维视觉
输入到 DLPFC 的跨核感知通信场景。

本轮命令在 `results/logs/lift_a8_rrrstage1_seed*.log`（五边）和
`results/logs/lift_a8_rrrstn_seed*.log`（三边）中，结果分别在
`results/lift_bdm_snn_alignonly_res10_compactstr_a8_rrrstage1_seed*/` 与
`results/lift_bdm_snn_alignonly_res10_compactstr_a8_rrrstn_seed*/`。`--rrr-first-stage` 当前保守地只
选择三条 STN/GPe 边；该开关仍是研究实验，不改变默认全通信 baseline。

### 13.36 从连续 RRR 到脉冲精确的公共模式 count 通信（2026-07-28）

13.35 的负结果表明，连续浮点 RRR 即使电流 EV 接近 1，仍可能因阈值/复位放大而改变闭环轨迹。为继续
压缩而不牺牲成功率，本轮不再对全部五边调参数，而是先审查单条连接的**精确结构**。当前 BDM-SNN 中
三条递归边的权重严格为常数：`STN->GPe` 和 `STN->GPi` 是 `2x8` 的全 `+0.5` 矩阵，`GPe->STN` 是
`8x2` 的全 `-0.25` 矩阵。因此对二值源 spike (s_i)，完整 MVM 本来就是

\[
I_j=w\sum_i s_i,\quad \forall j.
\]

源端无需发出每个 spike 的神经元身份；只发送一个公共模式 count
\(c=\sum_i s_i\)，目标端以固定权重 \(w\) 广播恢复每一个目标电流。这是 rank-1 的一维通信，但不是
近似 RRR：在当前固定全同权重结构下它是**代数无损**。代码新增 `CommonModeCountCommunication`，每次
重建都以 `torch.equal` 检查完整电流；若有任一元素不同则立即报错。独立 100 个内部 SNN 步检查也验证了
完整网络的所有脑区 spike 和状态逐元素相同。

为避免把不同运行批次的 MuJoCo 结果混在一起，进行了严格 paired 对照：同一当前代码版本、同一 seed、
同一 70 回合协议，仅是否启用 `--common-mode-count-stage` 不同；每个 seed 的 50 个无教师 evaluation
episode 的成功/失败序列逐项完全相同：

| seed | 当前全通信 | 公共模式 count | evaluation 序列是否逐项相同 |
| ---: | ---: | ---: | --- |
| 0 | 42/50 | 42/50 | 是 |
| 1 | 41/50 | 41/50 | 是 |
| 2 | 42/50 | 42/50 | 是 |
| 合计 | **125/150 (83.3%)** | **125/150 (83.3%)** | 是 |

早期历史全通信结果为 136/150 (90.7%)，而本次当前版本 paired 全通信为 125/150；两者来自不同执行批次，
不能将这一绝对差异归因于 count 模块。对于本轮唯一变量，能成立的结论是：**count 压缩相对同版本全通信
没有引入任何额外成功率或轨迹差异。** 后续应把环境 reset 与 MuJoCo 的确定性复核独立出来，才可缩小跨批次
方差；不能将 83.3% 反写成原基线退化。

三条 count 链路的实际统计也要分开理解。以 count 版三个完整运行计，`STN->GPe` / `STN->GPi` 每条每
内部步只产生一个 count（分别 25,906、24,551、23,470 条记录），源端 logical spike 分别为 33,252、
31,526、30,260；`GPe->STN` 对应 25,906/24,551/23,470 条 count 与 39,655/37,072/36,185 个源 spike。
仅在 count 非零时发送的潜在稀疏记录比例也已记录：STN 两边为约 64%，GPe->STN 为约 49%。这说明该
方法减少的是“多源身份事件到一个公共标量”的逻辑有效载荷；尚未规定 count 是 pulse-count、位图还是
多比特数字 payload，故仍**不能声称 AER 包数、bit、能耗或物理链路负载已经按这些比例下降**。

当前可保留的压缩候选是这三条精确公共模式边；DLPFC 广播、纹状体动作身份通路和 GPi->thalamus 仍为全
通信。下一步应为 count 定义硬件可实现的载荷（例如在一个内部时间槽中发送 0--2 / 0--8 的饱和计数，或
按每个源 spike 编码的事件序列），测量 AER 包/bit、峰值率、FIFO 与目标阵列激活，再加入量化、漏发和
链路延迟；同时再考虑更高维、真正有低秩余量的视觉感知到 DLPFC 通信，而不是重新对递归 BG 边施加
连续 RRR 替换。

相关结果目录为 `results/lift_bdm_snn_alignonly_res10_compactstr_a8_countstage_seed*/`；paired 全通信为
`results/lift_bdm_snn_current_full_seed*/`。默认路径仍是全通信，只有显式加入
`--common-mode-count-stage` 才启用这一无损公共模式传输。

### 13.37 RRR 主线转向 DLPFC--Str 的在线低秩权重约束（2026-07-28）

公共模式 count 是固定全同权重带来的无损协议，保留为硬件通信优化点；它不能替代本项目的有损 RRR /
流形压缩主线。13.35 直接把连续 RRR 电流送入递归 STN--GPe--GPi 环，虽有近乎 1 的电流 EV，仍因阈值、
复位和递归放大而使成功率降至约 52%。因此本轮把 RRR 的对象改为**前馈、可塑、且真正具有较大维度的
DLPFC--Str 状态--动作通路**，同时保留下游 BG 环全通信。

当前 `S=200,A=8` 的压缩 Str 模型有两张在线学习表：

\[
W^{D1},W^{D2}\in\mathbb{R}^{200\times8},\qquad
I^{D1/D2}=x^T W^{D1/D2},
\]

其中 \(x\) 是当前 DLPFC 状态编码（本任务通常为 one-hot）。这与原先对最近 128 个 spike--电流样本
临时拟合不同：后者只覆盖近期访问状态，遇到未访问状态会退化至均值，不能代表在线学习到的全状态策略。
本轮先在每一次三因子 TD/STDP 与教师克隆更新之后，对两张完整权重表做截断 SVD：

\[
W\leftarrow W_k=U_{:,1:k}\Sigma_{1:k,1:k}V_{1:k,:}^T=PD,
\]

并继续用 \(W_k\) 完成下一步 SNN 决策和在线更新。该操作是“训练期间持续施加 rank-\(k\) 约束”的
最小验证；只有它通过任务门槛，下一阶段才会将 \(P\) 放在 DLPFC 源核、将 \(D\) 放在 Str 目标核，并将
\(z=x^TP\in\mathbb{R}^k\) 作为实际跨核 latent payload。代码开关为 `--striatum-weight-rank k`，仅允许
`--compact-striatum`；每回合同时写出最终权重 checkpoint，支持独立奇异值审计。

先在同一训练协议的全通信 seed 0 轨迹上审计最终权重。两张 `200x8` 表的有效秩均为 5；rank 4 已保留
D1 的 99.855% 与 D2 的 99.899% 权重平方能量，故选择 `k=4`，而非直接尝试 rank 1/2。`k=4` 使每条
DLPFC--Str 通信在未来分解部署时从 8 个目标动作分量缩至 4 个 latent 分量；这只是连续 latent 数的
缩减，尚不是 AER 包、bit 或能耗结论。

| 同一 seed 0、70 回合协议 | 全通信 | D1/D2 每决策 rank-4 约束 |
| --- | ---: | ---: |
| 教师阶段 Lift 成功 | 15/20 | 15/20 |
| 无教师 evaluation Lift 成功 | **45/50 (90%)** | **43/50 (86%)** |
| 无教师阶段曾抓取方块 | 50/50 | 50/50 |
| 无教师平均决策数 | 149.0 | 157.4 |
| 无教师 PM 静默决策数 | 172 | 308 |
| 无教师内部 SNN 步数 | 19,534 | 24,472 |

因此本轮的谨慎结论是：**rank 4 的在线低秩权重约束在一个完整 seed 中仅少 2 个自主成功回合，且没有
造成抓取阶段失败，表明 DLPFC--Str 权重表存在值得继续研究的低维结构。** 但它还不是“成功证明”：目前
只有一个 seed，成功率下降 4 个百分点；此外 rank 4 增加了 PM 静默和平均决策数，说明脉冲动力学仍受
低秩近似影响，必须完成同 seed 的三 seed 对照后才可接受。

还发现一个更重要的实现/硬件限制：普通截断 SVD 不保持元素非负。原 DLPFC--Str D1/D2 权重初始化与
在线更新都限定为兴奋性正权重，但 rank-4 的 D1 投影在运行中出现最低约 `-0.236` 的元素。软件可用普通
浮点矩阵继续计算，差分 RRAM 也可以编码有符号数；但这不再保持“DLPFC 对 D1 是兴奋性突触”的原模型
解释，也不能直接作为单极性导通阵列的部署方案。因此当前 rank-4 结果应称为**无符号约束的算法可行性
探针**，而不是最终硬件方案。

下一步应固定本轮唯一有希望的对象 DLPFC--Str，并按以下顺序推进：先做 seed 1/2 的 full 与 rank-4
paired 验证；若总体下降不超过预设阈值，再以非负低秩分解（NMF）或“正基底 + 抑制/基线独立通路”替换
普通 SVD，使权重符号和 RRAM 映射一致；最后才显式启用 \(z=x^TP,Dz\) 的跨核 latent 传输，并分别记录
latent 发放、AER 包/bit、FIFO、阵列激活和控制延迟。count 协议仍可与该前馈 RRR 并列存在，但在不同的
公共模式 BG 边上测量，不能将两者的标量数简单相加为物理降耗。

本轮新增实现位于 `braincog/model_zoo/bdmsnn.py` 与
`examples/decision_making/BDM-SNN-Robosuite/lift_bdm_snn.py`。全通信审计结果为
`results/lift_bdm_snn_rrr_audit_full_seed0/`；rank-4 结果为
`results/lift_bdm_snn_rrr_weightk4_seed0/`；完整命令记录在
`results/logs/lift_rrr_audit_full_seed0.log` 与 `results/logs/lift_rrr_weightk4_seed0.log`。

### 13.38 三 seed 配对验证与保持兴奋性的非负低秩分解（2026-07-28）

13.37 的普通 SVD rank-4 探针先完成了严格的三 seed 配对。每对均使用相同的 70 回合协议（20 个教师
回合、50 个无教师且继续在线 TD/STDP 更新的 evaluation 回合）、相同 seed、固定方块与确定性机器人
初态；唯一变化是每次 DLPFC--Str 更新后是否施加 rank-4 SVD 投影。结果为：

| 版本 | seed 0 | seed 1 | seed 2 | 无教师合计 |
| --- | ---: | ---: | ---: | ---: |
| 全通信、无低秩约束 | 45/50 | 46/50 | 45/50 | **136/150 (90.7%)** |
| 普通 SVD，rank 4 | 43/50 | 46/50 | 44/50 | **133/150 (88.7%)** |

SVD 版相对 paired 全通信仅低 3/150（2.0 个百分点），满足“下降不超过 5 个百分点”的探索门槛，故
可以进入非负分解。但它也使三个 seed 的 PM 静默决策从 447 增至 845、内部 SNN 步从 59,476 增至
70,067，且 D1 最小权重稳定出现约 `-0.236`。这进一步证明普通 SVD 不是合适的最终 DLPFC--Str
硬件/生物约束实现。

为保持 DLPFC 到 StrD1/StrD2 的兴奋性权重，本轮实现在线 warm-start NMF。对每条通路的当前正权重
表 \(W\ge0\)，维护两个非负因子：

\[
W\approx UV,\qquad U\in\mathbb{R}_{\ge0}^{200\times4},\quad
V\in\mathbb{R}_{\ge0}^{4\times8}.
\]

首次投影以正 SVD 因子初始化，之后每个在线学习决策后执行 12 次 Lee--Seung 乘法更新并 warm-start：

\[
V\leftarrow V\odot\frac{U^TW}{(U^TU)V+\epsilon},\qquad
U\leftarrow U\odot\frac{WV^T}{U(VV^T)+\epsilon}.
\]

乘法更新与显式下限保证 \(U,V\ge0\)，再以 \(UV\) 写回本地 DLPFC--Str 权重表。因此这一步仍是**在线
低秩权重约束**，还没有把 \(z=x^TU\) 作为真实跨核 payload 发送；它先验证“非负低秩策略本身”能否
闭环工作。

| 版本 | seed 0 | seed 1 | seed 2 | 无教师合计 |
| --- | ---: | ---: | ---: | ---: |
| 全通信、无低秩约束 | 45/50 | 46/50 | 45/50 | 136/150 (90.7%) |
| SVD rank 4 | 43/50 | 46/50 | 44/50 | 133/150 (88.7%) |
| **非负 NMF，rank 上限 4** | **49/50** | **49/50** | **49/50** | **147/150 (98.0%)** |

三个版本的辅助闭环指标如下。NMF 版三个 seed 均 `50/50` 曾抓取方块，平均决策数与全通信基本相同，
PM 静默和内部 SNN 步反而显著减少：

| 版本 | 平均无教师决策数 | PM 静默决策总数 | 内部 SNN 步总数 |
| --- | ---: | ---: | ---: |
| 全通信 | 151.7 | 447 | 59,476 |
| SVD rank 4 | 154.5 | 845 | 70,067 |
| NMF rank 上限 4 | 151.6 | 157 | 55,867 |

NMF 三 seed 全程 D1 最小权重为 `0.0401--0.0403`，D2 最小权重为 `0.0311`，没有负权重。每决策投影的
平均 NRMSE 分别约为 D1 `6.95e-4`、D2 `2.09e-3`；偶发最大 NRMSE 为约 4--5%，是在线权重刚更新时的
短暂校正，而非持续误差。最终两个表的数值秩均约为 3：虽然 NMF 配置提供 4 个 latent 因子预算，但有
一个因子自然冗余。**软件目前仍按 4 因子计算；在显式链路实验中必须先检测并剪枝该分量，不能把数值秩
3 直接声称为已发送 3 个 AER 事件。**

本轮最强的、但仍需限定范围的结论是：**在当前固定 Lift 课程和无限制在线学习下，保持非负的低秩
DLPFC--Str 权重约束不仅保住性能，而且把三 seed 自主 Lift 从 90.7% 提升至 98.0%。** 合理解释是
NMF 同时提供了低维通信候选和非负结构化正则化：它抑制了状态--动作表中会引发 PM 歧义/静默的噪声方向。
这不是“latent 维度变小必然提高性能”的普遍证明，也不是 RRAM/AER 降耗结论；仍未测试随机初态、冻结
部署、器件非理想或真实的跨核 event transport。

实现开关为 `--striatum-weight-rank 4 --striatum-weight-factorization nmf
--striatum-nmf-iterations 12`。新增 NMF 实现在 `braincog/model_zoo/bdmsnn.py`，参数定义与日志记录在
`examples/decision_making/BDM-SNN-Robosuite/lift_bdm_snn.py`。三组结果目录依次为
`results/lift_bdm_snn_rrr_audit_full_seed*/`、`results/lift_bdm_snn_rrr_weightk4_seed*/`、
`results/lift_bdm_snn_rrr_nmfk4_seed*/`，对应命令日志在 `results/logs/lift_rrr_*_seed*.log`。

下一步应从 NMF 的 \(U,V\) 构造真正可独立开关的两核路径：DLPFC 核本地算 \(z=x^TU\)，跨核传输
latent，再由 Str 核本地算 \(I=zV\)，并保留当前 `UV` 本地 MVM 与全通信作逐步对照。首先以 4 个
latent 分量实现，记录连续 latent 标量数和目标电流/动作/成功率；随后根据因子范数剪枝近零分量，单独
验证有效 rank 3。只有把 latent 变成 LIF spike 或定义定长/稀疏 payload 后，才能分别记录并比较
`N_logic`、AER packet、bit、FIFO 与控制延迟。count 的三条无损 BG 公共模式边可在该模型上并存，但
应作为单独的协议变量测量。

### 13.39 从 NMF 权重约束到实际 DLPFC--Str latent 跨核前向（2026-07-29）

本轮保持 `--common-mode-count-stage` **关闭**，以免公共模式 BG 通信与前馈 NMF latent 的效果混杂。
在 13.38 已验证的 NMF 权重约束基础上，新增独立开关 `--nmf-striatum-latent-rank k`，将 DLPFC 与
StrD1/StrD2 视为两个逻辑核。每一个内部 SNN 步不再直接以 `xW` 得到 Str 电流，而是执行：

\[
\underbrace{x\in\mathbb{R}^{200}}_{\text{DLPFC source core}}
\xrightarrow{\ U\ }
\underbrace{z=xU\in\mathbb{R}^{k}}_{\text{跨核连续 latent}}
\xrightarrow{\ V\ }
\underbrace{I=zV\in\mathbb{R}^{8}}_{\text{Str target core}},\qquad U,V\ge0.
\]

StrD1 与 StrD2 分别拥有自己的 \(U,V\) 因子；NMF 在线权重投影仍在每次决策的 TD/STDP / 教师更新后
发生，下一决策才使用新因子。因子尚未生成的最初 3 个内部步显式回退到完整本地 MVM。除这两条
DLPFC--Str 边外，DLPFC--STN、DLPFC--thalamus、Str--GP、STN/GPe/GPi 回路、GPi--thalamus 和
thalamus--PM 均保持全通信；没有启用 count 或旧的连续 RRR。

软件实现将源端 \(U\) MVM 与目标端 \(V\) MVM 分开计为两次阵列激活，并记录 continuous latent 标量、
非零标量、源逻辑 spike、电流 EV/NRMSE 与峰值误差。它是**两核数据流的软件模拟**，并不意味着当前
GPU 上真的发生了物理 PCB 链路传输。

先以 `k=4` 运行完整三 seed 闭环。显式 latent 路径的每个 episode 成功/失败序列、决策数、PM 静默数
均与 13.38 的“仅 NMF 权重约束”版本逐项相同，表明浮点分解本身没有造成新的闭环分叉：

| 版本 | seed 0 | seed 1 | seed 2 | 无教师合计 |
| --- | ---: | ---: | ---: | ---: |
| NMF 权重约束、局部 `xW` | 49/50 | 49/50 | 49/50 | **147/150 (98.0%)** |
| NMF rank-4 实际 latent `xU -> z -> zV` | 49/50 | 49/50 | 49/50 | **147/150 (98.0%)** |
| NMF rank-3 实际 latent `xU -> z -> zV` | 49/50 | 47/50 | 50/50 | **146/150 (97.3%)** |

rank-4 的三 seed 合计 PM 静默为 157、内部 SNN 步为 55,867；rank-3 分别为 157、55,673，基本相同。
因此将 NMF 因子预算从 4 降到 3 没有系统性地破坏当前闭环；虽然 seed 1 少 2 次成功，三 seed 总体仅比
rank-4 少 1/150，仍优于全通信的 136/150。当前选择 **rank 3** 作为最小经验证的 latent 通信候选，
rank 4 作为更保守的性能对照。

两种 rank 的链路重建质量和连续载荷如下。每一条 DLPFC--Str 边在 active 内部步上的 latent 分量均非零；
因此“3/4”是每步连续标量数，不是 spike 数。

| latent rank | 每条边每内部步标量 | rank-4 相对下降 | 电流 EV | 电流 NRMSE | 峰值绝对电流误差 |
| ---: | ---: | ---: | ---: | ---: | ---: |
| 4 | 4 | - | 约 1 | `5.0e-8 -- 6.5e-8` | `2.38e-7` |
| 3 | 3 | **25%** | 约 1 | `2.5e-8 -- 2.7e-8` | `2.38e-7` |

例如 seed 0 的 rank-4 运行中，每条 DLPFC--Str 边有 22,445 个 active 内部步，传输 89,780 个连续
latent 标量；rank-3 seed 0 对应 21,996 个 active 步、65,988 个标量。三 seed 的内部轨迹长度略不同，
所以总标量数不能直接相除作公平能耗比较；在相同 active 步下，rank-3 的固定 payload 量严格是 rank-4
的 75%。每条 active 步还记录一次源端 \(U\) 阵列激活和一次目标端 \(V\) 阵列激活。

最关键的负结果同样需要保留：当前 DLPFC 状态是 one-hot，故完整通信每内部步只产生约一个源逻辑 spike，
而 NMF 的 3/4 个**连续** latent 值全部非零。它证明了状态--动作跨核 MVM 的目标维度和连续 payload 从
8 降至 3/4，但**没有证明 AER event、packet、bit、FIFO 占用、链路能耗或端到端延迟下降**；若直接把
rank 3 称为“三个 spike”或与一个原始地址事件比较，均是不成立的。后续必须为 \(z\) 设计 LIF/量化/
delta 残差事件编码，独立测量 `N_logic`、packet、bit、峰值率与 FIFO，且先验证该编码不会损害本节
已经建立的 97.3% 成功率。

实现位于 `braincog/model_zoo/bdmsnn.py` 的 `striatum_latent_rank` 和
`_nmf_latent_current()`，任务入口位于 `lift_bdm_snn.py`。rank-4 结果在
`results/lift_bdm_snn_rrr_nmfk4_latent_seed*/`，rank-3 结果在
`results/lift_bdm_snn_rrr_nmfk3_latent_seed*/`，命令日志为
`results/logs/lift_rrr_nmfk{3,4}_latent_seed*.log`。

### 13.40 NMF latent 的无损重复向量抑制（sample-and-hold，2026-07-29）

13.39 的 rank-3 latent 将每条边的连续 payload 从 8 降至 3，但仍在每个内部 SNN 步发送 3 个非零
标量；这既不是脉冲稀疏通信，也未必优于 one-hot DLPFC 的一个源地址事件。本轮在不改变 NMF 因子、
神经元动力学或其它 BG 链路的前提下，加入 `--nmf-latent-delta-transport`：仅在源端 DLPFC spike
向量改变时发送一个新的完整 latent 向量，目标端在随后重复源向量的内部步中保持上一次**已解码的
Str 电流**。

若当前源向量 \(x_t=x_{t-1}\)，且本决策内 NMF 因子尚未因 TD/STDP 更新而改变，则

\[
z_t=x_tU=x_{t-1}U=z_{t-1},\qquad I_t=z_tV=I_{t-1}.
\]

此时省略的是重复跨核 vector record 和两次重复 MVM（源端 \(U\)、目标端 \(V\)），而**不是**省略
Str 神经元的每内部步积分、泄漏、阈值和复位。每次在线 NMF 投影后缓存强制失效；episode reset 后缓存
也清空。因此在当前软件模型中该协议对保持段的电流是精确的，称为“代数无损的 sample-and-hold”，
不能泛化为有延迟、丢包、因子异步更新的物理链路仍无损。

三 seed 结果与未抑制的 rank-3 latent 逐项相同：

| 版本 | seed 0 | seed 1 | seed 2 | 无教师合计 |
| --- | ---: | ---: | ---: | ---: |
| NMF rank-3，每内部步发送 | 49/50 | 47/50 | 50/50 | **146/150 (97.3%)** |
| NMF rank-3，重复向量抑制 | 49/50 | 47/50 | 50/50 | **146/150 (97.3%)** |

两个版本的电流 EV、NRMSE、PM 静默和内部 SNN 步也逐项一致；最大 NRMSE 仍为 `2.70e-8`。合并 D1、D2
两条 latent 边、三 seed 的通信记录如下：

| 指标 | 每内部步发送 | repeat sample-and-hold |
| --- | ---: | ---: |
| active latent 样本（含 D1/D2） | 135,838 | 135,838 |
| 传输 vector record | 135,838 | **32,976** |
| 传输连续 latent 标量 | 407,514 | **98,928** |
| 被抑制的重复样本 | 0 | **102,862** |
| 连续标量相对减少 | - | **75.7%** |

各 seed 的重复抑制率为 75.4%、76.0%、75.7%。这比“rank 4 到 rank 3”的固定 25% payload 降幅更大，
原因是一次高层机械臂决策通常包含多个相同 DLPFC 状态的内部 SNN 积分步；并非 NMF 自动产生了稀疏
latent。每个真正发送 record 仍含 3 个连续且非零的浮点值。

因此当前可成立的结论是：**在当前数字 sample-and-hold 模型中，rank-3 NMF latent 加重复向量抑制
可无额外任务损失地减少 75.7% 的跨核连续标量 record，并同步减少同数量的源投影/目标解码 MVM 激活。**
它仍不是 AER 结论：尚未定义 vector record 的 header、定点位宽、时间戳、刷新周期、FIFO、链路延迟或
丢包重传，故不能把 75.7% 直接称为 packet、bit、能耗或物理带宽下降。尤其原 one-hot DLPFC 的完整
事件表示只需一个源地址；要证明此协议胜出，必须在下一步将 latent record 量化为具体 payload，并与
原始 AER 地址编码的 `N_packet/B_link` 在同一时间槽和同一刷新语义下比较。

实现开关为 `--nmf-latent-delta-transport`，仅能和 `--nmf-striatum-latent-rank` 共同使用；本轮仍不
允许与 count 或连续 RRR 同时启用。结果在
`results/lift_bdm_snn_rrr_nmfk3_latent_delta_seed*/`，日志在
`results/logs/lift_rrr_nmfk3_latent_delta_seed*.log`。

### 13.41 轻量定点与链路代理审计：先建立口径，再保留量化失败结论（2026-07-29）

本节不试图构建完整 RRAM/PCB 电路仿真，而是在 13.40 已验证的 rank-3 NMF latent + repeat
sample-and-hold 路径上加入一个最小、显式的通信代理。目标是回答两个更基础的问题：若把实际发送的
latent record 定点化，闭环是否仍稳定？在清楚写明 packet 格式的条件下，record/bit/FIFO 与阵列工作量
的数量级分别是什么？`count` 协议仍保持关闭，且除 DLPFC--StrD1/StrD2 外的 BG 链路均保持全通信。

**定点路径。** 对每个真正发送的新向量（重复样本不发送、目标端保持已经解码的电流）使用固定满量程的
无符号均匀量化：

\[
q=\operatorname{round}\left(\frac{2^b-1}{a}\operatorname{clip}(z,0,a)\right),\qquad
\tilde z=\frac{a}{2^b-1}q,\qquad \tilde I=\tilde zV .
\]

这里 (b) 为每个 latent 标量的 payload 位数，(a) 是离线配置的固定满量程；没有采用每 record 浮点
动态 scale，因此没有把未统计的 scale metadata 偷藏进“payload”。指标同时记录截断率、相对未量化 latent
电流的 NRMSE/峰值误差以及完整闭环成功率。

**链路代理。** 为避免“低维就一定省链路”的错误推断，采用对原 one-hot DLPFC 有利的 baseline：一个源
地址事件携带到 StrD1/D2 的 multicast mask。每个内部 SNN 步为一个同步 slot，并作如下人为约定：

| 项目 | 全通信 one-hot multicast | rank-3 paired latent delta |
| --- | --- | --- |
| record 格式 | 8-bit 源地址 + 2-bit 目的 mask + 8-bit framing | 8-bit framing + D1/D2 各 3 个定点标量 |
| 每个 record 位数（8-bit latent） | 18 bit | 56 bit |
| 发包规则 | 每个 active slot 一个 event record | 仅 source 变化时，把 D1/D2 更新合为一个 vector record |
| 服务/FIFO 代理 | 每 slot 服务一个不大于 64-bit 的 record；arrival 先于 service | 相同 |

因此本代理不含串行化、仲裁、时戳、编码/解码延迟、时钟域跨越、丢包/重传或真实 PCB 带宽；其 FIFO 峰值为
1、排队延迟为 0 只意味着在这个**刻意宽松**的“每 slot 可服务一帧”假设下没有积压，绝不表示真实硬件
延迟为零。阵列侧只报告 activation 与输入 row-drive proxy：baseline 每 active slot 驱动两次 `200->8`
目标 MVM；compressed 每个真实更新驱动 D1/D2 各一次 `200->3` 投影和 `3->8` 解码。row-drive 按本任务
one-hot DLPFC 的实际活动行计数（不是把 200 行都算作已驱动）；它仍不是能量模型，不能把该比例解释成能耗比例。

先运行 seed 0 的 70 回合完整协议（20 回合教师、50 回合无限制在线更新的无教师 evaluation），结果如下：

| 设定 | 无教师成功 | D1 截断率 | D1 电流 NRMSE（相对未量化 latent） | 结论 |
| --- | ---: | ---: | ---: | --- |
| 浮点 rank-3 + delta（13.40 对照） | 49/50 | 0 | `2.7e-8` | 已通过的算法通信对照 |
| 8 bit，固定 (a=8) | 32/50 | 1.06% | 3.16% | 失败；少量 D1 饱和已导致闭环分叉 |
| 12 bit，固定 (a=16) | 19/50 | 6.46% | 11.03% | 失败；在线 NMF 因子尺度漂移导致更严重的 D1 饱和 |
| 8 bit，NMF 等价列归一化、(a=1) | 18/50 | 0 | 0.10% | 仍失败；即使单步电流误差很小，LIF 阈值/复位与在线学习会累积放大 |

最后一行的“列归一化”只利用 NMF 的尺度不唯一性：(U_{:,r}\leftarrow U_{:,r}/c_r,
V_{r,:}\leftarrow c_rV_{r,:}) 不改变 (UV)，但将 one-hot DLPFC 下的 latent 约束到 `[0,1]`。该尝试确认
失败不只是 payload 饱和，而是当前控制闭环对持续定点扰动敏感；归一化试验代码未保留为默认算法，避免改变
已经验证的浮点 baseline。

以第一个 8-bit run 的实际轨迹为例，66,517 个 active internal slots 被 delta 抑制为 7,425 个 paired
latent records（88.84% record 减少）。在上表协议下，full multicast 为 1,197,306 bit，latent 为 415,800
bit，即**此代理口径**为 65.27% bit 减少。注意 row-drive proxy 也应按 one-hot source 的实际活动行而不是
矩阵总行数统计：baseline 为 133,034，compressed 为 59,400（约 55.35% 减少），但后者还多出目标端 `3->8`
解码的三条 active latent 行。这些数字只说明：在“D1/D2 可合帧、8-bit vector payload、repeat-delta 有效”的条件下，重复抑制足以抵消
rank-3 vector 比一个 one-hot 地址事件更宽的劣势。它们**不是**当前可部署方案的降耗结论，因为该 8-bit
设置同时把任务从 49/50 降到 32/50，而且其它 BG 链路尚未纳入该 bit 统计。

本轮实现增加 `--nmf-latent-quant-bits`（0 为浮点、不启用量化）和
`--nmf-latent-quant-scale`，并将量化/截断/电流误差以及 `simple_hardware_link_proxy` 写入每次
`lift_bdm_snn_metrics.json`。实现位置为 `braincog/model_zoo/bdmsnn.py` 与
`examples/decision_making/BDM-SNN-Robosuite/lift_bdm_snn.py`；运行结果分别在
`results/lift_bdm_snn_rrr_nmfk3_latent_delta_q8_seed0/`、
`results/lift_bdm_snn_rrr_nmfk3_latent_delta_q12_seed0/`、
`results/lift_bdm_snn_rrr_nmfk3_latent_delta_q8norm_seed0/`。

本阶段可保留的结论只有两点：（1）浮点 rank-3 + delta 的 record/MVM 减少已建立；（2）一个清晰但简化的
packet/FIFO/bit 代理已接入，且揭示 8-bit fixed vector transmission 在当前在线闭环中**尚不可接受**。下一步
应回到算法/表示层，而不是继续细化硬件参数：例如只在决策边界刷新并在目标端重构保持电流、对 quantization
error 做训练期噪声注入/鲁棒性约束，或把 latent 改为具有明确阈值与误差预算的残差事件编码；每项都必须先
恢复接近浮点 49/50 的同 seed 成功率，再扩大到三 seed 和更完整的硬件模型。

### 13.42 无 ADC 的 binary latent 脉冲链路：初始 IF 版本与失败诊断（2026-07-29）

13.41 的定点实验默认把连续 latent 标量作为 payload，这不符合本项目采用的 RRAM--模拟神经元硬件假设：
源端阵列的列电流应直接接入本核神经元积分电路，核间只传输二进制脉冲，不经 latent ADC。定点代码与其负
结果保留作对照，但后续实验不启用量化，新增独立开关 `--nmf-latent-spike-transport` 实现下列两核路径：

\[
x \xrightarrow{\text{source-core RRAM }U} i_z=xU
\xrightarrow{\text{local IF}} s_z(t)\in\{0,1\}^3
\xrightarrow{\text{AER / binary pulses}}\text{Str core}
\xrightarrow{\text{RRAM }V} I_{\rm Str}=g_{\rm dec}s_zV .
\]

其中 IF 神经元的膜电位、阈值、复位完全在源核；跨核 payload 仅为发生的 latent neuron 地址事件。目标端的
(g_{\rm dec}) 是每个到达脉冲对应的**固定**突触电流增益（可由目标端读电压、脉冲宽度或突触导通标定），
不是发送的浮点幅值。DLPFC--StrD1 与 DLPFC--StrD2 各有 3 个 IF latent neurons；其它 BG 链路依旧全通信，
`count`、连续 sample-and-hold 和定点量化均关闭。在线 NMF 更新后清空 latent IF 膜电位，避免旧因子下积累的
膜电荷被新因子错误解释。

这与 13.39 的连续 `z=xU` 有本质区别：后者把 (z) 直接送入 (V)，没有经过神经元；本节先用最直接的
rate code，将幅值通过重复内部 SNN step 中的 IF 发放次数近似。它是无 ADC 的物理上合理起点，但并非代数
等价的分解。

先做 seed 0、6 回合 smoke，测试三组固定阈值/目标端增益。`t=0.5, g_dec=0.4` 在短窗口中得到 4/4 无教师
成功；但这只是短轨迹，不足以证明在线学习稳定。降低 IF 阈值以获得更密的 rate code 并不稳定：`t=0.25,
g_dec=0.2` 为 2/4，`t=0.1, g_dec=0.08` 为 0/4。`t=0.5, g_dec=0.5` 也为 0/4，说明仅靠全局固定幅值微调
没有单调规律。

对短 smoke 最好的 `t=0.5, source gain=1, g_dec=0.4` 运行完整 70 回合协议（20 教师 + 50 无教师、继续
在线 TD/STDP），结果为：教师 `15/20`，无教师仅 **13/50**，显著低于浮点 rank-3 + delta 的 `49/50`。
两个路径的当前统计如下：

| 指标（完整 seed 0） | StrD1 latent | StrD2 latent | 合计/含义 |
| --- | ---: | ---: | --- |
| active internal samples | 85,443 | 85,443 | 每条边均每内部步执行源投影 |
| binary latent logical spikes | 76,398 | 87,367 | **163,765** 个实际跨核脉冲 |
| 平均每 sample latent spikes | 0.894 | 1.023 | D2 已超过一个脉冲/slot |
| 有至少一个事件的 sample | 65,873 | 78,901 | 目标 V 阵列被实际驱动的 slot 数 |
| 峰值每 sample spikes | 3 | 3 | 三个 latent neuron 可同时发放 |
| 相对连续 (xUV) 电流 NRMSE | 0.930 | 0.548 | rate code 仍有较大电流失真 |

作为公平的事件参照，本任务 DLPFC 状态仍是 one-hot；若两个 Str 目标由同一个 multicast AER source event
驱动，同一条完整轨迹的 baseline 仅有 85,443 个源事件。当前 IF latent 合计 163,765 个跨核 spike，约为
baseline 的 1.92 倍。因此本轮不能声称降低 `N_logic`、packet 或 bit；更重要的是，任务成功也没有保住。

失败主因不是 ADC 缺失或实现退回为浮点，而是**表示失配**：NMF (U,V) 是为连续乘积 (xUV) 形成的，单次
IF 阈值/复位把每个非负幅值替换为 0/1；其幅值只能靠未来时刻的 spike rate 重构。当前每个控制决策的内部
窗口有限，且 NMF 因子和权重持续在线更新，导致 rate code 的时间平均尚未稳定就进入 Str--BG--PM 的非线性
阈值回路。短 smoke 的偶然成功并没有跨越这一长期闭环稳定性门槛。

实现保留在 `braincog/model_zoo/bdmsnn.py`，入口参数位于
`examples/decision_making/BDM-SNN-Robosuite/lift_bdm_snn.py`。完整结果为
`results/lift_bdm_snn_rrr_nmfk3_latent_ifspike_t05_g1_d04_seed0/`，对应日志为
`results/logs/lift_rrr_nmfk3_latent_ifspike_t05_g1_d04_seed0.log`；三个 smoke 结果也保留在
`results/smoke_rrr_nmfk3_latent_ifspike_*/`。下一步不应继续扫单一阈值/增益，而应重新设计 spike encoder：
例如误差反馈的 residual/delta 脉冲、一个决策窗内明确的 pulse-count 预算，或在训练期让 NMF/在线学习直接
看到同一二进制 latent 通道。每个候选先要求恢复接近 49/50 的单 seed 长程成功，再讨论三 seed 与硬件代理。

### 13.43 残差脉冲率编码：保留幅值余量，但当前仍未达到部署门槛（2026-07-29）

为避免 13.42 的 hard-reset IF 在发放后把超过阈值的剩余电流全部清零，本轮实现源核内的 soft-reset
sigma-delta 编码器。对每一条 rank-3 通路和每个 latent 神经元维护本地模拟残差 (r_t)：

\[
r_t=r_{t-1}+z_t,\qquad s_t=\mathbb{1}[r_t\ge\theta],\qquad
r_t\leftarrow r_t-\theta s_t .
\]

跨核仍只发送 (s_t\in\{0,1\}) 的地址事件；残差和阈值比较在源核神经元电路中完成。与 hard reset 不同，
阈值以下部分及一次发放后的剩余部分都会留待下一内部步，因此在一个固定窗口内，脉冲计数可近似连续 latent
的时间积分。为消除在线 NMF 因子固有的尺度不确定性，在**本地**做等价重参数化
(U'_{:,r}=U_{:,r}/c_r,\;V'_{r,:}=c_rV_{r,:}\)，其中 (c_r=\max_iU_{i,r})。这保持连续权重
(UV=U'V') 不变，并确保 one-hot DLPFC 输入时 (z'=xU'\in[0,1]^3)；缩放不会作为 payload 传输。

此外启用 `--fixed-pm-window`，每个高层决策固定执行 30 个内部 SNN step 后才读取 PM。原因是 rate code
需要时间积累：若 PM 在早期首次发放后立即读出，源端的脉冲计数尚不足以表达 (z')。这增加了每个决策的
内部计算时间，因而是性能与时延之间的明确交换，而不是免费的编码改进。

对于理论匹配的 (θ=1, g_{\rm dec}=1)（时间平均中一枚脉冲对应一个归一化 latent 单位），6 回合
smoke 达到 `4/4` 无教师成功；完整 seed 0 得到教师 `15/20`、无教师 **25/50**。这优于 13.42 hard-reset
IF 的 `13/50`，说明残差保留和完整 30-step 积分窗确实改善了二值通信，但仍远低于浮点 rank-3 + delta 的
`49/50`。完整运行的 D1/D2 电流 NRMSE 为 `0.703/0.531`，对应 EV 为 `0.506/0.718`，仍不足以稳定穿过
后续 Str--BG--PM 的阈值回路。

完整轨迹的 D1/D2 binary latent spikes 为 `166,246/233,111`，合计 `399,357`；同轨迹中的 one-hot DLPFC
multicast source events 为 `258,990`，即脉冲 latent 约为 baseline 的 **1.54 倍**。因此尽管任务成功率提升，
本版本仍不具备事件数节省，不能作为最终通信压缩方案。把量化步长再减半（θ=0.5、目标固定增益 0.5）后，
短 smoke 仅 `2/4`，且每 sample 的脉冲数上升到 D1 `1.56`、D2 `1.24`；这表明单纯提高 rate 分辨率会增加
事件流量，未解决闭环鲁棒性问题。

本轮的结论是：**rank-3 未改变，核间 payload 已严格为二进制脉冲；残差率编码可把 13/50 提升到 25/50，
但仍没有同时实现高成功率和低事件数。** 后续不宜继续做阈值扫描。更有希望的下一类方法应让“需要发送的
量”本身是稀疏创新量：在源核维护对目标电流或 latent 的预测，仅用带残差累加的正/负误差脉冲更新目标端
估计；或在训练时以该离散通道替换连续 (xU)，使 NMF/TD-STDP 权重直接适配 spike decoder。前者必须处理
非负单阵列只能自然发送正电流的约束（可用独立基线/负残差通路，不能默认为零成本）；后者则需要把离散
编码纳入训练闭环。

### 13.44 在线 binary-aware NMF 投影：脉冲计数适应的首次尝试（2026-07-29）

13.43 的在线 TD/STDP 确实已经在 binary latent 通路运行，但 NMF 的投影目标仍是连续权重表
(W\approx UV)，没有显式看见 source IF/sigma-delta 的离散输出。本轮新增
`--nmf-latent-binary-aware-projection`，作为可与 soft-reset、因子归一化和固定 30-step PM 窗口叠加的
局部训练期适应。每个在线 TD/STDP 更新后，先做常规 warm-start NMF，再固定归一化 (U\in[0,1])，以
30-step 窗口的可达平均 pulse-count 基底近似它：

\[
Q_{30}(U)=\frac{\operatorname{round}(30U)}{30},\qquad
W\approx Q_{30}(U)V.
\]

随后固定 (Q_{30}(U))，再做 12 次非负乘法更新来调整本地 (V)。这表示训练期的低秩投影直接使用
binary latent 在固定决策窗内能表达的计数基底；`Q`、缩放和残差都保留在本地，链路仍只发送实际的 0/1
脉冲，没有把 pulse count 或浮点 scale 偷渡到跨核 payload。

在 seed 0、6 回合 smoke 中，叠加版教师为 `2/2`，但无教师只有 **1/4**，低于相同编码但未启用该投影的
`4/4` smoke，因此未继续运行昂贵的 70 回合完整 seed。此运行 D1/D2 的相对连续电流 NRMSE 为
`0.581/0.440`，脉冲平均率为 `0.771/1.130` events/sample；低秩重建本身看似接近（最后一次投影 NRMSE
`0.0287/0.0256`），却没有转化为闭环策略稳定性。

这个负结果说明“让 (W) 适应窗口平均 pulse count”仍然过于静态：真实控制不仅取决于窗口平均电流，还取决于
source residual 的相位、每个内部 step 的脉冲时刻、Str/GP/PM 的阈值与复位，以及 TD 更新后因子立刻变化的
时序。故当前不应把该近似称为完整的 binary-aware training，更不应只依据低的表重建误差宣称适应成功。
后续若继续训练期适应，需以实际离散前向的闭环损失/教师动作一致性为目标，采用 surrogate/straight-through
估计或只更新 decoder/阈值的稳定局部规则，并以完整 50 个无教师 episode 成功率作为门槛。

实现位于 `braincog/model_zoo/bdmsnn.py` 的 `binary_aware` NMF 投影，以及
`lift_bdm_snn.py` 的同名开关；smoke 结果为
`results/smoke_rrr_nmfk3_latent_binaryaware_fixedwin_seed0/`，日志为
`results/logs/lift_rrr_nmfk3_latent_binaryaware_fixedwin_smoke_seed0.log`。

### 13.45 实际脉冲 eligibility 的目标端 decoder 教师适应（2026-07-29）

13.44 的离散计数投影仍然只优化一个静态矩阵近似。本轮改为不改源端 (U)、不发送额外 payload，而是在教师
控制的决策后，于 Str 目标核使用**实际收到的 binary latent 脉冲计数**作为局部 eligibility。对每条通路
累计本决策内到达的 (e_r=\sum_t s_{z,r}(t)\)，归一化为 \(\bar e=e/(\sum_re_r+\epsilon)\)，并只改教师
动作列的 decoder 行：

\[
V_{\mathrm{D1},:,a_T}\leftarrow V_{\mathrm{D1},:,a_T}+\eta\bar e,qquad
V_{\mathrm{D2},:,a_T}\leftarrow\max(0,V_{\mathrm{D2},:,a_T}-\eta\bar e).
\]

这可理解为局部三因子更新：pre-synaptic factor 是目标核已接收的实际脉冲，post/modulatory factor 是教师动作
(a_T)，D1/D2 使用相反调制方向。每次更新后同步本地 shadow 表为 (UV)，然后下一决策的常规在线 NMF
投影继续执行；无教师阶段不使用教师标签，仍保留原 TD/STDP。它可叠加 13.43 的 rank-3、soft-reset、因子
归一化和固定 30-step 窗口，且与 13.44 的静态 binary-aware 投影保持独立。

以 `eta=0.02` 运行 seed 0、6 回合 smoke：教师 `2/2`，无教师 **2/4**。前两个教师回合中累计有 90 次
可调用 decoder-update 记录；实际发生非零更新的回合中 D1/D2 的 L1 改变量分别为约 `0.88--0.90` 与
`0.22--0.35`，说明规则已接在实际 binary event eligibility 上，而非退回连续 latent。然而它仍低于未启用
decoder teacher adaptation 的 `4/4` smoke，故不进行完整 70 回合验证。

这说明“教师动作 + 已到达脉冲”虽比 13.44 的静态计数近似更接近真实前向，但只调整 (V) 的单个动作列仍不能
校正 source spike timing、Str 本身阈值动力学及 D1/D2 全动作竞争。当前可保留为一个硬件友好的局部学习接口，
但尚无性能正证据。下一步若继续训练期适应，应将更新目标从单列突触强化提升为实际 PM readout 对教师动作的
时序/排名损失，同时对多个动作列做有界、可解释的调节；仍须先通过完整无教师成功率门槛。

实现新增 `--nmf-latent-decoder-teacher-adaptation` 与
`--nmf-latent-decoder-teacher-learning-rate`，结果位于
`results/smoke_rrr_nmfk3_latent_decoderteacher_fixedwin_seed0/`，日志为
`results/logs/lift_rrr_nmfk3_latent_decoderteacher_fixedwin_smoke_seed0.log`。

### 13.46 二值 latent 训练期适应的后续验证：当前不改善闭环（2026-07-29）

在 13.45 的基础上完成了两种不改变跨核 payload 的训练期适应；两者均保持 DLPFC 到 Str 的 rank-3 NMF、
source RRAM 投影、soft-reset sigma-delta、固定 30 internal-step 窗口和目标核 decoder RRAM。跨核仍只发送
二值 latent spike；未启用 float latent、量化、repeat-delta 或 count。

1. **PM 排名教师规则。** 当本决策的 PM 读出赢家 `a_PM` 与教师 `a_T` 不同，才以本窗口真实收到的 latent
   spike count 为 eligibility：D1 增强 `a_T` 的 decoder 列并抑制 `a_PM` 列，D2 使用相反符号；当二者相同则
   不更新。这避免了最初“正确动作也继续单列增强”的无谓漂移。两段教师轨迹共 90 个 SNN 决策中，只有 7 个是真正的
   错误赢家，故该规则只实际执行 7 次。结果仍为教师 `2/2`、无教师 **0/4**，低于未适应对照的 **4/4**。
   这表明 PM 的动作排名误差不是唯一瓶颈；改变少量 decoder 元素仍会改变 D1/D2 与下游 BG 动力学的平衡。

2. **无动作标签的 local-shadow 校准。** 训练时完整 DLPFC--Str shadow weight 表本来就必须留在本地以完成
   STDP/NMF 投影。于是每个教师决策把实际 latent 脉冲计数 `e=sum_t s_z(t)` 解码得到的窗口总电流，与本地完整表
   的窗口总电流 `I_shadow=sum_t W^T x(t)` 比较，并仅在 Str 目标核更新 decoder：

   \[
   V \leftarrow \max\left(0, V + \eta\frac{e^T(I_{\rm shadow}-g eV)}{\lVert e\rVert^2+\epsilon}\right).
   \]

   这不是把 shadow 或误差发送到另一核：它只是训练期在目标核读取本地监督信号；部署期只保留 `U -> binary spike -> V`
   前向。以 `eta=0.001`，90 次教师决策后 D1/D2 decoder 累积 L1 改变量约为 `(3.14, 3.01)`，无教师同样 **0/4**。
   增加 latent rank 至 4 也仍为 **0/4**，尽管 D1/D2 的瞬时电流近似 EV 提升到 `0.956/0.726`；因此当前失败并非只因
   rank-3 的线性容量不够，更关键的是二值脉冲的时序/阈值/复位与闭环控制之间的失配。

因此，这三类“只调 decoder V”的训练适应（单列教师、PM 排名、shadow 电流）目前都不应作为主结果或并入 baseline。
已保留为可复现消融开关：`--nmf-latent-decoder-teacher-pm-ranking` 和
`--nmf-latent-shadow-calibration`。最好的实际二值结果仍是 13.43 的无适应 rank-3 sigma-delta：seed 0 的完整
无教师评估 **25/50**；它仍不能宣称事件节省（该轨迹 latent 事件为 one-hot multicast baseline 的 1.54 倍）。

相应 smoke 结果分别位于：
`results/smoke_rrr_nmfk3_latent_decoderteacherrankonly_fixedwin_seed0/`、
`results/smoke_rrr_nmfk3_latent_shadowcal_fixedwin_seed0/`、
`results/smoke_rrr_nmfk4_latent_sigmadelta_norm_fixedwin_seed0/`；日志位于 `results/logs/` 下同名文件。

下一步不宜继续叠加局部 decoder 规则；更有信息量的方向是以真实二值前向为计算图，离线训练 `U,V` 和 latent neuron
阈值/泄漏（例如 surrogate-gradient 或 straight-through spike estimator），再冻结为硬件可部署的脉冲通信通路，最后才
在部署期间保留小幅、受稳定性约束的 R-STDP 更新。这样训练目标直接优化 PM/控制行为，而不是把连续电流的局部误差误当作
闭环成功的充分条件。

### 13.47 二值 soft-reset 前向的 STE 因子预热：局部逼近改善，但尚未改善 Lift 控制（2026-07-30）

本轮实现了一个更严格的训练期消融：对每次教师控制后的 DLPFC--Str NMF 因子 `(U,V)`，采用**与部署完全相同**的
soft-reset sigma-delta 前向展开 30 个 slot。每个 slot 的前向值是硬二值事件：

\[
m_{t+1}=m_t+U',\quad s_t=\mathbb{1}[m_{t+1}\ge\theta],\quad
m_{t+1}\leftarrow m_{t+1}-\theta s_t,
\]

其中 `(U',V')=(U/c,cV)` 是本地归一化重参数化；目标电流为 `(\sum_t s_t/30)V'`。反向时仅把硬阈值的导数替换为
sigmoid surrogate（slope 8），即 straight-through estimator (STE)。训练目标仍是本地完整 DLPFC--Str 表的窗口平均
电流，并可加入 spike 罚项；**前向 AER payload 从始至终仍是 binary spike，不传梯度、连续 latent 或误差。**

实现开关为 `--nmf-latent-spike-surrogate-projection`。它只在教师回合运行；评估阶段不会再执行 STE。每个教师
决策先完成原有行为克隆/TD 更新与 NMF warm-start，再做 16 次本地 STE 因子步。这样该实验测试的是“真实脉冲前向的
离线预热”，而非把反向传播放入部署硬件。

seed 0、2 教师 + 4 自主的比较如下：

| 条件 | 自主更新 | 自主 Lift 成功 | D1 / D2 电流 EV | 解释 |
| --- | --- | --- | --- | --- |
| 无 STE、无冻结 | 保留 TD/STDP | 4/4 | 0.692 / 0.868 | 原有短 smoke 对照 |
| 无 STE、冻结 | 无 | 0/4 | 0.974 / 0.685 | 证明该短训练下纯冻结策略本身不能完成任务 |
| STE, eta=0.05、冻结 | 无 | 0/4 | 0.989 / 0.899 | STE 的局部 binary-current 拟合确实改善，尤其 D2 |
| STE, eta=0.05、在线 TD/STDP | 保留 | 1/4 | 0.815 / 0.749 | 低于原始在线对照 |
| STE, eta=0.005、在线 TD/STDP | 保留 | 0/4 | 0.067 / 0.813 | 小步长仍可经后续在线投影漂移，且更差 |

因此本轮得到两个明确结论。第一，**只降低二值脉冲电流的 MSE/提高 EV 不足以保证 BG--PM 的动作排序及机器人闭环成功**；
它是通信保真度指标，不是任务 loss。第二，目前 STE 只在教师轨迹上拟合局部电流，随后在线 TD/STDP 又持续修改完整表并
重新 NMF 投影，二者的目标不一致，导致初始预热不能稳定保留。故不将 STE 版本替换现有 binary baseline，也不扩大为
50 episode 评估。

结果目录为 `results/smoke_rrr_nmfk3_latent_surrogate_fixedwin_seed0/`、
`results/smoke_rrr_nmfk3_latent_sigmadelta_norm_frozen_seed0/`、
`results/smoke_rrr_nmfk3_latent_surrogate_online_seed0/` 与
`results/smoke_rrr_nmfk3_latent_surrogate_lr005_online_seed0/`，日志在 `results/logs/`。

下一步若继续，不能仅以 shadow-current 回归训练 `(U,V)`；应收集教师的真实状态--动作样本，在**二值前向的 PM
动作排序/交叉熵**上训练，至少把 Str--GP--GPi--thalamus--PM 的可微 surrogate readout 纳入损失，或者明确把在线
R-STDP 的更新幅度投影回经过该任务 loss 验证的稳定流形。前者是较完整但工程量更大的端到端训练路线；后者是保留在线
学习且更符合部署目标的受约束适应路线。

### 13.48 二值 latent 的 BG--PM 教师动作损失：最小端到端 surrogate 验证（2026-07-30）

13.47 的 STE 只令 Str 输入电流接近完整 shadow 表，未直接约束动作。本轮将教师标签 `(a_T)` 置入损失：在每个
教师决策后，固定除 DLPFC--Str 以外的 BG 权重，展开与部署相同的 30 个 internal slot：

\[
x\rightarrow {\rm DLPFC}\rightarrow U\rightarrow s_z\in\{0,1\}^3
\rightarrow V\rightarrow({\rm StrD1,StrD2,STN,GPe,GPi,Th,PM}),
\qquad L_{\rm PM}={\rm CE}(\sum_t s_{\rm PM}(t),a_T).
\]

latent 使用 hard binary soft-reset 前向；DLPFC、Str、STN、GPe、GPi、thalamus、PM 则严格匹配 `IFNode` 的
**hard threshold + 全膜电位清零 reset**。反向仅以 sigmoid STE 近似阈值导数。训练仅更新两条 DLPFC--Str 的非负
`U,V` 因子，随后写回 `W=UV`；其他 BG/PM 权重与部署相同且固定。在线 R-STDP/TD 仍是独立模块，本次先冻结自主
阶段以隔离教师 BPTT 的效果。

初版错误地把 BG/PM 的 reset 写成 sigma-delta 减阈值，并绕过 DLPFC IF；它出现近零训练 loss 却 `0/4` 自主成功，
说明 surrogate 不可用。修正为 IFNode 对齐后，教师阶段的平均交叉熵为 `0.142 -> 0.046`，PM 教师动作相对其余
动作的 spike-count margin 为 `7.33 -> 7.76`，表明该**内部 surrogate**能够学会教师动作排序；但冻结自主仍为 **0/4**。
同参数 seed 0 的独立重跑逐项复现相同数值和 `0/4`，说明不是偶然运行错误。

这轮负结果的意义是：即使把教师动作接到了 PM loss，当前两段短教师轨迹和状态机的真实闭环分布仍不足以得到可冻结的
自主策略；surrogate 在教师访问的 state 上正确，不代表机器人因 PM 误差而偏离轨迹后的 OOD state 也正确。并且在
当前流程中行为克隆、TD/R-STDP、NMF 投影和 PM-BPTT 仍顺序更新同一 `W=UV`，存在目标互相覆盖的问题。故不将该版本
扩大为 50 episode，也不把训练 margin 宣称为控制成功。

实现为 `BDMSNN.train_striatum_spike_pm_teacher()`，命令开关为
`--nmf-latent-pm-teacher-surrogate`。两次可复现实验位于
`results/smoke_rrr_nmfk3_latent_pmteacher_resetmatch_frozen_seed0/` 与
`results/smoke_rrr_nmfk3_latent_pmteacher_resetmatch_rerun_frozen_seed0/`。

后续若继续该路线，首要改进不是继续调 STE slope，而是收集更多由学生实际访问的状态并用教师反标注（DAgger），然后在
固定且清晰的训练顺序中做 PM action loss：先基于 teacher/DAgger 数据训练低秩脉冲因子，再把 R-STDP 作为单独、受幅度
约束的部署适应项。否则仅在两条教师轨迹的行为克隆状态上压低 PM CE，会继续遭遇闭环分布偏移。

### 13.49 DAgger 反事实标签接入 binary PM loss：覆盖扩展成功，但短程训练仍未解冻闭环（2026-07-30）

13.48 的 PM loss 仅见过教师顺利到达的状态。本轮把既有 DAgger 阶段接入该损失：在 DAgger 回合中，**环境始终执行
学生的 PM/option 动作**；在学生实际到达状态 `(s)`，教师只产生反事实标签 `(a_T(s))`，并执行
`L_PM=CE(sum_t s_PM(t),a_T(s))`。这使训练样本包含学生偏离教师轨迹后真正访问的状态，而不改变机器人当步执行的动作。

为分开解释更新来源，增加 `--nmf-latent-pm-teacher-surrogate-supervised-only`：教师与 DAgger 样本上关闭 TD/R-STDP、
旧的逐行 `behavior_clone` 及 NMF shadow 重投影，只有 PM action loss 更新 nonnegative rank-3 `(U,V)`；训练启动时仅做一次
NMF 初始化。冻结测试不更新任何参数。新开关为
`--nmf-latent-pm-teacher-surrogate-dagger-labels`。

最小协议为 2 个教师执行回合 + 2 个学生执行 DAgger 回合 + 4 个冻结自主回合。DAgger 中学生共访问 285 个 SNN align
状态，相对初始教师支持集的离散 state OOD 比例为 `48.4%`、连续几何 OOD 比例为 `33.7%`；这确认它确实提供了常规教师
轨迹外的训练样本。第一段 DAgger 240 个样本的平均 PM loss 为 `2.79`、教师 margin 为 `-0.36`，第二段为
`1.20/-0.24`：学生状态显著比教师状态困难（教师 loss 约 `0.67`、margin 约 `8.5`），且短程更新尚未把 PM 排序翻正。

在严格隔离的实现中，DAgger 回合的 `dagger_label_decisions=0`，验证没有遗留的逐行行为克隆写入；但最终冻结自主 Lift
仍为 **0/4**。关闭旧克隆前后的轨迹和结果也一致，故负结果不是由更新混杂引起。结果位于
`results/smoke_rrr_nmfk3_latent_pmteacher_daggerisolated_frozen_seed0/`。

结论是 DAgger 解决了“看不到学生访问状态”的数据问题，却没有解决“在这些状态上 rank-3 binary latent 的 PM 动作
可学习性/样本效率”问题。当前不应仅把 DAgger 回合从 2 增至更多来追逐分数：每步在线 PM-BPTT 既昂贵，又在 OOD 状态
上下降缓慢。下一步若继续应改为**收集后离线批训练**：先存储 teacher/DAgger 的 `(state, teacher action)` 数据，再对
这些数据多 epoch、随机 mini-batch 训练 PM loss，保留独立验证集与 early stopping；完成后再测试冻结策略，最后才单独
恢复幅度受限的 R-STDP。这样才可区分“数据量不够”与“模型/低秩容量确实不足”。

### 13.50 离线 mini-batch 二值 latent PM 训练：流程已通，冻结闭环仍未通过

在 13.49 的严格隔离设置上，实现了 `--nmf-latent-pm-offline-train`。教师和 DAgger 阶段不直接做 PM-BPTT 更新，
而是只收集 `(DLPFC one-hot state, teacher action)`；结束后随机划出验证集，对剩余样本按 mini-batch 同时展开独立的
30 个内部 SNN 时间步。前向严格保持部署路径：DLPFC 脉冲经每条 DLPFC→Str 链路的 rank-3 非负 `U` 投影、soft-reset
sigma-delta latent IF 神经元产生**二值脉冲**，再由 `V` 解码为 Str 电流；BG--PM 的硬阈值/全复位前向和 STE 反向均保留。
batch 只是离线训练的并行化，不表示部署时把多个状态一起发送。每 epoch 在不更新参数的验证子集上计算 PM 交叉熵，选择
最小验证损失的 `(U,V)` 并 early stopping；训练数据同时保存为 `pm_offline_teacher_dataset.npz` 以便复审。

完整最小协议为 `2 teacher + 2 DAgger + 4 frozen evaluation`，seed 0，rank-3 binary latent，固定 30 内部步，
batch size 32，5 epoch。共得到 `248` 条 SNN-align 标签（训练 `198`、验证 `50`）；验证 PM loss 从 epoch 1 的 `1.6595`
降到最佳 epoch 4 的 `1.5617`，验证 teacher-margin 从 `0.066` 升至 `3.040`。因此离线优化确实学到了 held-out 动作
排序，而不是训练过程/数据管线失效。教师和 DAgger 回合均成功（`2/2`、`2/2`），但冻结自主为 **0/4**；失败回合始终停在
align 阶段，最小 xy 误差约 `6.95 mm`，未能触发下降阶段。结果目录为
`results/smoke_rrr_nmfk3_latent_pmteacher_offline_dagger_frozen_seed0/`。

这一步把“单样本在线 BPTT 样本效率太低”的解释排除了部分：批训练可以改善离线的 PM 分类指标，却仍不能保证递归闭环
稳定。更可能的主因是数据按单步标签切分后，丢失了“应持续朝同一方向动作直到误差跨格”的时序控制目标；小的首步偏差会
把下一状态带离训练支持，随后 PM 的单步分类即使局部正确也无法重新收敛。下一步应优先把 teacher/DAgger 数据组织为
短时序段，并以 progress / option-transition 为辅助损失或状态变量训练，而不是继续单纯增加相同的单步分类 epoch；这能
直接检验缺失的是时序策略表示，还是 rank-3 binary communication 本身的容量。

### 13.51 先审计再扩容：动作塌缩不是 rank-3 容量瓶颈（2026-07-30）

为避免平均交叉熵掩盖控制失败，新增 `audit_pm_offline_dataset.py`：加载保存的 checkpoint，对每个已采集 one-hot 状态从
清零后的真实二值脉冲/IF 前向展开 30 步，记录教师标签--PM top-1 混淆矩阵、macro-F1、状态标签冲突和训练/验证 state
重叠。它是单状态读出诊断，不替代机器人闭环测试。

对 13.50 的 `248` 样本，随机验证集中 `11/11` 个离散状态同时出现在训练集中，故原验证集并非 state-level 泛化测量。
真实部署前向对 `248/248` 个样本均读出 `+x`：表面 top-1 `75.8%` 恰为 `+x` 类别比例，而四类存在标签的 macro-F1
仅 `0.216`。将 latent 从 rank-3 扩至 rank-8、保持同一 seed、teacher/DAgger、loss、30 步窗口和评估不变后，最佳验证
loss 的确从 `1.562` 降至 `1.164`，但仍 `248/248` 为 `+x`、冻结自主 `0/4`。并且每内部步的 D1/D2 latent 事件从约
`2.83/3.00` 增至 `7.73/8.00`，没有事件效率优势。因此现阶段不能把失败归因于 rank-3；扩维只增加了饱和脉冲。

随后加入逆类别频率 PM loss。它没有产生正确的四类读出，反而使 PM 动作完全并列，动作由随机 tie-break 决定，冻结仍
`0/4`。这说明问题不仅是损失的多数类偏置：soft-reset 编码的 `U` 在 30 步内接近满发放（rank-3/8 对多数已见状态均为
所有 latent 通道约 30 spike），导致大量 state 在 `V` 前已失去可区分性。

### 13.52 方位均衡采样与系统/PM 归因消融

为测试“训练支持不足”而非简单堆叠轨迹，新增 `--fixed-cube-direction-curriculum`：仅在 teacher/DAgger 收集时将固定
cube 位置轮换四个 x/y 方位偏移；冻结评估仍保持原始固定位置。此轮获得 `321` 个样本、`19` 个状态，横向标签为
`[+x,-x,+y,-y]=[209,47,47,18]`，比原来的 `[188,26,23,11]` 更均衡。使用 rank-3、二值 soft-reset、逆频率 loss 的
`2 teacher + 2 DAgger + 4 frozen` 运行中，冻结系统级 Lift 达到 **4/4** 成功；其实际横向动作含多种方向。

但此结果不能写成“PM 已学会”：离线 PM 验证 loss 最好为 `1.3865`、margin 为 `0`；状态审计仍显示 PM 对 `321/321`
样本为 action-0 或并列，macro-F1 `0.197`。四个成功评估中有大量 `progress_memory`（53--85 次）和少量
`progress_value` 动作接管。为验证因果归属，新增 `--checkpoint-path` 和 `--disable-align-progress-safeguard`：对**同一
checkpoint**固定权重、关闭 progress-memory 与 progress-value 的动作干预后，纯 PM 评估立即为 **0/4**，四回合均为
模糊 PM 平局后的随机动作，教师一致率仅 `21.7--27.9%`。

所以当前最准确结论是：方位均衡采样让“BDM-SNN 二值低秩通信 + 基于真实 xy 进度的方向记忆”这个**系统**在固定 Lift
场景达到 4/4，但 PM/低秩通信本体尚未学得可部署横向策略。后续不应把该 4/4 用作压缩算法成功证据；应优先修正 source
latent 的饱和编码（而不是继续增加 rank），并在没有 progress safeguard 的纯 PM 对照上达到高 macro-F1 和无教师闭环，
再恢复系统安全模块。

### 13.53 先恢复 binary latent 的状态可分性：对比度脉冲编码

13.51 的根因是 `U` 列最大值归一化后多数元素接近 1；DLPFC 每个 internal slot 都发放，于是阈值为 1 的 soft-reset
latent IF 几乎每步发放，状态在源端被压成同一 pulse code。为直接检验该问题，新增
`--nmf-latent-spike-contrast-encode`。对每个源端 RRAM `U` 列，以固定的列最小导通值作为**本地校准抑制参考电流**，先做

\[
\tilde U_{ij}=\frac{U_{ij}/\max_iU_{ij}-\min_i(U_{ij}/\max_iU_{ij})}
{\max_i(U_{ij}/\max_iU_{ij})-\min_i(U_{ij}/\max_iU_{ij})+\epsilon}.
\]

再把 `\tilde U` 的列电流送进同样的 binary soft-reset IF；`V`、跨核 payload、BG、PM 和动作空间不变。参考电流是每列
静态校准量，可由源核偏置/抑制支路实现，**不额外跨核传输浮点数或 ADC 数据**。projection gain 同时从 `1.0` 降至 `0.5`
以避免动态范围恢复后再次每步满发放。

在 `2 teacher + 2 DAgger + 4 frozen`、rank-3、逆频率 PM loss 中，teacher/DAgger 仍保留原 progress safeguard 以收集
有效学生访问状态；仅最终冻结评估使用新增的 `--disable-align-progress-safeguard-evaluation-only`，因而动作不能被
progress-memory/value 接管。结果为 teacher `2/2`、DAgger `2/2`、**纯 PM 冻结 `1/4`**。成功的评估回合有 `73` 个唯一
PM 决策，横向动作 `[+x,-x,+y,-y]=[67,24,2,0]`，与反事实教师的一致率为 `72.0%`，并进入固定状态机阶段完成 Lift。
另外三个回合仍在 align 失败，反事实一致率为 `19.6%`、`35.8%`、`31.7%`。这证明对比度编码已把“永远 +x”的硬塌缩改为
部分有效的状态/历史相关决策，但尚不足以稳定闭环。

需谨慎解释一个诊断差异：逐样本审计会在每个 one-hot state 前清零 latent residual，仍报告 action-0；而真实部署与此次
闭环会在连续决策间保留 sigma-delta residual，这正是 soft-reset 编码的预期有记忆行为，因而能产生多种 PM 动作。前者测
无历史的静态可分性，后者测实际部署的有状态通信编码；后续应新增**按真实轨迹顺序重放**的 latent/PM 混淆矩阵，不能再用
清零审计单独判定该编码的闭环可分性。结果目录为
`results/smoke_rrr_nmfk3_latent_contrast_g05_balancedcollect_pm_only_seed0/`。

### 13.54 连续轨迹诊断：失败来自未训练的 residual 相位，而非动作索引错误

新增 `--record-decision-trace`，保存每个 SNN 决策的离散状态、两条 latent 的决策窗口 pulse count、PM/thalamus count、
网络动作、执行动作、反事实教师标签和 xy 误差；并新增 `--episode-seed-offset`，可用已保存 checkpoint 重放固定的
Robosuite reset。这使“清零单状态审计”和“连续、有 residual 状态的部署”可以严格区分。

对 13.53 checkpoint 纯 PM（关闭 progress safeguard）的成功/失败连续轨迹作重放：两者前 30 个 align 决策都选择教师所需
的 `+x`，xy 误差分别下降约 `88.4 mm` 与 `90.7 mm`。随后均在离散 state `58` 首次失稳：典型 binary code 为
`D1=[0,0,0]`、`D2=[14/15,15,15]`，PM count 变为约 `[2,3,0,0]`，错误偏向 `-x`。成功轨迹在中段虽有 15 次
`-x`，随后回到正确方向，75 个 align 决策总体教师一致率 `77.3%`，最终误差降至 `1.48 mm`；失败轨迹中段后无法恢复，
后 111 步一致率仅 `17.1%`，xy 误差累计恶化约 `202.6 mm`。

因此此前“自主输出 +x 较多”不是 action index 或 `+x/-x` 映射错误；相反，初段 `+x` 是正确控制。当前核心失配是：离线
PM loss 将每个 `(state,label)` 从 **零** latent/BG 膜状态独立展开，而部署的 soft-reset sigma-delta residual 与 BG 膜状态
跨高层决策保留。训练从未对 state 58 的上述 residual 相位施加正确 action-0 的损失，故一旦早期 PM/tie 随机扰动将系统
带入该相位，策略可能恢复也可能发散。下一步最有信息价值的算法不是再调增益，而是按 episode 顺序训练短连续段：在段内保留
latent residual/BG 状态，对每个决策的反事实 teacher action 累加 PM loss（truncated BPTT），并以同一连续轨迹的 PM
一致率、恢复率和纯 PM Lift 成功率为验收；随机 mini-batch 单步 CE 不再作为主要验收指标。

### 13.55 首次短连续窗口 PM-BPTT：实现完成，当前未提升纯 PM 闭环（2026-07-31）

按 13.54 的诊断，将 `train_striatum_spike_pm_teacher()` 扩展为输入 `(batch, decision, state)` 的短时序段；新增
`--nmf-latent-pm-offline-sequence-length`。每段连续 3 个高层决策（每个仍展开 30 个内部脉冲步）：段内连续保留 DLPFC、
Str、STN、GPe、GPi、thalamus、PM 的膜状态以及两条 source-side sigma-delta residual；每个决策末只清空 PM 投票计数并
对对应教师动作累计交叉熵；段与段之间重置状态，故这是 truncated BPTT，而不是把完整 episode 的长梯度硬展开。采集端也改为
按 teacher/DAgger episode 保存连续样本，避免把相邻片段随机拼接。部署仍是一状态一次的 **binary latent AER 脉冲**，训练的
batch/sequence 不表示跨核传输浮点 latent。

首次运行（rank-3 NMF、contrast encode、gain 0.5、`2 teacher + 2 DAgger + 4 frozen`、最终关闭 progress safeguard、
sequence 3、batch 8、3 epochs、seed 0）收集 `248` 个标签并形成 `82` 段。最初版本暴露了必要的数值问题：D2 decoder 的
非负梯度可增至 `5.1e9`，使验证 loss 达 `2.31e14`。这不符合有限 RRAM 导通范围，故加入局部 decoder 梯度裁剪（绝对值 10）
和最大导通/解码值 16；修正后 D1/D2 decoder 最大值分别为 `0.445/0.974`，最佳验证 loss 为 `2.051`，训练数值稳定。

但数值稳定**不等于控制成功**：两次纯 PM 评估均为 **0/4**。裁剪后的第一组评估 PM 动作以 `+y` 为主（154/240），反事实教师
一致率仅 `7.1--8.3%`；学习率从 `0.01` 降至 `0.001` 后动作分布较丰富，四回合一致率提升至 `24.2%`、`39.2%`、`25.4%`、
`30.8%`，但仍未进入 descend、Lift `0/4`。对应目录为
`results/smoke_rrr_nmfk3_contrast_g05_sequence3_clipped_pm_only_seed0/` 和
`results/smoke_rrr_nmfk3_contrast_g05_sequence3_lr001_pm_only_seed0/`。

这一步的结论不是“连续状态无用”，而是更具体：实现已消除了训练/部署 reset 条件不一致，且避免了导通值爆炸；但是仅有 2 条教师
和 2 条 DAgger 轨迹所覆盖的 `82` 个片段，无法覆盖自主第一步误差后的新 residual/连续状态分布。下一步不宜继续盲调 rank 或
增益；应在受控 2D 对齐子任务上以纯 PM policy 收集失败轨迹并追加 DAgger 标签（覆盖 state-58 及其连续 phase），或将 PM
训练目标改为与执行动作后的 xy 误差下降直接相关的局部 TD/R-STDP 信号，再以相同关闭 safeguard 的闭环成功率进行判断。

### 13.56 聚合 DAgger 的 episode 均衡采样：纯 PM 首次达到约 50%（2026-07-31）

13.55 的“仅失败轨迹”Dagger 采集了 `1345` 个 label / `448` 个 sequence-3 片段，PM 验证 loss 达 `1.150`，但冻结仍为
`0/4`，且 PM 又塌缩为 `+x`。原因是数据权重而非简单样本总数：成功教师轨迹通常约 45 个决策，失败学生轨迹可达 240 个，
把所有 truncated windows 等概率采样会使失败绕行轨迹压倒正确的接近--停下--换向行为。

因此新增 `--nmf-latent-pm-offline-episode-balanced`：只在离线训练 mini-batch 中按**采集 episode**等概率抽取 sequence
window；不修改 rank-3 binary latent、source IF、AER payload、BG/PM、状态机或评估。标准聚合数据为 2 条教师成功轨迹加 6 条
纯 PM 实际执行的 DAgger 轨迹；后者仍由教师仅提供反事实标签而不接管动作。共 `1274` label、`424` 个连续片段。期间修复了
一次新采样路径的 `Counter` 引用错误，修复后从头使用相同 seed 完整重跑。

训练末的冻结纯 PM 两回合为 **1/2** Lift 成功；随后固定完全相同 checkpoint，新进程、关闭 progress safeguard、额外运行 4
回合为 **2/4** 成功。成功回合的 align teacher agreement 为 `80.0%`、`84.1%`、`89.1%`、`80.0%`（后两项来自扩展评估中的
成功样本），并进入 descend；失败样本约 `19--33%`，通常停留在 align。该结果是当前最强的纯 PM 证据：学习模块在 8-action
rank-3 nonnegative binary-latent 通信下可以完成 Lift，但表现仍约 50%、对早期脉冲相位和 tie-break 轨迹敏感，尚不能称为可靠
控制。结果目录：`results/smoke_rrr_nmfk3_contrast_g05_sequence3_episodebalanced_dagger6_rerun_seed0/` 与
`results/eval_rrr_nmfk3_contrast_g05_sequence3_episodebalanced_seed0/`。

这一轮也给出方法论结论：在短连续 BPTT 下，普通 window-level 验证 loss 并非闭环成功的可靠代理（episode-balanced 模型的最小
验证 loss `2.076` 反而高于不均衡聚合的 `1.361`，但前者是唯一达到约 50% 纯 PM 成功的版本）。后续应保留 episode 均衡与数据
聚合，并把纯 PM 多 seed 闭环成功率、进入 descend 率、以及失败回合的 phase-conditioned teacher agreement 作为模型选择标准；
再考虑用真实 progress reward 驱动小幅在线三因子/STDP 更新来修复剩余闭环偏离。

### 13.57 BPTT 只用于离线因子训练；多 seed 与在线可塑性初测（2026-07-31）

需要明确区分两类学习。短连续 BPTT 只在离线 teacher/DAgger 数据收集结束后使用：它为 rank-3 NMF 的 source `U` 与 target
`V` 建立暂时的反向梯度图，段内保存 latent residual/BG/PM 膜状态，更新后图即释放。实际部署从不执行 BPTT，也不存储反向图；
每个控制时隙只执行 source RRAM MVM -> local IF/sigma-delta -> binary AER pulse -> target `V` MVM -> BG/PM 前向。部署时可选的
在线学习是本地 eligibility trace 与奖励/TD 三因子调制，再把更新后的 DLPFC->Str 权重重新投影到 rank-3 非负因子；它不是 BPTT。

为避免 seed-0 的小样本结果被过度解释，使用相同 `2 teacher + 6 pure-PM DAgger + 6 frozen pure-PM evaluation` 协议新增 seed 1、2。
seed 0 的已有冻结结果为 `3/6`（训练末 1/2 + 新进程扩展 2/4），seed 1 为 `0/6`，seed 2 为 `4/6`，合计 **7/18 = 38.9%**。
因此当前最准确表述是“该 binary-latent PM 在部分 seed 可完成 Lift，但方差很大”，而非稳定的 50% 成功。目录：
`results/multiseed_rrr_nmfk3_sequence3_episodebalanced_seed1/`、
`results/multiseed_rrr_nmfk3_sequence3_episodebalanced_seed2/`。

随后从 seed-2 已训练 checkpoint 开始、无教师、无 BPTT、关闭 progress safeguard，测试实际部署中的 TD-modulated 三因子更新与每步
rank-3 NMF 重投影。`three_factor_learning_rate=0.005` 在 8 个连续回合仅 **1/8** 成功，且失败回合 D1/D2 权重 L1 改变量最高
约 `41.4/13.8`，说明直接在线更新会破坏已有离线策略。将学习率降为 `0.0005` 后为 **3/8** 成功，成功回合的权重改变量较小，但
失败回合仍可达到 `39.0/13.7`。结果目录为 `results/online_td_rrr_nmfk3_sequence3_episodebalanced_seed2/` 与
`results/online_td_lr0005_rrr_nmfk3_sequence3_seed2/`。

结论：保留在线 STDP/三因子可塑性在系统目标上是合理的，但当前无门控的 reward/TD 信号会在失败轨迹上累积大幅漂移，尚不能声称提升。
下一步应先加一个局部更新门控（例如只在 xy 误差实际下降、PM 有唯一胜者且 TD 误差为正时更新，并记录 per-decision drift budget），
再与冻结 checkpoint 做同 seed、同 episode 顺序对照；这样在线适应才会从“持续改坏也持续写入”的机制变成有证据约束的部署学习。

### 13.58 门控在线三因子更新：发现全局 NMF 重投影是部署失配（2026-07-31）

实现并测试了最小门控：仅在 align、当前决策 xy 误差下降超过 `0.1 mm`、PM readout 有唯一胜者、且 TD error 为正时，才允许
局部 D1/D2 三因子写入；日志同时记录各条件计数和实际 applied 次数。以相同 seed-2 checkpoint、相同 8 个环境回合、无教师、无
BPTT、无 progress safeguard 测试，`lr=5e-4` 得 **0/8**，虽然第 0 回合的 D1/D2 L1 漂移从无门控的 `41.4/13.8` 降至
`0.36/0.25`。将名义学习率降为 `5e-5` 后为 **2/8**；加每突触 `1e-3` 预算仍为 `0/8`，更严格的 `1e-5` 预算实际触发
10--48 次 clipped writes，仍为 `0/8`。结果目录为 `results/online_td_gated_lr0005_rrr_nmfk3_sequence3_seed2/`、
`results/online_td_gated_lr00005_rrr_nmfk3_sequence3_seed2/`、
`results/online_td_gated_budget001_rrr_nmfk3_sequence3_seed2/` 和
`results/online_td_gated_budget1e5_rrr_nmfk3_sequence3_seed2/`。

该负结果定位了比门槛更根本的部署问题：当前在线代码先更新展开的 `200x8` DLPFC->Str 权重表，随后每个决策对**整个表**做 rank-3
NMF 重投影以恢复通信因子。即使局部写入被截断，全局重分解仍会重分配大量权重、改变 `U,V` 并影响 sigma-delta 脉冲相位；因此
连接 L1 指标不等于局部写入预算，且这一过程也不对应硬件上的局部 RRAM 在线学习。结论是停止在该路径上继续调学习率。下一步应将
在线三因子规则直接定义在 target-side decoder `V`：源端 `U` 与 latent IF 固定，只有已接收 binary latent spike 对应的 `V` 行，
按局部 post/pre eligibility 与 TD/reward 门控做有界写入。这样无需每决策 NMF 重分解，才是 binary AER--RRAM MVM 部署一致的在线学习。

### 13.59 Factor-native decoder 在线学习：硬件路径一致，但尚未提高成功率（2026-07-31）

据 13.58，新增 `adapt_striatum_spike_decoder_three_factor()` 与 `--online-decoder-three-factor`。每一高层决策结束时，source
`U`、contrast calibration、latent IF 和 sigma-delta residual 均固定；Str target core 只读取本窗口实际收到的 D1/D2 binary
latent event count (c)，构造局部 eligibility (c/(\sum c+\epsilon))，并对当前执行 action 的 decoder 列进行

\[
\Delta V^{D1}_{:,a}=+\eta\,\delta_{TD}\,c/\sum c,\qquad
\Delta V^{D2}_{:,a}=-\eta\,\delta_{TD}\,c/\sum c.
\]

更新仍受“正 xy progress + unique PM + positive TD”门控和单元写入上限约束；随后只以 `W=UV` 同步本地影子表，**不执行任何
NMF 重分解、不会改写 U，也不会改变跨核 payload 格式**。这正对应目标核 RRAM decoder 交叉阵列按收到 AER 脉冲做局部导通更新的
抽象。

在与 13.58 相同 seed-2 checkpoint、相同 8 个部署回合的对照中，`eta=5e-4`、write budget `5e-4` 得 **3/8**，每回合 D1/D2
因子改变量仅约 `1e-4--1e-3`；降至 `eta=5e-5` 后写入约 `1e-5--1e-4`，仍为 **3/8**，成功/失败回合完全相同。相比此前全局
投影路径，局部写入不再造成数十量级连接漂移，说明工程与硬件映射已修正；但严格逐回合比较，冻结 checkpoint 在相同前 6 个环境
种子为 `4/6`，decoder 在线版本为 `1/6`，故不能称为在线性能提升。结果目录为
`results/online_decoder_td_gated_lr0005_seed2/` 与 `results/online_decoder_td_gated_lr00005_seed2/`。

因此保留 factor-native decoder 更新作为后续可部署的在线 STDP/三因子接口，但不再进行无目标的 learning-rate 扫描。要获得收益，
后续应先解决 reward credit：目前 TD 信号只反映单步 shaping progress，未区分“暂时正向但最终错过对齐”与真正成功轨迹；更合理的
下一步是按 episode 成功/失败做 delayed eligibility consolidation，或以教师/安全监督仅在失败轨迹上写入反事实动作列，并和冻结
checkpoint 同 seed 对比。

### 13.60 Episode-holdout 与 warm-up recovery BPTT：修正验证泄漏，但全量重叠片段会退化（2026-08-01）

独立审查与连续 trace 复核确认，当前失败通常不是一开始就错误：seed-1/2 的多个失败回合前 10 个 align 决策均与教师一致，首个
错误多出现于第 33--35 个决策、state `57` 附近（教师应为 `+x`，PM 可读出 `+y/-y`）；成功轨迹也可能在该处短暂出错，但随后
恢复。因此主要缺口是“有 residual/BG 历史时的恢复”，而非静态 state 分类。审查还指出旧离线验证按 window 随机划分，会让同一
episode 的相邻窗口同时落在训练/验证中，不能评价连续轨迹泛化。

据此将离线 PM 训练改为完整 episode holdout，并加入 `--nmf-latent-pm-offline-warmup-decisions` 和
`--nmf-latent-pm-offline-sequence-stride`：每个样本先连续 warm-up 3 个决策（保留 latent residual/BG/PM 状态、不计算 loss），
再对后续 3 个决策计算 PM loss；stride 1 形成重叠窗口。为避免 180 个内部脉冲步的 STE 图出现非有限编程值，增加了梯度
`nan_to_num` 防护和非有限 loss 的显式失败检查。最小 smoke test 得到 227 个窗口、3 条训练 episode / 1 条验证 episode，证明
数据、梯度与 episode 隔离管线可工作。

完整 seed-2 对照仍为 `2 teacher + 6 pure-PM DAgger + 6 frozen`，仅替换该训练协议。它得到 903 个窗口、6 条训练 episode / 2
条验证 episode、最佳 held-out loss `1.365`，但冻结纯 PM 为 **2/6**；原 episode-balanced 非重叠协议为 **4/6**。新协议使一个
原先失败回合进入 descend，却让另外多个回合退化，说明不能把所有 240-step 失败轨迹尾部的重叠窗口都当成同等恢复数据：它们会再次
在训练分布中压倒短的成功/可恢复片段。目录：
`results/recovery_bptt_warmup3_episodeholdout_lr0001_seed2/`。

当前应保留 episode-level holdout 与 warm-up 机制（它们是正确的评估/状态初始化修正），但停止使用无差别 stride-1 全轨迹
overlap。下一步改为根据 trace 自动定位每条失败轨迹的首次教师不一致点，仅上采样其前 3、后 6 个决策的 recovery windows，保持
教师成功轨迹和正常 DAgger 片段的 episode 均衡；验收使用相同多 seed frozen PM success、enter-descend、首错后恢复率及后半段
agreement，而非 held-out CE 单独选模型。

### 13.61 首错局部 recovery windows：避免失败尾部过采样，seed-2 为中等结果（2026-08-01）

实现 `--nmf-latent-pm-offline-recovery-windows`：常规连续 BPTT 窗口改回 stride 3；仅对 DAgger episode 中首次
`PM network_action != counterfactual teacher_action` 的位置，加入其前 3、后 6 决策可覆盖到的 stride-1 窗口。这样 source-side
binary latent、U/V、硬件前向和部署均不变，仅令离线 BPTT 反复看到真正导致闭环偏离的非零相位，而不让 240-step 失败尾部获得
不成比例权重。

相同 seed-2 完整协议得到 329 个窗口（6 训练 / 2 验证 episode），冻结纯 PM **3/6**：优于全量 overlap `2/6`，但低于原非重叠
episode-balanced 协议 `4/6`。一个 agreement 仅 `27.9%` 的评估回合仍进入 descend，表明首错局部样本确实改变了局部恢复行为；
但总体没有超过基线。held-out loss `3.064`、margin 为负，而仍有三回合成功，再次说明在当前强状态性闭环中，window CE/margin 不能
单独选模型。目录：`results/recovery_bptt_firsterror_seed2/`。

因此保留该有针对性的 recovery-window 机制作为候选，但不在单一 seed 上再调参。下一步以原 episode-balanced 协议和
first-error recovery 协议在 seed 0/1/2 上做 paired 6 回合冻结评估，报告差分成功、enter-descend、late agreement 和 tie rate；
只有跨 seed 有正向证据才将其并入主方法。在线 decoder 更新继续冻结，直到离线恢复策略获得稳定收益。

### 13.62 首错局部 recovery windows 的三 seed 配对验证：当前不应作为默认方法（2026-08-01）

完成了上节定义的严格配对验证。每个 seed 都使用相同的初始 checkpoint、`2 teacher + 6 pure-PM DAgger + 6 frozen pure-PM
evaluation`、固定 cube/robot start、关闭 progress safeguard、每个动作决策固定 30 个 SNN 内部时间步；唯一变量是离线 BPTT
训练集是否加入首错局部 recovery windows。二者均采用 episode holdout、warm-up 3、监督长度 3、stride 3、class/episode
balanced sampling。评估阶段没有教师、没有在线写入、也没有动作修正，故表中均为纯 PM 闭环结果。

| seed | 常规窗口成功 | 首错 recovery 成功 | 差分 | 常规进入 descend | recovery 进入 descend | align 教师一致率（常规 / recovery） | PM 平均并列数（常规 / recovery） |
|---|---:|---:|---:|---:|---:|---:|---:|
| 0 | 4/6 | 1/6 | -3 | 4/6 | 3/6 | 0.651 / 0.398 | 1.043 / 1.255 |
| 1 | 2/6 | 2/6 | 0 | 2/6 | 2/6 | 0.433 / 0.422 | 1.116 / 1.114 |
| 2 | 2/6 | 3/6 | +1 | 2/6 | 4/6 | 0.486 / 0.546 | 1.361 / 0.996 |
| 合计 | **8/18 (44.4%)** | **6/18 (33.3%)** | **-2/18** | 8/18 | 9/18 | 0.523 / 0.455 | 1.173 / 1.122 |

这里“进入 descend”是状态机确认横向对齐、开始下降接近方块的回合比例；它比最终 Lift 成功宽松，所以 seed-2 的 recovery 可提高
该比例而不必然转化成整体成功。PM 平均并列数是每次读出中最高活动并列的动作数，越接近 1 越明确；它只提供读出歧义的诊断，不能
替代成功率。结果目录分别为 `results/recovery_control_warmup3_seed{0,1,2}/`、
`results/recovery_firsterror_warmup3_seed{0,1}/` 和 `results/recovery_bptt_firsterror_seed2/`。

结论是：该局部上采样在 seed-2 有正例、在 seed-1 持平，但 seed-0 显著退化，三 seed 总成功率从 44.4% 降至 33.3%，平均 align
一致率也下降。因此它证明“首错后的状态/相位是可被定向训练的”这一诊断方向值得保留，却**没有**提供把 recovery windows 写入主训练
配方的证据。后续默认回到常规非重叠、episode-balanced 训练；若再研究恢复能力，应先记录首错状态、动作类别、latent residual、
PM/BG membrane 与恢复结局，按可解释失败类型分层采样或引入显式恢复状态，而不是对所有首错邻域使用同一种额外 CE 权重。

### 13.63 PM 脉冲计数 margin：降低并列但未形成稳定闭环收益（2026-08-01）

对 13.62 的纯 PM trace 做进一步诊断后，确认大量失败来自固定 30-step 窗口的**并列脉冲读出**。三 seed 常规模型中，只有
40.5%--67.2% 的 align 决策有唯一 PM 胜者；在并列/静默回退决策中，教师动作一致率仅约 11%--38%，明显低于唯一胜者。尝试将
readout 切到 thalamus 后，离线轨迹的唯一胜者率和正确可选率均未稳定优于 PM，故未修改部署读出。

因此在 `train_striatum_spike_pm_teacher()` 增加可选的计数 margin 项，而没有改变任何部署前向：对教师动作的 PM 窗口计数
`c_y` 与最大竞争计数 `max_{a!=y} c_a`，训练时附加

\[
L_{margin}=\lambda\max(0,m-(c_y-\max_{a\ne y}c_a)).
\]

本次取 `m=1`、`lambda=1`，即要求教师动作在训练的硬脉冲前向中至少领先一个脉冲。它只通过 STE/BPTT 更新 NMF `U,V`；部署仍为
`U MVM -> source IF/sigma-delta -> binary AER -> V MVM -> 原 BG/PM`，仍无教师、无 progress safeguard、无在线更新。

| seed | 常规成功 | count-margin 成功 | 差分 | 常规 / margin 进入 descend | 常规 / margin 唯一 PM 胜者率 | 常规 / margin 平均并列数 |
|---|---:|---:|---:|---:|---:|---:|
| 0 | 4/6 | 2/6 | -2 | 4/6 / 2/6 | 0.551 / 0.498 | 1.603 / 1.528 |
| 1 | 2/6 | 1/6 | -1 | 2/6 / 3/6 | 0.672 / 0.663 | 1.381 / 1.400 |
| 2 | 2/6 | 4/6 | +2 | 2/6 / 4/6 | 0.405 / 0.609 | 1.690 / 1.533 |
| 合计 | **8/18 (44.4%)** | **7/18 (38.9%)** | **-1/18** | 8/18 / 9/18 | 0.543 / 0.590 | 1.558 / 1.487 |

seed-2 的探针提升与机制一致：唯一胜者增加、并列减少，成功由 2/6 到 4/6；但 seed-0、1 的成功下降，合计仍从 44.4% 降为
38.9%。所以“增加硬脉冲 count margin 能改变 PM 读出歧义”得到支持，但固定 `m=1, lambda=1` **不是稳定的主训练配方**。此外，首次
运行漏掉 contrast encoding、以及两次并行启动缺失 rank 参数，均在参数校验/配置核对时发现；前者的 0/6 和后两次无输出均不纳入表格。

后续不建议继续盲扫 margin 强度。更有信息量的方向是以真实的部署风险为目标：只对被记录为并列且教师动作属于最高并列集合的样本
施加很小的 tie-breaking margin，并按 action/state 支持度分层验证；同时仍需保持完整三 seed paired 评估。当前代码将 margin 默认为
零，因而原有实验和默认行为不变。

### 13.64 Binary latent 的 residual 边界 reset 与 pattern 审计：residual 是当前编码的一部分，不应直接清零（2026-08-01）

在 rank-3 NMF binary latent 路径中，source core 每个内部时间步先计算 `Ux`，再由本地 IF/sigma-delta membrane（soft reset residual）
生成 0/1 AER 脉冲；目标 core 只接收该二值脉冲并做 `V` MVM。因此残差在 source core 内部，不是跨核发送的浮点 payload。为分离这段
短期状态的作用，新增 `--nmf-latent-spike-reset-each-decision`：仅在每个高层动作决策起点清空 D1/D2 source residual；Str、STN、GPe、
GPi、thalamus、PM 的状态以及长期权重均不重置。开关默认关闭。

先按最小 A/B 诊断做 seed-2 checkpoint-only 冻结部署：两组均从
`results/recovery_control_warmup3_seed2/bdmsnn_final_state.pt` 加载同一训练权重，固定 cube/robot start，关闭教师、DAgger、在线写入与
progress safeguard，执行 6 个最多 240 决策的纯 PM 回合、每决策固定 30 个内部 SNN 时间步。A' 保留 residual，B 只在部署端 reset residual。
由于此次 A' 与 B 都保存了完整时序 trace，它们的比较不依赖旧版 aggregate event-count trace。

| 条件 | Lift 成功 | PM 唯一胜者率 | 平均并列数 | 与反事实教师动作一致率 | 完整 30x6 binary pattern 数 | 跨教师动作 collision |
|---|---:|---:|---:|---:|---:|---:|
| A'：residual 持续 | 0/6 | 43.3% | 1.99 | 32.7% | 286 | 15/286 (5.2%) |
| B：每决策 reset residual | 0/6 | 14.4% | 2.64 | 45.6% | 14 | 1/14 (7.1%) |

结果目录为 `results/residual_persistent_deployonly_seed2/` 与 `results/residual_reset_deployonly_seed2/`；后者的动作明显集中在 `+x/+y`
（网络动作计数 `811/456/152/21`），且 6 回合均未进入抓取成功。B 的反事实动作一致率表面略高，却没有转化为唯一读出、状态机推进或任务成功，
不能将该单一分类指标误读为控制提升。更关键的是，reset 把时序 binary code 的可观察模式数从 286 降至 14，并使 PM 并列显著增加；这支持
source residual 目前承担了 sigma-delta 脉冲相位/误差累积的功能，而不是可无损删除的历史噪声。

`audit_latent_patterns.py` 现保存 ordered 30 slots x (D1 rank-3 + D2 rank-3) = 180-bit pattern，并报告模式数、不同教师动作之间的模式
collision、类内/类间 Hamming 距离以及 `(state, teacher action)` 稳定性；若不同 trace 宽度被混合会显式报错，避免静默截断。审计仅用于
诊断 code separation 与 residual phase sensitivity，**不**统计 AER packet/bit，也不能据此宣称已得到非线性流形或通信节省。

本轮的保守结论是：不做“训练和部署同时 reset”的 C 组，也不把 reset 加入默认方法；同一 checkpoint 下 A' 和 B 都是 0/6，尚没有支持
reset 提升闭环成功的证据。下一步应保留跨决策 residual、将其视作 source IF 神经元的合法本地状态，并在多 seed 的成功/失败 trace 上按
`(离散状态, 教师动作, residual 末值, binary pattern, PM tie)` 分层分析：先判定现有失败主要是 code collision、PM readout 还是训练覆盖不足，
再决定是否值得引入保持 xy 拓扑的感知编码或受限的本地 recurrent latent，而不是直接叠加复杂“流形”模块。

### 13.65 配对 replay 修正与边界泄漏：全保留 residual 优于 reset 或 0.5 衰减（2026-08-02）

13.64 的 checkpoint-only A'/B 首次重放存在两个可重复性缺口，故其 `0/6` 与原训练进程 `2/6` 不能用于强因果比较：第一，独立重放从
episode 0 开始，环境 reset seed 为 `seed+0..5`，而原 evaluation 实际使用第 8--13 个 episode 的 seed；第二，固定 30-step PM readout
会出现并列，旧代码由全局 RNG 随机选其中一个动作，teacher/DAgger/BPTT 已消耗该 RNG，独立进程无法复现同一并列选择。模型 checkpoint 本身
包含 DLPFC--Str NMF `U,V`、连接权重和 mask，并无权重遗漏。

据此新增 `--reseed-policy-per-episode`：仅用于 checkpoint 配对 replay；每个 episode 用
`seed + episode_seed_offset + episode_index` 初始化独立 policy RNG。它使两个消融分支共享同一环境 reset、同一 PM tie-break / exploration
随机序列；默认关闭，因此不改变原 curriculum 训练行为。同时新增 `audit_closed_loop_failures.py`，从 trace 统计反事实教师一致率、PM 唯一胜者率、
平均并列数和首个网络--教师动作分歧。教师标签在自治评估中只作诊断，不参与动作或写入。

使用 seed-2 的同一冻结 checkpoint、`episode_seed_offset=8`、6 个配对环境回合、30 个内部 SNN 时间步、无教师/在线更新/safeguard，得到：

| source residual 边界规则 | Lift 成功 | align 决策数 | PM 唯一胜者率 | 平均并列数 | 180-bit pattern 数 | cross-label collision | D1+D2 logical latent spikes |
|---|---:|---:|---:|---:|---:|---:|---:|
| 全保留 (`decay=1`) | **4/6** | 767 | **62.7%** | **1.80** | **244** | **1.2%** | **52,071** |
| 半保留 (`decay=0.5`) | 1/6 | 1,113 | 52.8% | 2.17 | 92 | 8.7% | 75,878 |
| 每决策清零 (`reset`) | 1/6 | 1,245 | 41.4% | 2.26 | 14 | 7.1% | 81,391 |

三组目录分别为 `results/residual_persistent_pairedrng_seed2/`、`results/residual_decay05_pairedrng_seed2/`、
`results/residual_reset_pairedrng_seed2/`。这是一组有效的单 checkpoint/单 seed 配对诊断，说明当前 `rank-3 NMF + source sigma-delta + binary
AER` 编码依赖跨决策的 residual phase：减少这段 source-local 积分历史既降低时序 code 多样性与唯一读出，也让控制停留更久，从而增加总 logical
spike 数。它不是多 seed 性能宣称，更不是“非线性流形”证据；但足以否定直接 reset 或固定 `0.5` 泄漏作为当前默认。

现阶段应保留 `decay=1`，不在这个 checkpoint 上继续扫更多泄漏率。三 seed 原始自治 trace 的首错也集中在接近目标后的 align 状态：例如 seed 0/1
多出现在 state 56--58、seed 2 多在 57--59，教师动作为 `+x`，但 PM 会输出 `+y/-y` 或并列。成功回合也可能经历首错后恢复，所以“首错”是风险标记
而非充分失败原因。下一步应在**训练数据收集阶段**记录完整 180-bit pattern 和 source residual，并用相同配对 RNG 协议跨 seed 比较
`(state, teacher action)` 的 pattern 稳定性、混淆与 PM readout；只有确认是静态 code collision 而非闭环覆盖/读出问题，才引入保持 xy 邻域的
连续感知编码或受限 source-local recurrent latent。

### 13.66 Binary latent 扩展冻结评估：60 回合结果支持暂不改为拓扑编码（2026-08-02）

在 13.65 的单 seed 配对诊断后，对三个既有常规训练 checkpoint 分别执行 20 个全新、冻结的自治 Lift 回合（共 60 回合）。每一组均保持
rank-3 NMF、source soft-reset sigma-delta residual 全保留、D1/D2 binary AER、固定每动作 30 个内部 SNN 时间步、无教师/DAgger/在线写入/
progress safeguard；`episode_seed_offset=20` 避开原训练和最初 evaluation，`--reseed-policy-per-episode` 固定每回合 PM 并列选择。结果是：

| 训练 seed | Lift 成功 | 进入 descend | 发生 grasp | 平均决策数 | D1+D2 logical latent spikes/决策 | 完整 180-bit pattern 数 | 跨教师动作 pattern collision |
|---|---:|---:|---:|---:|---:|---:|---:|
| 0 | 8/20 | 9/20 | 9/20 | 193.6 | 61.89 | 712 | 19/712 (2.7%) |
| 1 | 7/20 | 9/20 | 9/20 | 196.8 | 63.29 | 1,399 | 19/1,399 (1.4%) |
| 2 | 12/20 | 13/20 | 13/20 | 174.0 | 51.26 | 525 | 14/525 (2.7%) |
| 合计 | **27/60 (45.0%)** | **31/60 (51.7%)** | **31/60 (51.7%)** | 188.1 | 58.8 | -- | -- |

近似 Wald 95% 区间为 32.4%--57.6%，仅说明该小样本下 binary-latent 纯 PM 闭环稳定地处于“可完成但仍有明显 seed 方差”的水平，不能与浮点
latent 的性能混为一谈。成功回合的 counterfactual teacher agreement 为 80.5%--88.1%，失败回合仅 21.4%--29.6%；PM 唯一胜者率也在成功
回合更高（67.1%--77.5%）而失败回合为 43.8%--66.7%。因此失败首先表现为 align 阶段的闭环动作/读出偏离，而不是后续固定状态机抓取失败。

完整时序审计的跨教师动作 exact-pattern collision 仅 1.4%--2.7%，故没有出现“大量不同教师动作落在完全相同 binary code”的直接证据；反之，
同一 `(state, teacher action)` 在跨决策 residual、BG/PM 历史不同下可有多个 pattern，这是当前 sigma-delta 动态编码的预期表现，不能直接称为
编码错误。结合 45.0% 成功率，本轮**暂不引入拓扑保持 xy 编码**：它会同时改变源端 DLPFC 感知维度、STDP eligibility 定义和训练分布，若现在
引入，将无法区分提升来自空间泛化还是训练重构。后续若需要继续提升，应先用新训练数据保留完整 trace，并在未见 xy 邻域上做专门泛化对照；只有
发现“邻近位置但未见格子”的动作/latent 表示明显断裂，才以重叠 RBF/place-cell 输入替换 one-hot 10x10 state，同时保持 downstream BG/PM、
rank-3 和 binary AER 不变。结果目录为 `results/binary_latent_extended20_pairedrng_seed{0,1,2}/`。

### 13.67 未见邻域诊断与 topology place-cell pilot：one-hot 泛化不足，但简单平滑编码未带来收益（2026-08-02）

此前的 teacher、DAgger 与 evaluation 均使用同一固定 cube 初始坐标 `(0.0088, 0.0069)`，所以 13.66 的 45.0% 只验证该固定工作点，
并不检验空间泛化。先不改网络，以三个既有 one-hot checkpoint 对未训练的相邻 cube 起点作冻结测试：分别沿 `+y` 与 `+x` 平移 12.5 mm，
即一个 10-bin residual grid 的 in-range cell 宽度；其余条件保持 rank-3、binary AER、30 内部步、full residual、无教师/在线写入，并用独立
episode seed offset 重放 PM tie-break。三 seed 结果如下：

| 初始 cube 位置 | 训练 seed 0/1/2 成功 | 总成功 | 进入 descend |
|---|---:|---:|---:|
| 已训练固定位置 | 8/20, 7/20, 12/20 | **27/60 (45.0%)** | 31/60 (51.7%) |
| 未见 `+y 12.5 mm` | 5/10, 2/10, 2/10 | **9/30 (30.0%)** | 10/30 (33.3%) |
| 未见 `+x 12.5 mm` | 2/10, 2/10, 3/10 | **7/30 (23.3%)** | 7/30 (23.3%) |

这说明当前 one-hot 10x10 表在相邻但未训练的空间工作点存在明显泛化退化，但不能单独归因于通信 latent：教师数据只覆盖一个 cube 起点，
同时覆盖、闭环误差累积和 PM readout 都会影响结果。目录为 `results/binary_latent_neighbor_y125mm_seed{0,1,2}/` 与
`results/binary_latent_neighbor_x125mm_seed{0,1,2}/`。

为做最小机制验证，新增默认关闭的 `--align-topology-place-cells`。它不增加 DLPFC 的 200 个感知神经元，不改 NMF rank-3、BG/PM 或跨核协议；
对于连续 xy residual，在同一 dominant-axis block 内以双线性权重共同激活最多四个相邻格。源端本地输入为

\[
x=\sum_{i\in\mathcal N(x,y)}\alpha_i e_i,\qquad \alpha_i\ge0,\quad\sum_i\alpha_i=1,
\]

并保持总 DLPFC 电流不变；三因子 eligibility / reward 更新也按 \(\alpha_i\) 分配。之后仍严格执行
`local U MVM -> 3 IF/sigma-delta neurons -> binary AER -> V MVM -> Str/BG/PM`，没有跨核浮点 latent。该表示使临近连续位置的感知活动连续，
但不自动保证学习到的 BG/PM policy 连续。

以相同固定点重新训练 topology seed-2（`2 teacher + 6 DAgger + 6 frozen`）获得固定点 4/6，说明该开关可完整运行且未使基本任务失效。
从其 checkpoint 以与 one-hot seed-2 相同的 10 回合/起点/PM RNG 冻结对照：

| 表示（seed-2） | 已训练固定点 | 未见 `+y` | 未见 `+x` | PM 唯一胜者率（固定 / +y / +x） | logical latent spikes/决策（固定 / +y / +x） |
|---|---:|---:|---:|---:|---:|
| one-hot | 5/10 | 2/10 | 3/10 | 约 50.6% / 12.2% / 45.0% | 51.3 / 80.1 / 62.0 |
| bilinear place cells | 6/10 | **0/10** | 2/10 | 89.0% / 84.3% / 86.6% | 104.6 / 157.9 / 146.6 |

place-cell pilot 让 PM 计数更少并列，却未提高教师一致率或未见邻域成功；`+y` 明显退化，且实际 binary latent logical event 约增加 2 倍以上。
故不能把“PM 更唯一”当作控制提升，也不能用更多激活的源端群体声称通信节省。当前结论是保留该可复现开关用于后续研究，**不替换 one-hot 主基线，
不将此 pilot 称为流形学习或有效创新**。更有前景的下一步不是继续扫 place-cell 宽度，而是扩展 teacher/DAgger 的 cube 起点覆盖，明确训练、验证、
未见邻域集合；之后再比较 one-hot 与拓扑编码，才能区分“数据覆盖不足”与“表示缺乏空间归纳偏置”。

### 13.68 训练起点覆盖的受控负诊断：先解决 DAgger 失败尾部的标签塌缩（2026-08-02）

为检验 13.67 的未见邻域退化是否主要来自训练位置单一，保持 one-hot 10x10 residual state、compact Str、rank-3 NMF、source-local
soft-reset sigma-delta、binary AER、30 内部 SNN 步和其余训练超参数不变，新增默认关闭的
`--fixed-cube-local-coverage-curriculum`。它只在训练期改变 cube 起点，冻结 evaluation 仍在 base
`(0.0088, 0.0069)`；局部偏移为 `6.25 mm`，故不等于保留给测试的 `+x/+y 12.5 mm` 邻域。新增的
`--dagger-teacher-prefix-decisions` 与 `--fixed-cube-local-coverage-teacher-only` 都默认 0/关闭，用于把后续的
DAgger recovery 与额外完整教师演示分开控制。

第一组等训练量试验仍为 `2 teacher + 6 DAgger + 6 frozen`，让这 8 条训练回合覆盖 base 周边的多个位置；第二组保留两个
base teacher、只移动 6 条 DAgger。两组均在 seed-2 训练内冻结 evaluation 得 `0/6`，再从同一 checkpoint 做配对的 10 回合
base / `+x` / `+y` replay；第一组为 **0/10、0/10、0/10**，因此没有报告其余外推结果作为性能证据。

根因不在于“多位置本身无效”，而在于 DAgger 的学生一离开教师轨迹便长期卡在 align；当前实现会把整段最多 240 决策的学生轨迹都加入
offline PM 序列训练。新位置的反事实几何教师在这些偏离状态中几乎总标为 `+x`，使数据集从原 base 训练的四类
`[+x,-x,+y,-y]=[570,173,22,178]` 变为局部覆盖的 `[1525,0,5,0]`（第一组）和 `[1524,0,6,0]`（第二组）。离线交叉熵/
margin 虽可很低，却只说明网络学会了这个偏置标签分布，无法完成横向闭环；对应冻结 PM 因此始终无法进入 descend。

随后做了“增加正确空间支持”的对照：`6 teacher + 6 base-position DAgger + 6 frozen`，其中前两条教师示范仍在 base，其余四条为四个
`(+/-6.25 mm,+/-6.25 mm)` 角点。此组明确增加了 4 条完整教师轨迹，不是等数据量比较；但仍产生 `[1694,0,16,0]` 的标签分布，训练内冻结
base 也为 `0/6`，故停止 `+x/+y` 外部评估。结果目录依次为
`results/binary_latent_localcoverage_train_seed2/`、`results/binary_latent_localcoverage_v2_train_seed2/` 与
`results/binary_latent_localcoverage_v4_teacher6_seed2/`。

这给出一个比“继续调 place cells 或 latent rank”更强的结论：当前瓶颈是**长失败 DAgger 尾部在序列监督中的样本选择/动作类别塌缩**，不是
已被证明的空间表示或 binary latent 不可用。下一步应只采集教师轨迹及学生仍在教师可恢复带内的短 recovery windows，并按动作类别和起点
分层平衡这些完整短窗口；先验收四个横向动作均有实质支持、base 不低于原 4/6（或 checkpoint 5/10）后，才重新测 `+x/+y`。这既避免把教师
永久放进部署，也不让不可恢复的学生尾部主导离线 BPTT。

### 13.69 稳定基线恢复检查：课程开关默认关闭时不存在 `+x` 塌缩（2026-08-02）

为区分新覆盖课程的失败与基础 binary-latent 系统是否被改坏，完整重放此前稳定的初始化与训练协议：加载
`results/smoke_rrr_nmfk3_latent_contrast_g05_balancedcollect_pm_only_seed0/bdmsnn_final_state.pt`，使用 seed-2、固定 base cube、
`2 teacher + 6 DAgger + 6 frozen`、rank-3 NMF、source soft-reset sigma-delta、binary AER、30 内部步；所有 local coverage、teacher-only
coverage 与 DAgger teacher-prefix 开关均保持默认关闭。

训练数据恢复为四个横向动作 `[+x,-x,+y,-y]=[570,173,22,178]`，没有单一 `+x` 类别塌缩；训练期 DAgger 为 `4/6` Lift 成功，训练内冻结为
`2/6`。后者样本较小且受 PM tie-break 影响，因而另从既有稳定 checkpoint
`results/recovery_control_warmup3_seed2/bdmsnn_final_state.pt` 做 10 条独立、逐回合重置 policy RNG 的冻结 replay，得到 **4/10** Lift
成功，**5/10** 进入 descend 且发生 grasp，平均 193.5 个动作决策。这与既有 seed-2 的 `4/6` 配对 replay 和 `12/20` 扩展 replay 一致地证明：
当前 one-hot + rank-3 binary-latent 主基线仍能闭环完成任务，新增训练课程的默认关闭不会破坏它。

稳定重训练结果在 `results/binary_latent_stability_replay_seed2/`，checkpoint replay 在
`results/binary_latent_stability_checkpoint10_seed2/`。后续任何覆盖/恢复采样改动都应先检查离线标签直方图至少保有多个横向动作、再检查 base
冻结成功，未通过这两个门槛不得进入未见 `+x/+y` 泛化评估。

### 13.70 稳定 checkpoint 的 3x3 空间部署图：base 可完成，但 y 向泛化窗口仍窄（2026-08-02）

在 13.69 恢复稳定基线后，不修改权重、不启用教师、DAgger、在线更新或 progress safeguard，而是固定
`results/recovery_control_warmup3_seed2/bdmsnn_final_state.pt` 做更大空间范围的部署测试。cube 起点取 base
`(0.0088,0.0069)` 周围的 `x,y∈{-12.5,0,+12.5} mm` 3x3 网格；每格 3 个独立回合，固定 30 内部 SNN 步、rank-3
binary AER、逐回合 policy RNG reset，合计 27 回合。

| y 偏移 \\ x 偏移 | -12.5 mm | 0 mm | +12.5 mm |
|---|---:|---:|---:|
| +12.5 mm | 0/3 | 0/3 | 0/3 |
| 0 mm | 1/3 | 1/3 | 1/3 |
| -12.5 mm | 0/3 | 0/3 | 0/3 |

总计为 **3/27** Lift 成功、**5/27** 进入 descend/发生 grasp。不同格点的三回合共享相同 episode seed 序列，所以表格刻画的是该
checkpoint 对初始位置的响应，不是多 seed 的总体成功率估计；中线 base 的 `1/3` 也不应取代 13.69 的 10 回合 `4/10` 验证。完整目录为
`results/binary_latent_spatialmap_{xm125,x0,xp125}_{ym125,y0,yp125}_seed2/`。

这张图的实际用途是给下一轮数据收集划边界：当前模型并非所有相邻方向都同样退化，而是 y 偏移后无法进入 descend。因而下一步不应继续将
240 决策的外围失败尾部无差别加入 DAgger；应由教师把学生带入 y 误差较小、仍能产生多方向反事实标签的可恢复带，采集固定长度的局部恢复窗口，
并在离线训练前检查每一动作类别的样本数。只有 base 成功率保持且标签直方图不塌缩，才将该训练扩展到 3x3 网格并重新测图。

### 13.71 闭环成功率提升：仅在 PM 模糊时启用本地进度 safeguard（2026-08-02）

13.69 的 stable checkpoint 纯 PM 在同一 10 回合、同一 episode seed / PM tie-break 流中为 `4/10` Lift 成功。保持**完全相同的
checkpoint、环境、binary AER、30 内部步和无在线写入**，只取消 `--disable-align-progress-safeguard` 后，系统级结果提升为
**9/10** Lift 成功、**9/10** 进入 descend / grasp。

该 safeguard 不是教师：它不访问 teacher action，不读取目标答案，也不更新网络权重。PM 有唯一胜者时动作不变；只有 PM 并列/不确定时，
`AlignProgressMemory` 才复用“上一动作实际使 xy 误差减少”的方向，`AlignProgressValue` 以同样实测进度的 state-action value 解除少数
并列。该 10 回合中原始 PM unique 动作有 413 次，progress-memory / progress-value 分别介入 179 / 20 次，说明提升来自对真实
闭环进度的短时纠错，而不是把全部控制移出 SNN。

同一 checkpoint 的 3x3 起点图（每点 3 回合）也从纯 PM 的 `3/27` 提升到 **9/27**，进入 descend / grasp 为 `11/27`：

| y 偏移 \ x 偏移 | -12.5 mm | 0 mm | +12.5 mm |
|---|---:|---:|---:|
| +12.5 mm | 0/3 | 2/3 | 1/3 |
| 0 mm | 3/3 | 2/3 | 0/3 |
| -12.5 mm | 0/3 | 1/3 | 0/3 |

这将当前结果清楚分成两层：**系统级 SNN + 进度记忆**在已训练工作点显著可靠（待扩大回合数确认）；**纯 PM SNN**仍是 4/10，且
较大 y 偏移的策略支持不足。不能把 9/10 说成权重本身的提升，也不应继续把 240-step 的失败尾部加入训练。合理下一步是先扩大 base
的 9/10 评估；若稳定，再把“已验证进度方向”的短期 token 作为可学习的 DLPFC context，以 teacher/DAgger 的短 recovery windows 蒸馏给
SNN，并保持纯 PM、系统级 safeguard 两种部署口径并报。

### 13.72 扩大 base 部署验证：系统级成功率稳定为 26/30（2026-08-02）

对 13.71 的同一冻结 checkpoint 进一步运行 30 个全新 base 起点回合（`episode_seed_offset=80`，逐回合 PM tie-break RNG 重置），
所有条件不变：无 teacher、无 DAgger、无在线权重更新、rank-3 binary AER、30 内部步，仅保留 PM 模糊时的本地 progress safeguard。
结果为 **26/30 = 86.7%** Lift 成功，**30/30** 进入 descend 并发生 grasp，平均 138.0 个高层动作决策。

26 个成功回合中，执行动作与仅作诊断的反事实几何教师一致率为 **87.7%**；4 个失败回合仅 **40.2%**，均在 align 阶段未完成最终收敛，
而非 descend/close/lift 状态机故障。30 回合合计中，PM 唯一胜者直接产生 1,205 个 align 决策；仅在模糊 PM 时，本地 progress-memory
与 progress-value 分别介入 582 / 97 次。因而该结果是“binary-latent BDM-SNN 负责候选横向决策 + 物理进度记忆负责不确定读出恢复”的
可靠**系统级**结果，仍不能表述成冻结 SNN 权重本身已达到 86.7%。

30 回合目录为 `results/binary_latent_progressguard_extended30_seed2/`。结合 13.71 的 3x3 图，当前最佳结论是：base 工作点上的
闭环成功问题已经显著改善，但大 y 偏移空间泛化仍不足；下一阶段应在不直接动作接管的条件下，把 progress token 编入 DLPFC 并以短、可恢复
序列蒸馏给 PM，随后报告纯 PM 与 safeguard 系统两组结果。

### 13.73 DLPFC progress token 的严格配对负结果：原型注入尚未提高纯 PM 成功率（2026-08-02）

为把 13.72 中只在 PM 模糊时动作接管的本地 safeguard，改成**不接管动作**的 SNN 上下文，新增了四个 DLPFC progress-token 神经元。
align 阶段维持原来的 200 个 `dominant-axis x 10x10 residual` 感知神经元；当上一个横向动作实际降低 xy 误差时，再共同激活对应
`+x/-x/+y/-y` token。该 token 与 residual 输入一样先在源核进入 rank-3 NMF 投影、经过三个 IF / soft-reset sigma-delta latent 神经元、
以 binary AER 事件跨核，再由解码 MVM 驱动 StrD1/StrD2；它不读取教师动作、不改变 FSM，也不覆盖 PM 输出。

为了不重训并扰动稳定的 200-state 策略，四条新增 NMF-U 行初始化为离线教师数据中相应动作的旧 U 行均值（样本数
`[+x,-x,+y,-y]=[570,173,22,178]`），保持旧 200 行、V、BG/PM 权重不变。迁移检查确认当 token 未激活时，旧/新网络逐内部 SNN 步的输出严格
相同；token 激活时才改变 latent 事件和 Str 电流。这是一个可部署的 context 通路，但尚不是端到端学习到的通信流形。

在同一冻结 checkpoint、同一 30 个 environment / PM RNG seed（offset 80）、rank-3 binary AER、30 内部步、无教师、无在线写入且禁用
progress safeguard 的严格 paired 对照中，token-on 为 **12/30 (40.0%)** Lift 成功，屏蔽 token 输入但保留完全相同 204-state 权重的
token-off 为 **14/30 (46.7%)**。两组均有 **15/30** 回合进入 descend / close / lift，故负差异发生在 SNN 主导的横向 align 阶段，而非后续
状态机。token-on 共在 4,707 个 align 决策中的 1,891 个决策激活，排除了“token 没有真正进入跨核脉冲通路”的解释；但仅 16/30 回合的成败相同，
说明它改变了闭环轨迹，却没有形成稳定正收益。结果分别位于
`results/binary_latent_progress_token_prototype_extended30_seed2/` 与
`results/binary_latent_progress_token_prototype_suppressed_extended30_seed2/`。

另以相同前 10 个 paired seed 将 token DLPFC 输入从 0.5 降为 0.25，结果仅 **2/10**，而 0.5 token-on / token-off 分别为 **5/10 / 4/10**。
因此不能把失败归为“token 仅仅太强”，也不应继续做无监督的幅度扫描。当前保留默认关闭、且可由
`--align-residual-progress-token-weight` 控制的实现，作为通信上下文原型；主 pure-PM baseline 仍为 200-state rank-3 binary-latent 模型。
下一步若要继续学习该信息，应冻结旧 U/V 与 BG/PM，只用**动作类别平衡、可恢复的短窗口**更新四条 token 行，并先验收 token-on 相对 token-off
的配对 teacher-action agreement，再进行 30 回合闭环评估；否则应停止该 token 路线，转而修复已定位的 align 行为覆盖问题。

### 13.74 仅训练四条 token-U 行：工程隔离成立，但短 recovery 监督仍未改善闭环（2026-08-02）

为避免 13.73 的端到端 BPTT 同时改动旧 sensory U、decoder V 与下游有效权重，新增
`--nmf-latent-pm-offline-progress-token-only`。该模式以 checkpoint 的 prototype token 为起点，反向传播仍穿过
`DLPFC IF -> rank-3 binary sigma-delta latent -> V -> Str/BG/PM`，但梯度写入被严格限制到新增的四行 `U_token`；原有 200 行 U、两条
V、BG/PM 均冻结。contrast 编码的 token 行还被限制不低于既有 U 每列 floor，以免 token 改写全局 `min(U)` 后间接改变无 token 输入。
同时新增 `--nmf-latent-pm-offline-recovery-windows-only`：完整教师轨迹保留为锚点，而 DAgger 仅取首个 PM/教师不一致附近的短窗口，
不把 240-decision 的不可恢复尾部放入 BPTT。

实施时发现一个重要工程陷阱：训练循环末尾的低秩重投影也会重写完整 U/V。token-only 模式已显式关闭该重投影和通信 refit；用
`2 teacher + 6 DAgger` 完整采集加 BPTT 检查，旧 200 行 U 与两侧 V 的最大绝对差均为 **0**，只有 token 行变化。因此随后评估的结果确实
可归因于 token-U 局部更新，而不是主通信因子漂移。

在固定 base、rank-3 binary AER、30 内部步、无 safeguard 的 `2 teacher + 6 DAgger + 6 frozen` 训练中，训练内评估为 **1/6**；从该 checkpoint
按 13.73 的同一 10 个独立 episode/PM RNG seed 冻结回放为 **2/10** Lift 成功、**4/10** 进入 descend/grasp、平均 217.1 决策。作为同一
评估协议的参考，未局部训练的 prototype token 是 `5/10`，屏蔽 token 为 `4/10`。所以尽管这次局部训练的 held-out BPTT loss 降到约 0.6，
闭环反事实教师一致率却只有 23.9%，低于 prototype 的 33.9% 与 token-off 的 47.6%。这再次表明小型离线 spike-count 交叉熵不能替代
闭环控制验收。

此实验的有效结果目录是 `results/binary_latent_progress_token_localu_final_train_seed2/` 与
`results/binary_latent_progress_token_localu_final_checkpoint10_seed2/`。此前未关闭重投影的
`results/binary_latent_progress_token_localu_recovery*_seed2/` 会改写旧 U/V，故明确标为无效排障运行，不作为性能结果。
结论是：当前 measured-progress token 在“正确动作已使误差下降”后才出现，监督数据缺少它应如何把错误/模糊状态拉回的反例；仅调这四条
静态 U 行无法产生 recovery policy。后续不应继续扫描学习率或窗口长度，而应回到主 baseline 的 align 覆盖问题，或者将 token 改为具有
明确 prediction-error/状态转换语义的动态残差事件后再重新设计训练目标。

### 13.75 动态反向 prediction-error token：rank-3 decoder 的四向可分性不足（2026-08-03）

作为不同于 13.73 的动态上下文 pilot，新增 `--align-residual-error-token`：上一横向动作使实际 xy 误差恶化时，下一次决策在原 residual cell
外共同激活**相反方向** token；该事件只存在一个高层决策，仍仅经 DLPFC、rank-3 binary latent、Str/BG/PM 竞争，既不用教师也不覆盖 PM 动作。
初版把“不改善”都视为 error，token 出现在 `1435/2014` 个 align 决策，结果 `2/10`；对照 token-off 为 `6/10`。将规则严格为误差实际增加
超过 1 mm 后，token 降为 `757/1905` 个决策，仍只有 **3/10**，而同架构、同十个 environment/PM seed 的 token-off 仍为 **6/10**。

该负结果不是 token 未传播：静态读取 stable checkpoint 的 NMF 因子，四个 action-prototype token 经 contrast 编码及固定 V 后，StrD1 的最大
电流全部落在动作 2，四个 decoded current 的 cosine 为 `0.925--1.000`。全体 200 个原 residual state 也有 `192/200` 的 StrD1 最大电流
落在动作 2。进一步在固定 V、非负 `z∈[0,1]^3` 下优化各目标动作的 margin，动作 0/1/2 可得到正 margin，而动作 3 的最大 margin 为 0。
这说明当前 rank-3 非负 DLPFC→Str decoder 本身不能稳定承载四个独立的纠错方向；失败主要是表示容量/可分性边界，而不是阈值选择。

结果目录为 `results/binary_latent_residual_error_token_checkpoint10_seed2/`、
`results/binary_latent_residual_error_token_w1mm_checkpoint10_seed2/` 及相应 `suppressed` 严格对照。下一步仅在 rank-4 的原权重重投影可恢复四向
token 可分性时，才测试其闭环；否则应停止将 four-way action code 塞入 rank-3 compressed link，转而传递低维连续误差/事件幅度而非动作身份。

### 13.76 容量边界审计与 rank-4 冻结迁移：单纯扩维不能恢复纯 PM（2026-08-03）

对稳定的 `recovery_control_warmup3_seed2` checkpoint 做了两步只读诊断。其离线 teacher/DAgger 数据有 943 个 align
样本、26 个 residual 状态；每个已访问状态只有一个教师方向，因此失败**不是**同一可观测状态被要求输出不同动作。可是从干净膜电位
重放硬件一致的 binary latent--BG--PM 路径时，943 个样本的 PM 计数均相同，最终都由并列规则读为 `+x`，statewise top-1 仅为
教师 `+x` 的先验比例 `570/943=60.4%`。即：静态输入可区分，但当前 PM 训练未保证每个方向能在硬脉冲读出中即时形成胜者。

从保存的有效 `200x8` DLPFC--Str 权重重新分解还发现 D1、D2 的数值秩都已为 2。把它们重新做 rank-4/5/6/8 NMF 并不会凭空增加
方向信息；四个 action prototype 经部署一致的归一化/contrast 编码后仍主要译为 action 2，连续且更宽松的 `z in [0,1]^k` 也仍有
至少一个方向的最大 current margin 不为正。因此 binary spike 只会更严格。

新增显式 `--checkpoint-migrate-striatum-nmf-rank`，只保留 checkpoint 的局部/BG/PM 参数，从原有效 DLPFC--Str 矩阵重建指定 rank 的
NMF `U,V`，并记录迁移误差，拒绝隐式 shape 不匹配加载。在固定 cube、30 内部步、无教师、无在线写入、禁用 safeguard、相同 10 个
episode/PM RNG seed（offset 80）的 pure-PM 回放中，rank-4 冻结迁移为 **0/10**，rank-3 对照为 `4/10--6/10`；所有 rank-4 回合均未
离开 align，四个横向动作近似随机游走，教师一致率仅 `18.3%--32.9%`。rank-4 冻结迁移是明确负结果，不能作为新基线。

下一步保持 binary AER 与 PM 唯一部署读出，在训练期可选地增加一个辅助约束：以真实到达 StrD1 的**硬脉冲累计电流**对教师方向做 CE，
迫使 `U,V` 保留即时四向可分性；PM 的原 BPTT loss 仍是主损失，部署时不增加教师、浮点 payload 或动作旁路。只有该约束先改善状态级
脉冲可分性，才进行 paired robot 闭环评估。

### 13.77 rank-4 硬脉冲 StrD1 辅助约束：局部可分性恢复，但闭环仍为 0/6（2026-08-03）

按 13.76 的判据，试验了一个只在离线训练期存在的辅助项。原有 PM spike-count BPTT 仍为主损失；额外对每个决策窗口内**实际由 hard
binary AER pulses 驱动的 StrD1 累计电流**做教师动作 CE。它不读取浮点 latent、没有部署旁路，也不改变 D1--GPe/GPi--thalamus--PM 的
前向结构。rank-4 因子从原 rank-3 checkpoint 的有效权重显式重建，再用 `2 teacher + 6 DAgger + 6 frozen` 训练；辅助系数为 `0.05`。

局部指标确有改善：最佳 held-out 离线损失从 `1.84` 降至 `1.50`，而保存后对 1,530 条离线状态以 clean membrane 重放，硬脉冲 PM
statewise top-1 为 **99.3%**、macro-F1 为 **0.771**（迁移前 rank-3 checkpoint 的对应值为 `60.4%/0.188`）。但这不能等同于机器人
控制：六个 DAgger 回合和六个冻结评估回合都未离开 align，最终 pure-PM 为 **0/6**。冻结评估的网络/反事实教师一致率只有 **33.1%**，且
约 **75%** 的自主 residual state / 连续位置落在初始教师演示支持之外。辅助项约束的是局部 StrD1 电流，不能保证 StrD2、GPe/GPi、
thalamus 与 PM 的时序竞争仍给出正确胜者；同时长失败 DAgger 尾部仍扩大了 OOD 分布。

因此 rank-4 + 辅助项是一个**局部编码正、闭环控制负**的结果，不能替代 rank-3 baseline。下一步应回到已验证的 200-state rank-3
binary-latent checkpoint，先保留完整教师锚点、只采集首错附近且教师仍可恢复的短 DAgger 窗口，并先检查四个方向均有实质样本支持；每次更新
都以同 episode/PM RNG 的 10/30 回合 pure-PM paired replay 为最终验收。结果位于
`results/rank4_latent_action_aux_train_seed2/`，statewise 审计为该目录的 `pm_offline_dataset_audit.json`。

### 13.78 rank-3 可恢复窗口再训练：训练内 3/6 不可替代严格 checkpoint 回放（2026-08-03）

为直接检验 13.77 提出的“去掉不可恢复 DAgger 失败尾部”是否有效，恢复稳定的 compact-Str checkpoint
`recovery_control_warmup3_seed2/bdmsnn_final_state.pt`，保持原有 **200-state、8 action、rank-3 NMF、source IF +
soft-reset sigma-delta binary AER、normalized + contrast encoding、30 个内部 SNN 步、pure PM、无 token、无
progress safeguard**。唯一改变是离线 PM BPTT 的样本选择：两条完整教师成功轨迹全部保留；六条 DAgger 轨迹只保留首次
PM/反事实教师动作不一致附近、且教师仍可恢复的连续窗口（长度 3、warm-up 3），不再让每条长达 240 个决策的失败尾部进入损失。

训练收集期间 teacher 为 `2/2`，DAgger 为 `2/6`；筛选后虽从 1,165 个收集决策中构成了 70 个序列窗口，但实际训练/验证仅为
49/21 个样本，方向类别权重为 `[0.041, 1.000, 1.959, 1.000, ...]`，说明窗口仍很小且类别很不均衡。最佳 validation loss 为
`1.075`（第 2 epoch）；随后六条冻结 evaluation 为 **3/6**。这个数只适合说明该方案没有立即造成全动作塌缩，不能作为优于旧模型的
证据。

因此再以训练后的 checkpoint 做严格部署验收：固定 cube、相同的 10 个 environment reset / 每回合 PM RNG（seed-2,
`episode_seed_offset=80`）、无教师、无 DAgger、无任何在线写入、禁用 safeguard。该 checkpoint 得到 **2/10** Lift 成功；在**完全
相同**十个回合条件下，未再训练的原 rank-3 checkpoint 为 **6/10**。故可恢复窗口选择本身是合理的防污染措施，但当前用两条教师和六条
DAgger 得到的 70 个窗口覆盖不足，端到端 BPTT 更新反而损害了既有策略；本轮是明确负结果，不扩展至 30 回合，也不替换 `12/20`
的稳定 pure-PM 基线。

最后澄清本实验中“有效权重数值秩约为 2”的对象：当前 compact-Str 的两条 DLPFC--Str 有效矩阵分别都是 **`200 x 8`**，而不是
`200 x 800`；其奇异值前两项显著（D1 约 `59.219, 2.364`，D2 约 `48.584, 1.694`），其余低于默认数值容差，因此数值秩均为 2。
虽然部署因子仍配置为 rank-3（`U:200x3, V:3x8`），第三个通道并不代表原矩阵具有第三个稳定独立方向。`200 x 800` 仅是早期
expanded-Str（每个 state 复制 8 个纹状体动作神经元）的潜在形状，并非当前 checkpoint 的实现。

本轮训练目录为 `results/rank3_recoverable_windows_train_seed2/`，严格 paired replay 分别为
`results/rank3_recoverable_windows_checkpoint10_seed2/` 与 `results/rank3_original_checkpoint10_seed2/`。下一步不应继续小数据
微调 U/V；应先收集跨 x/y 起点、完整且成功的教师轨迹，按四个横向动作和 residual 区域分层形成足量训练/验证集合，再以同一 paired
10/30 回合 pure-PM 回放验收端到端更新是否真正提升闭环。

### 13.79 四方向完整教师与更新剂量检验：静态离线 BPTT 梯度未对齐闭环目标（2026-08-03）

13.78 的窗口数据仍小且含 DAgger 样本，因此进一步做了一个更干净的对照：从同一稳定 rank-3 checkpoint 开始，**不使用
DAgger**，只让几何教师在四个相对 cube 起点 `(+x,-x,+y,-y)` 各运行两次，得到 8 条完整轨迹；其中 7/8 Lift 成功，采集到
444 个标注决策、136 个长度为 `warm-up 3 + loss 3` 的序列窗口。四个横向动作在训练集均有非零支持，class weight 为
`[0.130,0.724,1.291,1.855,...]`。其余条件仍是当前 hard binary AER rank-3 路径，且部署时仍无教师、无在线写入、无 safeguard。

离线 BPTT 的最佳 validation loss 为 `1.287`，训练内冻结 base evaluation 为 **4/6**；但这仍不是最终验收。固定相同 10 个
environment reset 及逐回合 PM RNG（seed-2、offset 80）的 strict pure-PM paired replay 中，完整更新 checkpoint 只有 **2/10**，
而未更新的原 checkpoint 为 **6/10**。这说明补齐四个动作标签本身不足以提高 base 闭环，不能把训练内 `4/6` 解读为算法改善。

为区分“BPTT 步长过大”与“梯度方向不对齐”，只读地把训练后两条 NMF 因子 `(U,V)` 与原 checkpoint 插值；除
DLPFC--Str 因子及同步的局部 shadow `W=UV` 外，所有 BG/PM 参数都严格保持原值。全量更新相对原有效 `W` 的 Frobenius 改变量仅为
D1 `0.33%`、D2 `0.78%`，但阈值/复位系统仍可能对小扰动发生离散动作改变。更保守的 25% 插值仍为 **2/10**，5% 插值为 **1/10**，
均低于原 checkpoint 的 **6/10**。这虽不是跨 seed 的总体统计，但在完全配对的十条回放内已经排除了“仅仅因为更新太大而退化”的解释。

结论是停止继续扫描此类直接 `U,V` 离线微调：问题不是教师动作类别缺失或单次写入尺度，而是当前静态 PM spike-count /
交叉熵损失无法充分约束 binary sigma-delta residual、Str--BG--PM 膜状态及闭环恢复轨迹。稳定 rank-3 binary-latent checkpoint
继续作为 pure-PM 主基线（既有扩展评估 `12/20`）；后续创新应先设计具有明确状态转移或预测残差语义、且能在脉冲时序层保持方向可分的
通信表示/训练目标，而不是再以离线分类损失小幅调节同一组因子。

本轮目录为 `results/rank3_teacher_direction8_train_seed2/`、
`results/rank3_teacher_direction8_checkpoint10_seed2/`、
`results/rank3_teacher_direction8_blend25_checkpoint10_seed2/`、
`results/rank3_teacher_direction8_blend05_checkpoint10_seed2/`；原 checkpoint 的严格 paired control 位于
`results/rank3_original_direction8_control10_seed2/`。

### 13.80 Binary-latent 主基线的动作维度复核与 3 checkpoint x 50 回合冻结评估（2026-08-03）

首先更正一个容易混淆的维度表述。当前**最佳 binary-latent checkpoint 并非 4-action 模型**，而是 compact-Str 的
`S=200, A=8` 模型：DLPFC--StrD1 及 DLPFC--StrD2 各有一张 `200x8` 有效矩阵，分别再分解为
`U:200x3, V:3x8`。代码中确实保留了 `--align-action-count 4` 的更小版本；该版本的 Str/GP/PM 都只有四个横向
通道，`+z/-z/open/close` 完全不进入 SNN，而由 FSM 固定执行。它先前用于未压缩 compact-Str 消融，但目前尚无与本节
checkpoint 同训练协议的 binary-latent 4-action checkpoint，因此不能把两者混称为同一 baseline。

当前 `A=8` checkpoint 的运行语义是：align option 中 `allowed_actions=(+x,-x,+y,-y)`，故 SNN **实际读出/执行的仍只有
四个横向动作**；但 Str、GPe、GPi、thalamus、PM 保留八个神经元通道，后四通道在 descend/close/lift/recover option
中作为 FSM 可行的固定原语使用。这就是此前“网络有 8 个通道、align 决策只有 4 个动作”两个说法都成立的原因。`2x4=8`
也不是该网络动作数的来源：`2` 是 dominant x/y error axis 的一位状态编码，状态数为 `10x10x2=200`；它与动作数独立。

为降低此前 20 回合汇总的有限样本不确定性，对三个已冻结 checkpoint 分别运行 50 个**全新** episode，合计 150 回合。所有部署条件固定：
rank-3 NMF、source IF + normalized/contrast soft-reset sigma-delta、binary AER、30 internal SNN steps/decision、fixed
cube、`epsilon=0`、无 teacher、无 BPTT、无 R-STDP/TD 写入、无 progress safeguard；每回合按 reset seed 重新初始化环境和
PM tie-break RNG（offset 200）。因此这是 strict pure-PM 结果，而不是系统级 safeguard 结果。

| checkpoint seed | 新 50 回合 Lift 成功 | grasp-any | 平均动作决策数 |
|---|---:|---:|---:|
| 0 | 13/50 = 26.0% | 14/50 | 213.5 |
| 1 | 10/50 = 20.0% | 14/50 | 214.8 |
| 2 | 20/50 = 40.0% | 26/50 | 198.4 |
| 合计 | **43/150 = 28.7%** | **54/150 = 36.0%** | **208.9** |

该新汇总低于旧的 3x20 回合 `27/60=45.0%`，说明原小样本对性能较乐观；当前更稳妥的表述应为：**在固定 base 起点，冻结的
rank-3 binary-latent pure-PM 可完成约三成 Lift，但 checkpoint/随机回合方差显著，尚不是稳定高成功率控制器。**

阶段诊断进一步定位瓶颈：150 回合中 57 回合进入 descend，且这 57 回合全部继续到 close/lift；54 回合发生 grasp，最终 43 回合 Lift
成功。故 93/150 回合主要卡在 SNN 主导的横向 align，后段 FSM 只造成较小的额外损失。成功回合的 align 网络动作与仅作诊断的反事实
几何教师一致率为 `81.6% +/- 10.0%`，失败回合仅为 `26.6% +/- 9.6%`。PM/readout 没有 silent decision，但 align 中 PM 唯一胜者
动作 15,168 次的正 xy-progress 比例为 48.2%，而 12,423 次 ambiguous tie fallback 仅为 24.2%，累计 xy progress 还是负的
`-11.62 m`；这解释了为什么禁用 progress safeguard 后许多回合在 align 反复游走。

通信统计同样仅描述本次软件事件，不直接等同物理链路节省：150 回合累计内部样本 827,730，D1/D2 binary latent logical events
分别为 978,272/1,050,418，合计 2,028,690（约 64.8 events/高层决策、2.45 events/internal sample）；目标电流相对连续
latent shadow 的 online EV 约为 D1 `0.554--0.566`、D2 `0.563--0.568`。这表明三个 latent neuron 确实以二值脉冲传输，但当前
重建误差和时序竞争仍足以造成控制方差；没有 packet/bit/FIFO 的实测链路模型时，不能仅据此声称通信成本已下降。

本节评估目录为 `results/binary_latent_extended50_pairedrng_seed{0,1,2}_offset200/`。后续应将 4-action compact-Str binary
latent 作为与本节严格匹配的独立架构重新训练和评估，而不是把它与现有 8-action checkpoint 直接比较；本轮按约定不继续做优化。
