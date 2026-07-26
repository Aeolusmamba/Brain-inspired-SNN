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
