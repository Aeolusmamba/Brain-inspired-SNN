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
