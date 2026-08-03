# 部署期脉冲减事件方案

> 历史基线、RRR/NMF 与 binary-latent 的已完成实验见
> [Baseline_and_RRR.md](Baseline_and_RRR.md)。本文只规划从高性能**全通信**基线出发的真实跨核脉冲减事件，
> 不把它与 RRR 低维 payload 混为同一问题。

## 目标与边界

本阶段从 Robosuite Lift 的高性能全通信 BDM-SNN 开始，不讨论低秩 latent 压缩，也不以“矩阵维度减少”替代实际链路降耗。目标是在**推理/部署期**减少跨核二值 AER 脉冲，同时保持任务性能。所有结论必须分别报告：

\[
N_{logic}=\sum_{l,t,i}s_{l,i}(t),\qquad N_{packet},\qquad B_{link},
\]

以及峰值事件率、目标端阵列激活次数、FIFO 占用、控制延迟和 Lift 成功率。批量封包只能降低 packet/bit，不能被写成降低 logical spike 数。

## 锁定基线

采用历史全通信紧凑 Str 基线：`S=200`（`10x10` xy residual 格乘 x/y dominant-axis 1 bit）、`A=8`、仅 align 由 SNN 决策、其余 option 由 FSM 下发固定安全原语。三 seed 使用 20 个教师回合后继续无限制在线 TD/STDP 的 50 个自主回合，Lift 为 `45/50, 46/50, 45/50`，合计 **136/150 = 90.7%**。

这不是冻结 pure-PM binary-latent 基线；它是本阶段的高性能软件全通信参考。首次减事件实验必须保持相同任务、seed、教师课程、在线更新、240 高层决策上限和 30 内部 SNN 步，不能混用后续冻结 binary-latent 结果。

三 seed 全通信监视器共记录 1,291,343 个跨核 source logical events。按边排序：

| 跨核边 | source events | 占比 | events / 内部步 | 优先级 |
|---|---:|---:|---:|---|
| StrD2 -> GPe | 508,775 | 39.4% | 7.20 | 第一优先 |
| GPi -> thalamus | 226,627 | 17.6% | 3.21 | 第二优先 |
| StrD1 -> GPi | 122,438 | 9.5% | 1.73 | 第三优先 |
| GPe -> STN | 108,399 | 8.4% | 1.53 | 后续 |
| STN -> GPe / GPi | 各 91,838 | 各 7.1% | 1.30 | 后续 |
| DLPFC -> STN / thalamus | 各 70,714 | 各 5.5% | 1.00 | 后续 |

因此，真正的大幅削减必须作用于 StrD2、GPi、StrD1 的**源端真实发放**；仅处理 DLPFC 广播或仅做封包不能显著改变总 logical event 数。

每个高层决策最多允许 30 个 SNN 内部 slot，但这个高性能基线会在 PM 首次可靠发放后提前结束，实际平均约为
2.2 个 slot/决策。因此所有节省必须同时按 slot、决策和完整 episode 计量，不能把固定的 30-slot 上限误当成实际执行长度。

## 候选方法

### A. 决策内保持的 DLPFC 广播（零风险接口验证，先做）

在一个 align 决策内，DLPFC 的 one-hot 状态不变，且 DLPFC -> STN 与 DLPFC -> thalamus 的现有外送权重为全同权。
因此可在第一个内部 slot 发送一次共享的 `DLPFC-active` AER 事件；两个目标核用本地保持寄存器或慢 PSC 在余下 slot
重放相同的恒定输入电流，并在下一个决策边界清除状态。

- **它减少什么：** 这两条边合计约占当前 `N_logic` 的 11.0%。若两个目的端允许同一 source event 携带 destination mask，
  还可同时减少 packet/bit；但必须分别报告 multicast 前后的 `N_packet`。
- **为何是第一步：** 可逐 slot 比较两个目标电流，并要求与全通信前向严格一致。这先验证 AER 计量、保持电流与目标端阵列触发的
  实现正确，而不引入任务性能风险。
- **边界：** 这是调度型、语义保持的事件削减，不是学习得到的稀疏神经编码；其上限小，不能替代后续主路线。

### B. 可训练的跨核发放成本 + burst 成本（主路线）

对跨核源群体加入部署一致的训练期正则：

\[
\mathcal L=\mathcal L_{control}+\lambda_E\sum_{l\in\mathcal C,t,i}s_{l,i}(t)+
\lambda_B\sum_{l\in\mathcal C,t>0,i}s_{l,i}(t)s_{l,i}(t-1).
\]

其中 \(\mathcal C\) 第一阶段仅取 StrD2；第二阶段才扩展到 GPi、StrD1。用 surrogate gradient 或 straight-through 估计训练期硬阈值 spike 的梯度；部署前向仍是原生二值脉冲和原 BDM-SNN 动力学，不发送浮点值。

- **硬件映射：** 不增加部署模块；RRAM MVM 后原模拟神经元积分、阈值、复位和 AER 地址发送保持不变。
- **真实节省：** 若训练后源端 spike 减少，则 `N_logic`、AER address record、目标端输入事件和潜在阵列激活都下降。
- **风险：** 直接给高流量 StrD2 加大惩罚可能破坏 indirect pathway 的 tonic inhibition，导致 GPe/GPi/PM 动力学改变。故必须从很小的 \(\lambda_E\) 和单边开始，并以任务成功和每条边事件数共同验收。

### C. relay-IF 的自适应阈值 / refractory burst 抑制（主路线的部署执行器）

不直接改动递归 StrD2 神经元本身，而在 StrD2 源核的出口加入 8 个二值 relay-IF 神经元：StrD2 spike 先在 relay
本地积分，只有 relay 发放才跨核发送 AER 并驱动 GPe 输入。这使产生、发送、接收三类事件可独立审计，且不必改变
StrD2 -> GPe 的原始递归细胞动力学。relay 可采用发放后瞬时提高阈值并指数衰减，或一到数个内部 slot 的 refractory：

\[
\theta_{t+1}=\theta_0+\rho(\theta_t-\theta_0)+\beta s_t.
\]

这不会把事件合并为 count，而是阻止冗余连续脉冲实际产生。第一版将其作为固定、可开关部署参数；若基线可承受，再在 A 的训练期将其纳入前向并联合训练阈值基线/\(\beta\)。

- **硬件映射：** relay 是源核本地的模拟 IF/阈值电容或带泄漏的数字状态、refractory flip-flop；不需要 ADC、FPGA 参与神经计算或额外跨核数据。
- **真实节省：** 直接降低跨核发送的 high-rate burst（尤其 StrD2）；比 packet aggregation 更符合目标。报告中必须区分
  `N_logic,produced`（StrD2 产生）、`N_logic,sent`（relay 实际 AER）、`N_logic,received` 与目标 MVM 激活数。
- **风险：** 固定 refractory 会删掉本来对递归 BG 回路有意义的短时序。必须测 decision 内 first-spike 时间、burst ratio、PM tie/silent rate，而非只看均值发放率。

### D. 预测残差事件（第二阶段）

对某条跨核边，源/目标维护相同预测电流或低阶泄漏状态；只在创新量超过阈值时发送事件：

\[
e_t=x_t-\hat x_t,\qquad s_t=\mathbb{1}(|e_t|>\theta_e).
\]

目标端用相同预测器恢复基线，收到事件后校正。该法有机会显著消除稳定 tonic firing 的重复事件，但必须以**实际脉冲发生时刻**验证，不能用一个 decision-level count 冒充 lossless。

- **硬件映射：** 源端、目标端各有匹配的轻量 leak/integrator；threshold crossing 触发普通 AER。预测器状态是链路端状态，不是教师或外部控制器。
- **真实节省：** 稳态/缓变时可同时降低 `N_logic`、records、bits 和目标端激活。
- **风险：** 两端 leak、阈值、RRAM 电导漂移或丢包失配会累积误差；递归 BG 边可能比前馈边更敏感。故只在 A/B 已给出可靠基线后，先用于单条非动作身份边并加入周期性同步/误差上界监测。

### E. 语义保持的 protocol 聚合（独立硬件对照）

对现有 spike 不作算法抑制，仅将同一短时间桶的地址压缩为 bitmap、计数或 burst record。它可降低 `N_packet` / `B_link`，但不降低 `N_logic`；若 target leak、阈值、复位依赖桶内时序，则不是无损替换。它应作为通信协议对照，而不是主创新结论。

## 推荐验证路线

1. **完整只读审计。** 用高性能全通信 checkpoint 在相同 3 seed 协议下记录每边 `N_logic,produced`、每内部步 firing rate、连续 slot burst ratio、决策内峰值事件率、PM tie/silent、每阶段成功率；补充 packet/bit/FIFO 的明确简化模型。
2. **严格等价的 DLPFC 保持接口。** 逐 slot 验证 DLPFC -> STN/thalamus 的目标电流与全通信完全相同，再跑 paired 3x50，作为约 11% logical-event 削减的低风险对照。
3. **单边 relay 筛查。** 只为 StrD2 -> GPe 加 relay，先在部署期作固定 1-slot refractory（再测 2-slot）或小幅阈值自适应。每 seed 10 回合 paired smoke；目标是该边 sent AER 至少降低 20%，成功率相对 baseline 不低 5 个百分点，PM silent/tie 不恶化。通过后再跑 3x50。
4. **训练期 event/burst 正则。** 将最好的 relay 动力学置于教师和自主训练的前向，从小 \(\lambda_E\) 起先只优化 relay 阈值/恢复参数，必要时小范围解冻 DLPFC -> StrD2。比较 `event-only` 和 `event+burst`；surrogate gradient/STE 只用于训练的反向估计，部署前向仍为二值 relay spike。接受条件：3x50 成功率相对 90.7% 下降不超过 5 个百分点、StrD2 sent AER 显著下降、总 `N_logic,sent` 至少下降 15%。
5. **逐边扩展与预测残差。** 仅在 StrD2 成功后加入 GPi -> thalamus，再加入 StrD1 -> GPi；每次新增一条边。最后才在更安全的 DLPFC 广播边验证 event-trigger 预测残差，显式报告 prediction error、同步次数、events、packets、bits、FIFO、延迟和任务性能。

## 成功判据与失败判据

- 主要成功：相同 3x50 协议下 Lift 成功率不低于全通信 90.7% 的 5 个百分点以内，同时总 `N_logic` 至少下降 15%，并给出每边贡献。
- 强成功：在成功判据上，总 logical events 下降至少 30%，峰值事件率和目标端激活不升高。
- 失败：事件下降仅由 episode 更早失败、减少内部步、或改用 count/bitmap 而 source spike 未降；这些只能作为失败或协议对照，不能称为算法稀疏化成功。
- 公平性：每组固定相同 seed、课程、online learning、决策窗口、初态与控制预算；单独报告教师期和无教师期。不能把冻结 binary-latent 或 progress-safeguard 数字拿来同此全通信在线基线比较。

## 当前决定

首个实现分两步：先完成 DLPFC 广播的“一次 AER + 决策内保持电流”严格等价对照，再实现**仅对 StrD2 -> GPe 的可开关 relay-IF adaptive threshold / short refractory**及真实事件、burst、时序和任务联合审计。前者低风险地验证接口和计量；后者直接瞄准 39.4% 的逻辑事件且不改变 AER 的二值语义。其作用机制清楚后，再加入训练期 event/burst 正则寻找可部署的更低发放解。

## 文献定位

- Semedo et al. (2019) 的 RRR communication subspace 是基于分箱 spike-count 的目标预测分析；它不是 AER 节省证据。
- Boerlin et al. (2013) 从表示误差和发放代价推导预测性稀疏脉冲；可作为 event/burst 成本的理论参照。
- Hu, Genkin and Chklovskii (2012) 讨论 WTA 稀疏脉冲表示；若 relay 路线不足，可作为后续 pre-spike membrane-potential WTA 的依据，不能在同值二进制 spike 后简单 top-k。
- Mostafa (2018) 的 TTFS 更适用于决策窗口内状态保持的前馈 DLPFC 广播，不宜一开始施用于递归 BG 环。
