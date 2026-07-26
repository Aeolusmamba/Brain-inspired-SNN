# BDM-SNN RoboSuite Lift Baseline

This independent first-stage prototype connects BrainCog's `BDMSNN` to
robosuite 1.5's single-Panda `Lift` task.  It does not modify the Flappy Bird
entry points and does not enable RRR communication compression.

## Stage-1 scope

- State is simulator low-dimensional truth, not RGB: cube relative to the end
  effector is encoded as 8 spatial octants x 4 distance bins x 2
  closed-or-grasped modes (`S=64`).
- The action population has `A=8`: `+x`, `-x`, `+y`, `-y`, `+z`, `-z`, open,
  and close.  A discrete decision is held for a fixed number of Panda OSC pose
  control steps.
- The BDM-SNN topology remains DLPFC, StrD1, StrD2, STN, GPe, GPi, thalamus,
  and PM.  Each episode resets neural / STDP / eligibility state but retains
  long-term DLPFC-to-striatum weights.  Reward-modulated STDP supports the
  general state-action slice `s*A:(s+1)*A`.
- The output records shaping return and sparse Lift success separately, plus
  action distribution, region spike totals, episode length, PM silence, and
  D1/D2 weight-change magnitude.  A nonzero shaping return is not reported as
  a successful lift.

## Environment

The verified `lph` environment versions are Python 3.10.4, PyTorch
2.1.2+cu121, CUDA, robosuite 1.5.2, MuJoCo 3.3.7, and NumPy 1.26.4.  MuJoCo
3.10 is not compatible with robosuite 1.5.2's OSC controller API.

```bash
cd /home/lph/Brain_SNN
MUJOCO_GL=egl /home/lph/.conda/envs/lph/bin/python \
  Brain-Cog/examples/decision_making/BDM-SNN-Robosuite/lift_bdm_snn.py \
  --validate-env --device cuda:0

MUJOCO_GL=egl /home/lph/.conda/envs/lph/bin/python \
  Brain-Cog/examples/decision_making/BDM-SNN-Robosuite/lift_bdm_snn.py \
  --episodes 10 --seed 0 --device cuda:0 \
  --output-dir results/lift_bdm_snn_baseline_seed0
```

The MuJoCo simulator remains CPU-side; `--device cuda:0` runs the BDM-SNN and
its online plasticity tensors on the GPU.  This software baseline is not yet a
256-neuron-per-core hardware mapping: the `64 x 8` striatal state-action
population contains 512 logical neurons and would need a multi-array mapping.

## Next stage

Only after the full-communication success baseline is stable should a separate
visual encoder and RRR link be enabled.  That comparison must record latent
dimension, source logical spikes, AER packets/bits, peak rate, FIFO occupancy,
target array activations, task success, and control latency independently.
