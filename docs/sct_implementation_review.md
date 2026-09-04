# PocketVE SCT 实现审查报告

## 1. 审查范围

本报告以以下仓库中的 single-time consistency training（SCT）实现为参考：

- 参考仓库：`/shared/healthinfolab/phz24002/VEDA-shortcut-sct`
- 当前仓库：`/shared/healthinfolab/phz24002/pocketVE-CM/PocketVE`

重点检查了：

- SCT 的 joint JVP 定义及 linear/log-sigma 时间参数化；
- coordinate SCT residual；
- categorical logit-SCT；
- tangent normalization；
- graph-wise loss reduction 和 loss 权重；
- student/teacher 状态复用及 stop-gradient；
- 动态邻接图对 JVP 的处理；
- SCT 在训练总 loss 和日志中的接入；
- 单元测试、模型级 GPU 检查和缺失功能。

主要相关文件：

- 参考数学及行为说明：`VEDA-shortcut-sct/docs/consistency_training.md`
- 参考实现：`VEDA-shortcut-sct/models/fm.py`
- PocketVE SCT helper：`PocketVE/code/models/sct_loss.py`
- PocketVE 模型和训练 loss：`PocketVE/code/models/molopt_score_model.py`
- PocketVE 动态图实现：`PocketVE/code/models/uni_transformer.py`
- PocketVE 训练循环：`PocketVE/code/scripts/train_diffusion.py`
- PocketVE 单元测试：`PocketVE/code/tests/test_sct_loss.py`

## 2. 总体结论

当前 SCT 的核心实现总体正确。没有发现会导致 JVP 方向、stop-gradient、virtual
teacher 或训练总 loss 完全错误的高优先级问题。

当前实现正确完成了：

- coordinate single-time SCT；
- atom categorical logit-SCT；
- 可选 bond categorical logit-SCT；
- linear/log-sigma joint JVP；
- `fixed-c`、`scale-c` 和 `no-norm` tangent normalization；
- teacher/tangent stop-gradient；
- graph-wise reduction；
- SCT loss 到训练总 loss 的接入。

纯 loss 单元测试 19 项全部通过，真实 GPU 模型级 JVP、有限差分和 backward
检查也通过。

需要重点注意的不是 SCT 主公式错误，而是：

1. 当前主要训练配置采用 `log + fixed-c`，可能显著压低低噪声区域的 SCT 信号；
2. PocketVE coordinate loss 的 xyz reduction 与参考仓库相差一个因子 3；
3. 梯度累积时训练控制台输出的总 loss 数值不正确；
4. 完整模型级自动化集成测试仍不足。

## 3. 发现的问题

### 3.1 中优先级：主要训练配置使用 `log + fixed-c`

当前配置 `PocketVE/code/configs/training_cfg_muon_sct.yml` 使用：

```yaml
use_sct_coordinate_consistency: true
sct_time_parameterization: log
sct_tangent_norm_mode: fixed-c
sct_tangent_norm_constant: 0.1
```

`fixed-c` 本身不是实现错误，参考仓库也支持该模式。但在 log-sigma
参数化下，其 normalized tangent 为：

\[
\frac{\sigma\tau}
{\operatorname{RMS}(\sigma\tau)+c}.
\]

当 \(\sigma\) 很小时，固定的 `c=0.1` 容易主导分母，使 normalized tangent
及 SCT 信号随 \(\sigma\) 一起趋近于零。

如果目标是切换到 log-sigma 时间坐标，但保持与 linear-time 相近的 normalized
tangent 尺度，应使用 `scale-c`：

\[
\frac{\sigma\tau}
{\operatorname{RMS}(\sigma\tau)+\sigma c}
=
\frac{\tau}{\operatorname{RMS}(\tau)+c}.
\]

建议：

- 将 `fixed-c` 明确视为历史模式或对照实验；
- 正式的 log-sigma SCT 实验优先考虑 `scale-c`；
- 不应把现有 fixed-c checkpoint 的结果直接解释为 scale-invariant log-SCT。

### 3.2 中优先级：coordinate SCT reduction 与参考实现相差因子 3

PocketVE 在 `PocketVE/code/models/molopt_score_model.py:1130` 使用：

```python
pos_atom_loss = residual.square().sum(dim=-1)
```

即先对 xyz 三个坐标求和，再在 graph 内对 atom 求平均。

参考仓库在 `VEDA-shortcut-sct/models/fm.py:2176` 的
`_masked_coordinate_mean()` 中对所有有效 scalar coordinates 求平均。因此对相同
的 coordinate residual：

\[
L_{\mathrm{PocketVE,pos}} = 3L_{\mathrm{reference,pos}}.
\]

这与 PocketVE 原有 coordinate loss 的 xyz 求和约定一致，因此可以视为有意的
架构适配，而不是必须修改的错误。但其后果是：

- PocketVE 与参考仓库的 `sct_coord_loss_weight` 数值不能直接比较；
- 若要严格匹配参考实现的 coordinate SCT 强度，PocketVE 权重应除以约 3；
- 当前 `sct_coord_loss_weight: 0.3` 按参考 reduction 折算后约相当于 `0.9`。

### 3.3 明确错误：梯度累积时控制台输出的总 loss 不正确

`PocketVE/code/scripts/train_diffusion.py:349-365` 中先执行：

```python
loss_original = results.get('loss_original', loss)
loss_sct = results.get('loss_sct', ...)
loss = loss / config.train.n_acc_batch
loss.backward()
```

之后直接打印已经除以 `n_acc_batch` 的 `loss`，但打印的 `loss_original` 和
`loss_sct` 没有除。因此当 `n_acc_batch=2` 时，日志可能出现：

```text
Loss 2.28 (original 4.37 | sCT 0.19 ...)
```

这些显示值显然不满足：

\[
L_{\mathrm{total}}=L_{\mathrm{original}}+L_{\mathrm{SCT}}.
\]

实际反向传播没有问题：每个 microbatch loss 除以累计次数是正确的。错误只在
日志层面，但会明显误导训练诊断。

此外，目前日志只保留最后一个 microbatch 的 `results`，而不是所有累计
microbatch 指标的平均值。

建议分别维护：

- 未缩放、用于日志的 batch loss；
- 除以 `n_acc_batch`、用于 backward 的 scaled loss；
- 跨 microbatch 累计后的平均日志指标。

### 3.4 低优先级：模型级自动化集成测试不足

`PocketVE/code/tests/test_sct_loss.py` 中现有 19 项测试全部通过，但主要覆盖 pure
loss helper。尚未自动覆盖以下关键集成行为：

- SCT 全关闭时不调用 JVP，也不增加额外 forward；
- condition dropout 在 student 和 teacher 之间严格一致；
- student/teacher 复用相同 noisy atom/bond state；
- coordinate、atom 和 bond outputs 使用同一个 JVP 方向；
- 固定动态图邻接后的真实模型有限差分检查；
- coordinate weighted loss 明确按 `1 / sct_coord_probe_step` 计算；
- bond SCT 的完整 forward/backward；
- 新配置加载旧 checkpoint；
- sampling 输出不受 SCT 训练配置影响。

`PocketVE/docs/sct_modification_guide.md:608-623` 已经列出了这些集成测试，但当前
测试文件尚未全部实现。

### 3.5 低优先级：一个运行过的实验配置不在正式配置目录中

正在运行或曾运行的实验命令引用：

```text
configs/training_cfg_muon_sct_from_scratch.yml
```

该文件目前不在 `PocketVE/code/configs/` 中，只能在对应的训练日志目录中找到
保存副本。这不影响已经启动的训练，但原始命令不能直接复现。

建议将最终实验配置恢复到正式配置目录，或把日志目录中的 resolved config
明确作为该实验的归档配置。

### 3.6 低优先级：SCT 关闭时 baseline 数值路径并非绝对逐 bit 不变

为支持 JVP，`PocketVE/code/models/uni_transformer.py` 将多个距离计算替换为：

```python
sqrt(sum(x**2) + 1e-8)
```

这能避免零距离处 norm 的 JVP 非有限问题，是合理修改。但即使 SCT 关闭，距离
的 primal value 也会发生极小变化，尤其零距离会从 `0` 变为 `1e-4`。

一般不会造成实质训练错误，但“关闭 SCT 后 baseline 完全逐 bit 不变”并不严格
成立。若需要严格回归，应增加关闭 SCT 前后的数值对比测试。

## 4. 已确认正确的实现

### 4.1 共享 noisy state 和时间

PocketVE 只采样一次：

- graph-level sigma；
- ligand coordinate noise；
- noisy atom state；
- noisy bond state；
- protein pocket noise；
- condition dropout。

student forward 和 JVP teacher 使用同一组状态。没有重新采样第二个时间、第二份
坐标噪声或第二次 categorical corruption。

### 4.2 Joint JVP 方向正确

连续路径为：

\[
x_\sigma=x_0+\sigma\epsilon.
\]

linear-time 使用：

\[
(\dot{x},\dot{\sigma})=(\epsilon,1).
\]

log-sigma 使用：

\[
(\dot{x},\dot{\sigma})=(\sigma\epsilon,\sigma).
\]

实现位于 `PocketVE/code/models/molopt_score_model.py:997-1046`。coordinate、atom
logits 和可选 bond logits 全部来自同一个 `torch.func.jvp()`。

### 4.3 Teacher 和 tangent 正确 stop-gradient

JVP teacher 使用当前 live model 参数，但整个 JVP 位于 `torch.no_grad()` 中，
并在返回前对 teacher outputs 和 tangent outputs 显式 `detach()`。

SCT 梯度只通过先前已有的 student forward 输出传播，符合参考实现的 surrogate
gradient 设计。

### 4.4 Coordinate SCT residual 符号和 probe scaling 正确

当前实现使用：

\[
r_{mathrm{SCT,pos}}
=D_{\mathrm{student}}
-\operatorname{sg}(D_{\mathrm{teacher}})
+a_x\operatorname{sg}(\widetilde{\tau}).
\]

实现位于 `PocketVE/code/models/sct_loss.py:326-369`。

最终 coordinate optimization loss 还会除以 `sct_coord_probe_step`，并乘以：

- `loss_pos_weight`；
- `sct_coord_loss_weight`。

### 4.5 Categorical logit-SCT 正确

categorical virtual teacher 为：

\[
q_a=\operatorname{softmax}
\left(z_{\mathrm{teacher}}-a\widetilde{\tau}_z\right),
\]

loss 为：

\[
\frac{1}{a}\operatorname{KL}
\left(q_a\,\middle\|\,\operatorname{softmax}(z_{\mathrm{student}})\right).
\]

实现位于 `PocketVE/code/models/sct_loss.py:372-491`，其符号、KL 方向、probe
归一化及 teacher detach 都与参考实现一致。

### 4.6 Tangent normalization 正确

实现支持：

- `fixed-c`；
- `scale-c`；
- `no-norm`。

coordinate tangent 按 graph 对所有有效 scalar coordinates 计算 RMS；categorical
tangent 先去除 class-constant gauge，再按 token 对 class 维计算 RMS。

`no-norm` 模式下 categorical branch 仍保留 gauge projection，行为正确。

### 4.7 Graph-wise reduction 正确

- coordinate：先 graph 内对 atom 平均，再跨 graph 平均；
- atom logit-SCT：先 graph 内对 token 平均，再跨 graph 平均；
- bond logit-SCT：先 graph 内对 \(N_g^2\) bond tokens 平均，再跨 graph 平均。

因此大分子不会仅因拥有更多 atom/bond tokens 而在 batch 中获得更大权重。

### 4.8 总 loss accounting 正确

模型内部最终使用：

\[
L_{\mathrm{train}}
=L_{\mathrm{original}}
+L_{\mathrm{SCT,pos}}
+L_{\mathrm{SCT,atom}}
+L_{\mathrm{SCT,bond}}.
\]

diagnostic tensors 没有重复加入总 loss。

## 5. 验证结果

### 5.1 CPU 单元测试

运行：

```bash
pytest -q PocketVE/code/tests/test_sct_loss.py
```

结果：

```text
19 passed
```

联合运行其他 transformer 测试时，本地默认 Python 环境缺少 `torch_scatter`，
因此 `test_uni_transformer_scatter.py` 无法在登录节点收集。这是环境依赖问题，
不是测试断言失败。

### 5.2 GPU 模型级检查

按照项目要求，通过 SSH 在 GPU 节点使用真实 `ScorePosNet3D` 执行了小图检查。

teacher/student 同状态输出最大差异：

```text
coordinate: 1.19e-7
atom logits: 1.19e-7
```

这说明固定相同邻接拓扑后，teacher 和 student 的 primal forward 一致。

log-sigma JVP 与有限差分的相对误差：

| 有限差分步长 | Coordinate | Atom logits |
|---:|---:|---:|
| `1e-2` | `5.92e-3` | `3.37e-2` |
| `3e-3` | `1.80e-3` | `1.47e-2` |
| `1e-3` | `6.25e-4` | `7.10e-3` |

误差随步长减小而下降，说明 joint JVP 方向正确。

完整 loss/backward 检查：

```text
loss     = 17.649246
original = 17.409386
sct      = 0.239860
```

满足：

```text
loss = original + sct
```

所有参数梯度均为 finite。

## 6. 当前缺失功能

以下功能优先级低于前述配置和日志问题。

### 6.1 Velocity-space coordinate SCT

参考仓库支持在 sampler reverse velocity 空间计算 coordinate SCT；PocketVE 当前
只支持 x0 prediction space。

若补充该功能，不能简单把 x0 tangent 除以 sigma，必须实现带 `sigma_min` floor
的商法则 JVP。

### 6.2 Charge SCT

参考仓库包含 charge categorical head 和 charge logit-SCT。PocketVE 当前没有
charge prediction head，因此暂时无法实现 charge SCT。

### 6.3 Atom SCT 与 bond SCT 的独立开关

当前 `discrete_consistency_mode: logit_sct` 会启用 atom SCT；当 `bond_loss: true`
且 bond target 可用时也会计算 bond SCT。

可以通过把 bond SCT 权重设为 0 关闭其优化贡献，但仍会执行相应 JVP 输出和 KL
计算。当前没有完全独立的 atom/bond SCT 开关。

### 6.4 自动化模型级有限差分测试

本次人工 GPU 检查已验证 JVP，但尚未将该检查加入测试套件。建议增加小模型或
小图版本，固定：

- noise；
- sigma；
- condition dropout；
- dynamic edge topology。

然后分别验证 linear 和 log-sigma JVP。

### 6.5 明确不属于当前 single-time SCT 的功能

以下功能参考仓库当前 canonical SCT 路径也不要求，不能视为当前实现错误：

- EMA/target-network teacher；
- dual-timestep 或 pairwise consistency；
- finite-step trajectory loss；
- representation/REPA loss；
- 用 finite difference 替代训练时 JVP；
- SCT 专用采样 scheduler。

## 7. 建议处理顺序

1. 确认正式实验应使用 `fixed-c` 还是 `scale-c`；若目标是 scale-invariant
   log-sigma SCT，切换为 `scale-c`。
2. 修复梯度累积下的训练日志，使 total/original/SCT 指标使用一致尺度并跨
   microbatch 平均。
3. 明确记录 coordinate reduction 的因子 3，避免跨仓库直接比较 SCT 权重。
4. 补充模型级 integration tests，尤其是 condition dropout、禁用路径和 joint
   JVP 有限差分。
5. 恢复或归档缺失的 from-scratch 实验配置。
6. 根据实际需求再考虑 velocity-space SCT、独立 atom/bond 开关等低优先级功能。
