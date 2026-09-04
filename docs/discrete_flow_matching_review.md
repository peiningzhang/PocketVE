# PocketVE Discrete Flow Matching 实现审查

## 1. 总体结论

PocketVE 当前所谓的 discrete flow matching（DFM）可以分成三个相对独立的部分：

1. 训练时使用 uniform-prior categorical corruption 构造带噪 atom/bond state；
2. 模型通过 clean-label cross entropy 预测原始 atom/bond 类别；
3. 推理时使用 `campbell` 或 `gat` categorical sampler 更新离散状态。

训练端的默认 `uniform` corruption 与 clean-label CE 基本自洽。主要正确性问题集中在
默认使用的 `campbell` sampler：其概率归一化破坏了小步长极限，而且代码计算出的
重新加噪概率没有实际作用到离散状态。因此当前推理过程不应称为正确实现的
Campbell continuous-time discrete flow。

当前默认配置为：

```yaml
discrete_prior: uniform
dfm_beta: 2
dfm_type: campbell
mask_mode: uniform
```

需要特别注意：`dfm_beta` 当前没有被任何 Python 代码读取，所以配置中的该参数实际
无效。

## 2. 当前训练端是什么样的

### 2.1 时间和 corruption rate

训练与连续坐标部分共享同一个 EDM noise level `sigma`。默认训练配置使用
log-normal 分布抽样 sigma，然后 clamp 到：

```text
[sigma_min, sigma_max]
```

当前 SCT Muon 配置为：

```text
sigma_min  = 0.001
sigma_max  = 800
sigma_data = 10
```

`mask_mode: uniform` 的实际含义不是“uniform categorical prior”；它表示 mask rate
对 `log(sigma)` 线性：

\[
m(\sigma)=
\frac{\log\sigma-\log\sigma_{\min}}
{\log\sigma_{\max}-\log\sigma_{\min}}.
\]

因此：

```text
sigma = sigma_min  -> m = 0
sigma = sigma_max  -> m = 1
```

实现位于 `PocketVE/code/utils/misc.py:139-146`。

另一个可选模式 `mask_mode: edm` 使用：

\[
m(\sigma)=\frac{\sigma}{\sigma+\sigma_{data}}.
\]

### 2.2 Atom corruption

给定 clean atom class \(v_0\)，训练带噪状态从以下分布采样：

\[
q(v_\sigma\mid v_0)
=(1-m(\sigma))\operatorname{OneHot}(v_0)
+m(\sigma)\frac{1}{S}.
\]

实现位于 `PocketVE/code/models/molopt_score_model.py:694-714`。

这等价于：

- 以 `1 - mask_rate` 保留 clean class；
- 以 `mask_rate` 从所有 atom classes 均匀重采样。

因为均匀重采样仍有 `1/S` 的概率采中原类别，所以实际类别发生变化的概率是：

\[
m(\sigma)\left(1-\frac{1}{S}\right),
\]

而不是严格等于 `mask_rate`。

### 2.3 Bond corruption

当 `bond_loss: true` 时，代码把每个分子的 dense \(N_g\times N_g\) adjacency 展平，
将以下五类作为 categorical state：

```text
0: no bond
1: single
2: double
3: triple
4: aromatic
```

bond 使用和 atom 相同的 mask-rate schedule，并从 5 类 uniform prior 中重采样。

当前主 SCT 配置中：

```yaml
bond_loss: false
sample_bond: false
```

所以当前主训练实验实际上只有 atom DFM，没有启用 bond DFM。

### 2.4 训练目标

模型不是直接回归 categorical flow/rate matrix，而是预测 clean endpoint
distribution：

\[
p_\theta(v_0\mid v_\sigma,\sigma,\text{pocket}).
\]

Atom loss 是 clean target cross entropy：

```python
loss_v_atom = F.cross_entropy(pred_ligand_v, ligand_v, reduction='none')
loss_v = scatter_mean(loss_v_atom, batch_ligand, dim=0).mean()
```

即先对每个 graph 的 atoms 平均，再跨 graph 平均。实现位于
`PocketVE/code/models/molopt_score_model.py:1548-1552`。

Bond loss 同样是 clean-label CE，但 class 0（no-bond）使用
`loss_non_bond_weight` 降权，以避免 dense adjacency 中 no-bond 类占主导。

这种训练方式更准确地说是：

```text
categorical denoising endpoint prediction + DFM-inspired sampler
```

它不是直接监督 probability velocity 的传统 flow-matching regression，但可以为正确
推导的 posterior/ratio-based categorical sampler 提供 clean distribution。

### 2.5 Discrete logit-SCT 的位置

`discrete_consistency_mode: logit_sct` 是额外的 logits consistency loss。它复用同一
带噪 categorical state，仅对连续坐标路径和 sigma 做 joint JVP。

它不会改变：

- DFM corruption distribution；
- clean CE target；
- Campbell/GAT sampling kernel。

所以 DFM sampler 的问题不能由 logit-SCT 自动修复。

## 3. 当前推理端是什么样的

### 3.1 初始状态

Atom type 从 uniform categorical prior 初始化。`sample_diffusion.py:68-73` 中直接使用
全零 logits 采样，因此每个 atom class 等概率。

当 bond sampling 启用时，所有 \(N_g^2\) entries 也从五类 uniform prior 独立初始化。

### 3.2 Sigma 和 mask-rate 方向

推理从 `sigma_max` 走到 `sigma_min`。在默认 `mask_mode: uniform` 下，对应：

```text
mask rate: 1 -> 0
t = 1 - mask_rate: 0 -> 1
```

因此从完全 categorical noise 逐渐走向 clean endpoint。

### 3.3 默认 Campbell update

当前所有主要配置都使用：

```yaml
dfm_type: campbell
```

模型先给出：

\[
p_1=p_\theta(v_0\mid v_t,t).
\]

然后代码定义：

\[
\Delta t=t_{next}-t,
\qquad \alpha_t=t,
\]

并计算：

\[
s=\frac{\text{campbell\_stochasticity}}{m'(\sigma)},
\]

\[
u=\Delta t\frac{\alpha'_t+s\alpha_t}{1-\alpha_t},
\qquad
r=\Delta t\,s.
\]

代码把 `u` 称为 `unmask_prob`，把 `r` 称为 `mask_prob`。

理论上，如果 `u` 和 `r` 是一个小时间步内两类事件的概率，它们都应该是
\(O(\Delta t)\)，并在 \(\Delta t\rightarrow 0\) 时趋近于 0。

## 4. 发现的错误

### 4.1 高优先级：Campbell `unmask_prob` 的归一化破坏小步长极限

在 `PocketVE/code/models/molopt_score_model.py:1837-1842` 中，代码先计算并 clamp
`unmask_prob` 与 `mask_prob`，随后执行：

```python
unmask_prob = unmask_prob / (unmask_prob + mask_prob)
```

源码自己也注明：

```python
# 这里的归一化并不是符合理论的，需要重新推导
```

问题在于归一化前：

\[
u=O(\Delta t),\qquad r=O(\Delta t),
\]

但归一化后：

\[
\frac{u}{u+r}=O(1).
\]

因此当 sampling step 变小时，token replacement probability 不会趋近 0。增加采样
步数不会收敛到同一个 continuous-time process，反而会增加大量 O(1) 概率的类别
重采样。

极端例子：如果 `campbell_stochasticity=0`，则 `mask_prob=0`，上述归一化会使：

```text
unmask_prob = 1
```

于是除第一步数值边界外，每一步都会重新从模型预测分布采样所有 token。这不是
Euler/CTMC 小步长离散化。

### 4.2 高优先级：`mask_prob` 没有实际更新 categorical state

当前状态更新为：

```python
x1 = Categorical(p_1_given_t).sample()
will_unmask = rand < unmask_prob
ligand_v_next = ligand_v.clone()
ligand_v_next[will_unmask] = x1[will_unmask]
```

虽然代码计算了 `mask_prob`，但没有按照 `mask_prob` 把任何 atom 重新采样到 uniform
prior。

因此 `campbell_stochasticity` 所声称的“额外随机重新加噪”没有直接作用到实际
`ligand_v_next`；它只通过分母间接改变 `unmask_prob`。

与此同时，返回用于日志的 `p_step` 却包含：

\[
(1-u-r)X_t+u p_1+rU.
\]

所以：

- `ligand_v_next` 的实际采样 kernel 不含 uniform re-noising；
- 返回的 `p_step` 却声称含有 uniform re-noising；
- trajectory 中记录的 probability 与实际 categorical state transition 不一致。

Atom 和 bond 的 Campbell 函数都有同样的问题。

### 4.3 高优先级：默认 100-step 配置产生异常大的 token replacement

使用当前主配置：

```text
sigma_min = 0.001
sigma_max = 800
mask_mode = uniform
campbell_stochasticity = 2.0（代码隐式默认）
sampling steps = 100
```

按当前代码直接计算，部分步骤的实际 `will_unmask` 概率为：

| Step | Sigma | `t=1-mask_rate` | 实际 replacement probability |
|---:|---:|---:|---:|
| 0 | 800 | 0.00 | 0.0099 |
| 1 | 698.3 | 0.01 | 0.5000 |
| 10 | 205.5 | 0.10 | 0.5000 |
| 25 | 26.75 | 0.25 | 0.5000 |
| 50 | 0.894 | 0.50 | 0.5198 |
| 75 | 0.0299 | 0.75 | 0.8879 |
| 90 | 0.00389 | 0.90 | 0.9904 |
| 98 | 0.00131 | 0.98 | 0.9993 |
| 99 | 0.00115 | 0.99 | 0.9997 |

也就是说，大量步骤会重新采样约 50% 的 atom types，后期接近每一步重采样所有
atom types。这更像反复 Gibbs-style resampling，而不是对一个 categorical flow 的
小步积分。

这可能解释不同 sampling step 数量对结果产生非收敛或反常影响。

### 4.4 中优先级：`dfm_beta` 配置完全未使用

所有主要配置都含有：

```yaml
dfm_beta: 2
```

注释称其控制 predictor-corrector noise injection，但在整个 `PocketVE/code` 中，
`dfm_beta` 只出现在 YAML 文件，没有任何 Python 代码读取它。

实际使用的是两个不同的隐式参数：

- GAT sampler：`corrector_weight`，缺省为 `0.1`；
- Campbell sampler：`campbell_stochasticity`，缺省为 `2.0`。

当前配置文件没有显式设置这两个参数。因此修改 `dfm_beta` 不会改变任何训练或采样
行为，checkpoint/config 也没有准确记录实际 sampler 超参数。

### 4.5 中优先级：`discrete_prior: marginal` 实际没有完整实现

初始化模型时，`discrete_prior: marginal` 可以构造 `self.prior_dist`。但训练 corruption
仍硬编码为：

```python
mask_rate / num_classes
```

而不是：

```python
mask_rate * self.prior_dist
```

采样脚本初始化 atom types 时也始终使用 uniform logits，没有使用
`model.prior_dist`。

因此选择 `discrete_prior: marginal` 后：

- 训练带噪分布仍是 uniform prior；
- 初始采样分布仍是 uniform prior；
- 只有额外 noise-injection 分支的一部分会使用 `self.prior_dist`。

当前正式配置使用 `uniform`，所以这不会影响现有实验，但该配置选项目前是残缺的。

### 4.6 中优先级：GAT sampler 的 schedule 定义与训练 interpolant 不一致

训练 corruption 的 clean coefficient 是：

\[
1-m(\sigma).
\]

但 `DFMTimeScheduler.kappa()` 返回：

\[
\kappa(\sigma)=(1-m(\sigma))^2.
\]

此外，若 `d_kappa_dt()` 被解释为对 sigma 的普通导数，则实际应为：

\[
\frac{d\kappa}{d\sigma}
=-2(1-m)m'(\sigma),
\]

而当前函数返回正值：

```python
2 * (1 - mask_rate) * mask_rate_derivative
```

它可能想表达 reverse-time derivative magnitude，但函数名、docstring 和调用方式没有
明确这一点。更重要的是，平方后的 `kappa` 与训练实际使用的 mixture coefficient
不一致。

当前配置默认使用 Campbell，因此该问题不影响主要运行路径，但 `dfm_type: gat` 目前
也不能视为经过验证的替代方案。

### 4.7 中优先级：Bond sampling 没有强制有效 adjacency constraints

Bond state 包含完整 \(N_g^2\) 矩阵，初始化和更新均逐 entry 独立采样。当前代码没有
强制：

- 对角线 `(i,i)` 必须是 no-bond；
- `(i,j)` 与 `(j,i)` 的 bond type 必须一致。

`_flat_bond_to_index_type()` 会保留所有 class 非 0 的 entries，包括可能的 self-bond。
因此若启用 `sample_bond`，中间状态和最终状态都可能包含：

- self-loop chemical bonds；
- 两个方向类别不同的非对称 bond。

当前主配置关闭 bond sampling，所以这是潜在功能错误，不影响当前 atom-only DFM。

### 4.8 低优先级：默认推理不是纯 DFM sampler

`sample_diffusion()` 的 `noise_injection` 默认是 `True`。在指定 sigma 范围内，它会：

- 增大当前 sigma；
- 给坐标增加额外 Gaussian noise；
- 根据 mask-rate 增量把 atom/bond 重新采样到 prior。

这套外部 noise injection 与 Campbell 内部的 `campbell_stochasticity` 同时存在。因此
默认评估实际运行的是：

```text
Campbell-like repeated resampling + external re-noising
```

而不是单一、容易解释的 DFM integration kernel。比较 sampler 或 sampling steps 时，
应明确记录是否使用 `--no-veda-noise-injection`。

### 4.9 低优先级：没有 DFM 专用测试

当前 `PocketVE/code/tests` 中没有覆盖以下行为：

- mask-rate endpoint 和单调性；
- training corruption 的经验分布；
- Campbell/GAT transition row sums；
- transition probability 的 \(O(\Delta t)\) 小步长性质；
- `p_step` 与实际 sample kernel 一致；
- uniform/marginal prior 一致性；
- bond diagonal 和 symmetry constraints；
- sampling-step refinement convergence。

因此上述明显的 sampler 问题没有被自动化测试捕获。

## 5. 哪些部分是正确的

以下实现本身没有发现明显问题：

- 默认 log-uniform mask rate 在 `sigma_min` 和 `sigma_max` 达到精确的 0/1 endpoint；
- atom corruption 正确按 graph/atom sigma broadcast；
- atom clean-label CE 先 graph 内平均，再跨 graph 平均；
- bond training 同时包含 bond 和 no-bond entries；
- no-bond class 可以通过 class weight 降权；
- atom 和 bond 可共享同一个 sigma/mask-rate schedule；
- 模型输出 clean categorical logits，适合作为 posterior endpoint predictor；
- discrete logit-SCT 与原始 CE 分开计权，并正确加入总 loss；
- 当前 `uniform` prior 的训练 corruption 与 uniform sampling initialization 一致。

## 6. 建议的修复方向

### 6.1 第一优先级：重新实现单步 categorical transition

不要使用：

```python
unmask_prob = unmask_prob / (unmask_prob + mask_prob)
```

作为“本步发生 replacement”的总概率。

应先明确目标 CTMC/rate matrix，再由有限步 transition 构造完整 categorical row：

```text
current-state mass
+ model-predicted clean transition mass
+ optional prior-noise transition mass
```

所有 off-diagonal probability 的总质量必须满足：

\[
P(\text{state changes in one step})=O(\Delta t).
\]

然后直接从该完整 `step_probs` 采样 actual next state，保证返回的 `p_step` 与实际
transition 完全一致。

参考仓库 `VEDA-shortcut-sct/models/fm.py::_uniform_sample_step()` 已具备更合理的结构：

- 先计算 off-diagonal rate × step size；
- 清零当前类别对应的 off-diagonal entry；
- 当 off-diagonal 总质量大于 1 时按整行同比缩放；
- 剩余质量放到 diagonal；
- 直接从该 probability row 采样。

### 6.2 第二优先级：统一配置参数

建议删除无效的 `dfm_beta`，或把它明确映射为唯一的 sampler stochasticity 参数。

配置应显式记录：

```yaml
dfm_type: ...
dfm_stochasticity: ...
mask_rate_strategy: ...
discrete_prior: ...
```

不要分别依赖未写入配置的 `corrector_weight=0.1` 和
`campbell_stochasticity=2.0` 隐式默认值。

### 6.3 第三优先级：决定保留哪个 sampler

推荐先保留一套经过推导和测试的 categorical sampler。当前 Campbell 与 GAT 两条
路径都存在定义不一致，双路径会扩大验证成本。

可优先将当前训练 interpolant：

\[
q_t=(1-m_t)\delta_{v_0}+m_t U
\]

与参考实现的 uniform-sample reverse kernel 对齐，再考虑 predictor-corrector 或
额外 stochasticity。

### 6.4 补充测试

至少增加：

1. `mask_rate(sigma_min)=0`、`mask_rate(sigma_max)=1`；
2. empirical corruption frequency 与解析 mixture 一致；
3. 单步 transition 非负且 row sum 为 1；
4. 将 step size 缩小一半时，off-diagonal mass 也近似缩小一半；
5. 返回 `p_step` 的 empirical samples 与其概率一致；
6. 10/20/50/100 steps 的 sampler refinement sanity check；
7. marginal prior 的训练、初始化和 re-noising 全路径一致；
8. bond diagonal 恒为 no-bond，并明确处理 symmetry。

## 7. 建议处理顺序

1. 修复 Campbell `unmask_prob` 归一化和未执行 `mask_prob` 的问题；
2. 用完整 `step_probs` 直接采样 next categorical state；
3. 增加小步长收敛和 empirical transition 测试；
4. 清理或接通无效的 `dfm_beta`；
5. 决定 `gat` 分支是修复、重写还是移除；
6. 完整实现或删除 `discrete_prior: marginal`；
7. 若启用 bond generation，加入 diagonal/symmetry constraints；
8. 将外部 noise injection 与基础 DFM sampler 分开做消融实验。
