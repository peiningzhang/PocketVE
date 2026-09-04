# PocketVE 连续与离散 sCT Loss 修改指南

## 1. 目标与范围

本指南说明如何参考
`/shared/healthinfolab/phz24002/VEDA-shortcut-sct`，在 PocketVE 的 VEDA
训练路径中加入 single-time sCT（single-time consistency training）loss。

第一版实现范围：

- 连续部分：ligand 坐标 `x0` prediction 的 coordinate sCT；
- 离散部分：ligand atom type logits 的 categorical logit-sCT；
- 可选离散部分：当 `bond_loss: true` 时，对 bond logits 使用 logit-sCT；
- coordinate 和 categorical branch 共用 `linear` / `log` 时间参数化，
  在 PocketVE 中 `log-t` 具体表示 `log-sigma`；
- coordinate 和 categorical branch 共用可配置的 tangent normalization
  mode：`fixed-c` / `scale-c` / `no-norm`；
- coordinate 和 categorical branch 分别使用独立的 probe step；
- 不增加 charge sCT，因为 PocketVE 当前没有 charge prediction head；
- 不改变采样器，也不引入第二时间点、finite step、EMA teacher、pairwise
  consistency 或 trajectory consistency；
- 所有新功能默认关闭，以保持旧配置和 checkpoint 的行为不变。

参考实现：

- `VEDA-shortcut-sct/docs/consistency_training.md`
- `VEDA-shortcut-sct/models/fm.py::_compute_sct_joint_target`
- `VEDA-shortcut-sct/models/fm.py::_compute_sct_loss`
- `VEDA-shortcut-sct/models/fm.py::_sct_jvp_directions`
- `VEDA-shortcut-sct/models/fm.py::_sct_effective_norm_constant`
- `VEDA-shortcut-sct/tests/test_loss_accounting.py`

本文档的 log-time 和 tangent normalization 设计以上游
`4347979` 为基准；该版本包含 `b188d9a` / `7c6316c` 的
log-t 变更、`e256b69` 的 scale-c 修正，以及最新的可配置
normalization modes。

## 2. PocketVE 当前训练路径

主要入口如下：

- 模型和 VEDA loss：`code/models/molopt_score_model.py`
- 训练循环：`code/scripts/train_diffusion.py`
- 默认配置：`code/configs/training.yml`
- Muon 配置：`code/configs/training_cfg_muon.yml`
- 动态构图和消息传递：`code/models/uni_transformer.py`

`ScorePosNet3D.get_diffusion_loss()` 当前执行：

1. 每个 graph 采样一个 `sigma`；
2. 构造 ligand 连续带噪状态
   `x_sigma = x0 + sigma * epsilon`；
3. 采样 noisy atom types，以及可选的 noisy bonds；
4. 对同一个 noisy state 执行一次模型 forward；
5. 计算 coordinate MSE、atom cross entropy、可选 bond cross entropy 和
   certainty loss。

sCT 必须复用第 2 至第 4 步产生的同一个状态，不能重新采样噪声、条件
dropout 或时间。

## 3. 数学定义

### 3.1 共享轨迹、时间参数化和 JVP

PocketVE 的连续路径为：

\[
x_\sigma = x_0 + \sigma\epsilon,
\qquad
\frac{d x_\sigma}{d\sigma}=\epsilon.
\]

定义一个全局 consistency coordinate $q$：

\[
q=\sigma\quad\text{(`linear`)},
\qquad
q=\log\sigma\quad\text{(`log`)}.
\]

在同一个 `(x_sigma, sigma)` 上，使用 live model 计算 teacher 和对
$q$ 的 joint JVP：

\[
f_\sigma=f_\theta(x_\sigma,\sigma),
\qquad
\tau_q
=J_xf_\sigma\frac{dx_\sigma}{dq}
+\partial_\sigma f_\sigma\frac{d\sigma}{dq}.
\]

令

\[
\alpha(\sigma)=\frac{d\sigma}{dq}
=\begin{cases}
1,&q=\sigma,\\
\max(\sigma,\sigma_{\min}),&q=\log\sigma,
\end{cases}
\]

则 JVP 方向为 `(alpha * epsilon, alpha)`。PocketVE 训练时本来就将
`sigma` clamp 到 `[sigma_min, sigma_max]`；`max` 仍需保留，使 helper
和测试对越界输入也是数值安全的。实现可写为：

```python
time_scale = sct_time_scale(
    sigma_per_atom,
    sigma_min=self.sigma_min,
    time_parameterization=self.sct_time_parameterization,
)
torch.func.jvp(
    teacher_fn,
    (ligand_pos_perturbed, sigma_per_atom),
    (time_scale * pos_noise, time_scale),
    strict=False,
)
```

`time_scale` 的 shape 必须可与 `[num_ligand_atoms, 3]` 坐标和
`sigma_per_atom` 正确 broadcast。同一对 JVP 方向同时用于 coordinate、
atom logits 和 bond logits，不得只对 coordinate branch 切换 log-time。

teacher 使用当前模型参数，但 teacher 输出和 tangent 必须全部
`detach()`。sCT 只通过普通 student forward 的输出传播梯度。

以下输入在 JVP 中保持不变：

- noisy protein pocket；
- noisy atom types；
- noisy bond state；
- condition bins 和 condition dropout mask；
- batch index 和动态图的离散邻接拓扑。

### 3.2 Coordinate sCT

首先对 coordinate tangent 按 graph 做 RMS normalization。PocketVE 使用
flat PyG batch，因此对 graph `g`：

\[
r_g=\sqrt{
\frac{1}{3N_g}
\sum_{i:\,b_i=g}\sum_{d=1}^{3}\tau_{i,d}^2
}.
\]

其中 `c` 为 `sct_tangent_norm_constant`。对 graph $g$ 还要取
与该 graph 的 `sigma` 对应的 $\alpha_g$。

`sct_tangent_norm_mode` 定义 normalized tangent：

\[
\widetilde\tau_i=
\begin{cases}
\displaystyle\frac{\tau_i}{r_{b_i}+c},
&\text{`fixed-c`},\\[8pt]
\displaystyle\frac{\tau_i}{r_{b_i}+\alpha_{b_i}c},
&\text{`scale-c`},\\[8pt]
\tau_i,&\text{`no-norm`}.
\end{cases}
\]

`fixed-c` 使用与时间无关的 additive floor，也是上游默认值：

\[
\frac{\sigma\tau_\sigma}
{\operatorname{RMS}(\sigma\tau_\sigma)+c}.
\]

`scale-c` 在 `log` 模式中将 additive floor 也乘以
$\sigma_g$：

\[
\frac{\sigma\tau_\sigma}
{\operatorname{RMS}(\sigma\tau_\sigma)+\sigma c}
=\frac{\tau_\sigma}{\operatorname{RMS}(\tau_\sigma)+c}.
\]

因此 `scale-c` 的 normalized tangent 不会仅因从 linear 切换到
log-sigma 坐标而改变尺度。`no-norm` 则完全跳过 RMS 和
additive constant，coordinate branch 直接使用 joint JVP tangent。

`linear` 模式中 $\alpha_g=1$，所以 `fixed-c` 与 `scale-c` 等价。
对新的 log-sigma 训练，若目标是保持时间坐标之间的
normalization 尺度不变，应显式选择 `scale-c`；`fixed-c` 用于
保留默认/历史行为，`no-norm` 主要用于 ablation。

给定 coordinate probe step $a_x>0$，加入 warmup 后构造：

\[
r_{\mathrm{sCT,pos}}
=D_{\theta,\mathrm{student}}
-\operatorname{sg}(D_{\theta,\mathrm{teacher}})
+a_x\operatorname{sg}(w\widetilde\tau_q).
\]

为保持 PocketVE 当前坐标 loss 的量纲，loss reduction 使用：

1. 对 xyz 求和；
2. 每个 graph 内对 atom 求均值；
3. 对 graph 求均值。

最终权重：

\[
L_{\mathrm{sCT,pos}}
=\lambda_{\mathrm{sCT}}
\lambda_{\mathrm{pos}}
\frac{1}{a_x}
\operatorname{MeanGraphAtom}
\left(\lVert r_{\mathrm{sCT,pos}}\rVert_2^2\right).
\]

$a_x$ 由 `sct_coord_probe_step` 控制。用 $1/a_x$ 而不是
$1/a_x^2$ 对 optimization loss 做尺度归一，与 categorical
probe loss 的约定保持一致。日志中应同时保留未除以 $a_x$
的 raw coordinate residual MSE 和实际加入总 loss 的 weighted loss。

第一版只实现 `x0` output space。若后续需要 velocity-space sCT，必须根据
PocketVE 采样器使用的
`d=(x_sigma-D_theta(x_sigma,sigma))/sigma` 单独推导商法则，不能直接复制
参考实现中的 velocity 符号。

### 3.3 Atom/Bond logit-sCT

categorical 输入状态保持不变，只对 ligand 坐标路径和 `sigma` 求导：

\[
\tau_{z,q}
=J_xz\left(\alpha\epsilon\right)
+\partial_\sigma z\,\alpha.
\]

logits 存在 class-constant gauge direction。对每个 atom 或 bond token，先
去掉所有 class 的均值，再做 token-wise RMS normalization：

\[
\tau'_z=\tau_{z,q}-\operatorname{mean}_{class}(\tau_{z,q}),
\qquad
\widetilde\tau_z=
\begin{cases}
\displaystyle\frac{\tau'_z}
{\operatorname{RMS}_{class}(\tau'_z)+c_z},
&\text{`fixed-c`},\\[8pt]
\displaystyle\frac{\tau'_z}
{\operatorname{RMS}_{class}(\tau'_z)+\alpha c_z},
&\text{`scale-c`},\\[8pt]
\tau'_z,&\text{`no-norm`}.
\end{cases}
\]

给定 probe step `a`，构造 simplex 内的 virtual teacher：

\[
q_a=\operatorname{softmax}
\left(\operatorname{sg}(z_t)-a\operatorname{sg}(w_z\widetilde\tau_z)\right),
\]

\[
L_{\mathrm{logit-sCT}}
=\frac{1}{a}
\operatorname{KL}
\left(q_a\,\middle\|\,\operatorname{softmax}(z_s)\right).
\]

atom loss 先在 graph 内对 atom 求均值，再对 graph 求均值。bond head 当前
返回每个 graph 的 `[N_g, N_g, 5]` logits；应先扁平化每个 graph，再图内
求均值，避免大分子因为 `N_g^2` 边数获得更高权重。

`log` 模式中 $\tau_z$ 必须来自同一个 log-sigma joint JVP，
即 $\tau_{z,\log\sigma}=\sigma\tau_{z,\sigma}$。`no-norm` 仍然必须去掉
class-constant gauge 分量，只跳过后续的 RMS 和 additive-constant
除法。`time_parameterization` 和 `tangent_norm_mode` 在所有 sCT
branch 之间共享，coordinate 与 categorical 的 probe step 仍分别控制。

## 4. 配置接口

在 `ScorePosNet3D.__init__()` 中加入并校验：

```yaml
model:
  # Continuous coordinate sCT
  use_sct_coordinate_consistency: false
  sct_coord_loss_weight: 1.0
  sct_coord_probe_step: 0.1
  sct_time_parameterization: linear  # linear | log; log means log-sigma
  sct_tangent_norm_mode: fixed-c     # fixed-c | scale-c | no-norm
  sct_tangent_norm_constant: 0.1
  sct_tangent_warmup_steps: 2000

  # Categorical logit-sCT
  discrete_consistency_mode: none  # none | logit_sct
  sct_discrete_probe_step: 0.1
  sct_discrete_tangent_norm_constant: 0.1
  sct_discrete_tangent_warmup_steps: 2000
  sct_atom_loss_weight: null        # null => inherit loss_v_weight
  sct_bond_loss_weight: null        # null => inherit loss_bond_weight
```

校验规则：

- 所有 loss weight 必须大于或等于 0；
- normalization constant 和 probe step 必须大于 0；
- warmup steps 必须大于或等于 0；
- `sct_time_parameterization` 只能为 `linear` 或 `log`；
- `sct_tangent_norm_mode` 只能为 `fixed-c`、`scale-c` 或
  `no-norm`；
- `discrete_consistency_mode` 只能为 `none` 或 `logit_sct`；
- bond SCT 只有在 `bond_loss: true` 且本 batch 有 bond target 时参与总损失。

`sct_time_parameterization` 默认保持 `linear` 以兼容旧实验；新的
log-sigma 实验应显式设置为 `log`。`sct_tangent_norm_mode`
默认为 `fixed-c`；若要使用 log-sigma 下的时间坐标不变归一化，
应同时显式设为 `scale-c`。上游已将含义不明确的
`sct_loss_weight` 重命名为 coordinate-only 的 `sct_coord_loss_weight`。
如果需要读取旧 PocketVE 配置，可将 `sct_loss_weight` 作为 legacy alias；
两个字段同时出现且值不同时应报错，新配置和日志只使用新名称。

建议新增 `code/configs/training_sct.yml` 作为实验配置，不直接打开现有默认
配置中的 sCT。

## 5. 文件级修改步骤

### 5.1 新建 `code/models/sct_loss.py`

该文件只放不依赖模型结构的纯函数：

```python
def sct_warmup_factor(step, warmup_steps): ...

def sct_time_scale(
    sigma, sigma_min, time_parameterization
): ...

def graph_coordinate_rms_normalize(
    tangent,
    batch_ligand,
    norm_constant,
    norm_constant_scale=None,
    norm_mode="fixed-c",
): ...

def categorical_tangent_normalize(
    tangent_logits,
    norm_constant,
    norm_constant_scale=None,
    norm_mode="fixed-c",
): ...

def coordinate_sct_residual(
    student,
    teacher,
    normalized_tangent,
    warmup,
    probe_step,
): ...

def categorical_logit_sct_loss(
    student_logits,
    teacher_logits,
    tangent_logits,
    batch_index,
    probe_step,
    norm_constant,
    warmup,
    norm_constant_scale=None,
    norm_mode="fixed-c",
): ...
```

这些函数应进行 shape、dtype、finite value 和参数范围检查。atom 和 bond
最好共用同一个 categorical helper；bond 通过 `batch_bond` 表示每个 token
属于哪个 graph。

`norm_constant_scale` 只在 `norm_mode="scale-c"` 时生效；在 `log`
时它为每个 atom/token 对应的 `max(sigma, sigma_min)`。
`fixed-c` 必须忽略该 scale，`no-norm` 必须直接返回 coordinate
tangent 或 gauge-centered categorical tangent，同时仍返回 raw RMS 供日志
使用。coordinate helper 应检查同一 graph 内的 scale 一致，并在
graph RMS 上 broadcast；categorical helper 将 scale broadcast 到 class 维。
bond token 的 scale 由 graph-level sigma 通过
`sigma_per_graph[batch_bond]` 展开，不要根据 bond 矩阵重复采样时间。

### 5.2 修改 `ScorePosNet3D.__init__()`

- 读取第 4 节配置；
- 解析继承权重；
- 将 `sct_loss_weight` 仅作为 `sct_coord_loss_weight` 的 legacy alias；
- 校验 coordinate probe step 与全局 time parameterization；
- 读取并校验三种共享 tangent normalization mode；
- 保持旧配置缺少新字段时全部关闭；
- 不给旧 checkpoint 增加新的可训练参数。

### 5.3 固定 stochastic training state

启用 sCT 时，`get_diffusion_loss()` 必须在 student forward 前完成一次：

- sigma sampling；
- ligand coordinate noise sampling；
- atom/bond corruption；
- protein noise sampling；
- condition dropout sampling。

student 和 JVP teacher 都接收已处理的 condition bins，并设置
`condition_apply_training_dropout=False`，防止 teacher forward 再次抽样。

禁用 sCT 时保留当前代码路径，避免仅仅加入功能就改变 baseline 的随机数
消耗顺序。

### 5.4 增加 `_compute_sct_joint_target()`

建议接口：

```python
def _compute_sct_joint_target(
    self,
    *,
    protein_pos_noisy,
    protein_v,
    batch_protein,
    ligand_pos_perturbed,
    ligand_pos_velocity,
    ligand_v_perturbed,
    batch_ligand,
    sigma_per_atom,
    noisy_bond_index,
    noisy_bond_type,
    vina_bin,
    qed_bin,
    sa_bin,
    include_bond,
): ...
```

`teacher_fn(path_ligand_pos, path_sigma)` 内调用普通 `forward()`。返回 pytree：

```python
(
    pred_ligand_pos,
    pred_ligand_v,
    pred_bond_logits_flat,  # 仅 include_bond 时
)
```

bond logits 在 teacher function 内按 graph 顺序 `reshape(-1, 5)` 后拼接，以
避免在 JVP 后处理不规则 Python list。

调用 JVP 前通过 `sct_time_scale()` 构造一次 `coordinate_direction`
和 `sigma_direction`：`linear` 为 `(epsilon, 1)`，`log` 为
`(alpha * epsilon, alpha)`。该 joint JVP 返回的所有 pytree leaves
都是对同一个 $q$ 的 tangent。

使用 `torch.no_grad()` 包裹 `torch.func.jvp()`，完成后再次显式
`detach()` teacher 和 tangent。

### 5.5 增加 `_compute_sct_loss()`

建议输入 student 已经计算出的 `preds`，不要重新执行 student forward：

```python
def _compute_sct_loss(
    self,
    *,
    preds,
    teacher_outputs,
    tangent_outputs,
    pos_noise,
    sigma_per_atom,
    batch_ligand,
    batch_bond,
    step,
): ...
```

函数根据 `sigma_per_atom`、`sct_time_parameterization` 和
`sct_tangent_norm_mode` 处理 normalization。只有 `scale-c` 需要构造
与 JVP 严格一致的 normalization-floor scale；`fixed-c` 传
`None`，`no-norm` 跳过 scale normalization。coordinate residual 使用
`sct_coord_probe_step`，其 raw MSE 除以该 probe step 后再乘
`loss_pos_weight * sct_coord_loss_weight`。categorical branch 继续使用
`sct_discrete_probe_step`，两者不得混用。

返回字典至少包含：

```text
loss_sct
loss_sct_pos
loss_sct_pos_weighted
sct_pos_probe_step
sct_pos_tangent_rms
sct_pos_warmup
loss_sct_atom
loss_sct_atom_weighted
sct_atom_tangent_rms
sct_atom_normalized_tangent_rms
sct_atom_virtual_kl
sct_atom_teacher_entropy
sct_atom_target_entropy
sct_atom_argmax_flip
loss_sct_bond
loss_sct_bond_weighted
...
```

最终：

```python
loss_original = loss
loss_sct = loss_sct_pos_weighted + loss_sct_atom_weighted + loss_sct_bond_weighted
loss = loss_original + loss_sct
```

不要把纯 diagnostic tensor 重复加到总 loss。

### 5.6 修改动态图构图和 safe norm

`torch.func.jvp` 可能无法直接穿过 `torch_geometric.nn.knn_graph`。在
`code/models/uni_transformer.py::_connect_edge()` 中，邻接拓扑应使用
detached primal coordinates：

```python
edge_index = knn_graph(x.detach(), ...)
```

消息传递中的距离仍使用原始 `x`，从而保留坐标 tangent。该处理等价于把
kNN 邻接看作局部分段常数。

所有可能接收零向量的距离函数应使用 safe norm，例如：

```python
def safe_vector_norm(x, dim=-1, keepdim=False, eps=1e-8):
    return torch.sqrt(x.square().sum(dim=dim, keepdim=keepdim) + eps)
```

重点检查：

- transformer edge distance；
- coordinate update 中的距离；
- bond head 的 `[i, i]` 对角距离。

只在 JVP smoke test 证明需要时扩大修改范围，避免无关的模型数值变化。

### 5.7 修改训练循环

在 `code/scripts/train_diffusion.py` 中：

- 将当前 iteration `it` 显式传入 `get_diffusion_loss(..., global_step=it)`；
- resume 后继续使用恢复的 iteration，保证 warmup 不重置；
- console log 加入总 sCT、coordinate sCT、atom sCT 和可选 bond sCT；
- 当前训练代码已经遍历 scalar result 写入 TensorBoard/W&B，应继续复用该
  机制；
- validation 应动态聚合启用的 sCT scalar，scheduler 继续使用完整总 loss。

## 6. 建议实施顺序

按以下顺序提交，便于定位问题和回滚：

1. 配置解析、time/norm mode 参数校验、pure helper 和 unit tests；
2. linear/log-sigma joint JVP compatibility smoke test，不接入总 loss；
3. coordinate sCT 与日志；
4. atom logit-sCT 与日志；
5. 可选 bond logit-sCT；
6. 训练/validation 聚合、实验配置和文档；
7. GPU 短程训练验证。

每一步都应保证 sCT 默认关闭时现有测试和训练路径不变。

## 7. 测试计划

新建 `code/tests/test_sct_loss.py`，至少覆盖以下单元测试。

### 7.1 Continuous tests

- flat batch 的 graph-wise tangent RMS 正确；
- graph 内 atom 数变化不会改变归一化定义；
- student 有梯度，teacher 和 tangent 没有梯度；
- warmup 在第 1 步接近 `1 / warmup_steps`，结束后为 1；
- joint JVP 在解析 toy model 上等于
  `J_x f @ epsilon + partial_sigma f`；
- log-sigma joint JVP 对 coordinate 和所有 categorical outputs 都等于
  `sigma * (J_x f @ epsilon + partial_sigma f)`；
- `sigma < sigma_min` 的 helper 输入使用 `sigma_min` floor；
- `fixed-c` 在 log-sigma 下使用
  `sigma * tangent / (RMS(sigma * tangent) + c)`；
- `scale-c` 在 log-sigma 下同时缩放 tangent RMS 和 additive
  constant，与对应 linear normalized tangent 等价；
- `no-norm` coordinate 直接返回 raw joint JVP tangent，但 raw RMS
  diagnostic 仍正确；
- 非法 normalization mode 会报错；
- coordinate residual 显式乘 `sct_coord_probe_step`，weighted loss 显式
  除以该 probe step；
- coordinate loss reduction 与现有 `loss_pos` 的 graph weighting 一致。

### 7.2 Discrete tests

- virtual teacher 概率非负且 class sum 为 1；
- tangent 加同一个 class-constant shift 后结果不变；
- `fixed-c` 的 categorical additive constant 不按 token 的 `sigma`
  缩放，`scale-c` 则同比缩放；
- `no-norm` categorical tangent 仍去掉 class-constant gauge 分量，
  但不做 RMS 除法；
- zero tangent 时 virtual teacher 等于原 teacher distribution；
- teacher/tangent 无梯度，student logits 有梯度；
- atom loss 先 graph 内平均，再跨 graph 平均；
- bond loss 不因 `N_g^2` token 数量而改变 graph 权重；
- 关闭 `bond_loss` 或 batch 没有 bond target 时，bond SCT 精确为 0。

### 7.3 Integration tests

- SCT 全关闭时不调用 JVP，也不增加额外 forward；
- condition dropout 在 student/teacher 间一致；
- student/teacher 使用同一个 noisy atom/bond state；
- coordinate、atom 和 bond leaves 使用同一个 linear 或 log-sigma JVP
  direction；
- coordinate、atom 和 bond branches 使用同一个
  `sct_tangent_norm_mode`；
- 不存在 `t_target`、第二份 coordinate noise 或第二次 categorical
  corruption；
- 零距离、bond diagonal 和小 `sigma` 下所有 loss/tangent 均 finite；
- 新配置可以加载旧 checkpoint；
- legacy `sct_loss_weight` 可读取，与 `sct_coord_loss_weight` 冲突时
  会报错；
- sampling 输出路径不受训练 loss 配置影响。

## 8. GPU 验证流程

GPU 任务必须 SSH 到 GPU 节点运行，例如 `gpu35` 或 `gpu37`。

先确认环境和显卡：

```bash
ssh gpu37
cd /shared/healthinfolab/phz24002/pocketVE-CM/PocketVE/code
nvidia-smi
```

依次验证：

1. baseline：所有 sCT 关闭；
2. `linear + fixed-c` continuous only；
3. `log + fixed-c` continuous only；
4. `log + scale-c` continuous only；
5. `log + no-norm` continuous only；
6. 对 atom logit-sCT 至少覆盖 `log + fixed-c/scale-c/no-norm`；
7. `log + scale-c` continuous + atom；
8. 若 bond head 可用，再运行 `log + scale-c` continuous + atom + bond。

每个模式先执行一个真实 mini-batch forward/backward，再执行 100 至 500
iteration trial。记录：

- total/original/sCT losses 是否 finite；
- tangent RMS 和 normalized RMS；
- 当前 time parameterization 和 tangent normalization mode；
- gradient norm；
- 每 step 时间；
- peak allocated/reserved GPU memory；
- atom/bond virtual KL、entropy 和 argmax flip；
- 与 baseline 相比的运行开销。

同时对一个小 graph 做 finite-difference sanity check：固定 noise
$\epsilon$、条件 dropout 和邻接拓扑。`linear` 模式比较 JVP 与

\[
\frac{f(x+h\epsilon,\sigma+h)-f(x,\sigma)}{h}.
\]

`log` 模式使用 $q=\log\sigma$ 的扰动，并在
$\sigma>\sigma_{\min}$ 处比较 JVP 与

\[
\frac{
f\!\left(x+\sigma(e^h-1)\epsilon,\,\sigma e^h\right)
-f(x,\sigma)
}{h}.
\]

该检查只用于验证 JVP，不作为训练 loss 的替代实现。

## 9. 验收标准

实现完成需满足：

- 默认配置行为与修改前一致；
- continuous、atom discrete、可选 bond discrete 可以独立启停；
- `linear` 与 `log` 使用同一个全局时间参数化，所有 joint
  JVP outputs 的方向一致；
- coordinate/categorical 共用 `fixed-c`、`scale-c` 或 `no-norm`，
  默认为 `fixed-c`；
- `fixed-c` 保持 additive constant 不变，`scale-c` 将 log-sigma
  normalization floor 按 `max(sigma, sigma_min)` 缩放，`no-norm`
  不做 RMS scale normalization；
- categorical `no-norm` 仍然执行 class-constant gauge projection；
- coordinate 和 categorical probe step 独立，且两种 probe loss 都按
  `1 / probe_step` 归一；
- 所有 teacher 和 tangent 都 stop-gradient；
- student/teacher 严格复用同一个训练状态；
- unit tests 和真实 GPU mini-batch backward 通过；
- 主要 branch 组合在 linear/log-sigma 及三种 normalization mode
  的短程训练中都没有
  NaN/Inf；
- 旧 checkpoint 可加载，采样接口和输出格式不变；
- 日志能够区分 original loss、各 sCT raw loss、weighted loss 和诊断指标。

## 10. 明确不在第一版中实现的内容

- velocity-space coordinate sCT；
- EMA/target network teacher；
- dual-timestep 或 pairwise consistency；
- finite-step trajectory loss；
- representation/REPA loss；
- sampling scheduler 修改；
- charge SCT；
- 用 finite difference 替代训练时 JVP。

这些功能若后续需要，应在 single-time x0/logit-sCT 通过完整验证后独立设计，
不要混入第一版实现。
