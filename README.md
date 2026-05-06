# 2D 材料元素替换筛选工具集

一套用于 VASP 声子计算的元素替换预筛选 + 作业自动化工具。

## 文件结构

```
element_screener/
├── replace.py           # 启发式筛选器（本机运行，零计算成本）
├── server_pipeline.py  # 服务器流水线（VASP + phonopy 自动化）
└── README.md           # 本文件
```

## 方案 A: 启发式筛选 (replace.py)

基于离子半径、电负性、氧化态兼容性对候选替换元素评分排序。

### 用法

```bash
# 替换特定元素的所有位点
python replace.py POSCAR --element V --candidates Nb,Ta,Mo,W,Cr

# 替换特定位点（1-indexed）
python replace.py POSCAR --site 2 --candidates all

# 使用内置分组
python replace.py POSCAR --element S --candidates chalcogen

# 指定氧化态（推荐，提高准确性）
python replace.py POSCAR --element V --candidates all --ox-state 4

# 输出候选 POSCAR 文件 + JSON 结果
python replace.py POSCAR --element Cl --candidates F,Br,I --ox-state -1 \
    --output-dir ./candidates --json results.json

# 排除特定元素、只看 Top 10
python replace.py POSCAR --element V --candidates all --exclude Tc,W \
    --ox-state 4 --top 10
```

### 内置候选分组

| 名称 | 元素 |
|------|------|
| `all` | 全部常见元素 (75+) |
| `tm` | 过渡金属 |
| `main` | 主族元素 |
| `chalcogen` | 氧族: O, S, Se, Te |
| `pnictogen` | 氮族: N, P, As, Sb, Bi |
| `halogen` | 卤素: F, Cl, Br, I |
| `alkali` | 碱金属 |
| `alkaline` | 碱土金属 |
| `early-tm` | 前过渡金属 |
| `late-tm` | 后过渡金属 |

### 评分维度

总分 = w₁×半径 + w₂×电负性 + w₃×氧化态 + w₄×结构

默认权重: `[0.4, 0.3, 0.2, 0.1]`

可通过 `--weights 0.3 0.3 0.3 0.1` 自定义。

评分说明（每个维度 0-1，越高越好）：
- **半径**: 离子半径越接近 → 分数越高。使用 Shannon 半径，缺省时回退到原子半径
- **电负性**: Pauling 电负性差异越小 → 分数越高
- **氧化态**: 与目标氧化态匹配（相同=1.0，±1以内=0.8，±2=0.5）
- **结构**: 周期表位置越近（同族>同行>邻近）→ 分数越高

### 重要限制

启发式筛选只考虑**几何和化学兼容性**，不预测声子稳定性。
高分 ≠ 声子稳定。例如 S→Se 几何上完美匹配（0.922），
但实际 DFT 声子可能不稳定。

**正确用法**: 先用启发式筛掉明显不合适的选项（如离子半径差 >30%），
再对 Top N 候选进行 DFT 验证。

---

## 方案 B: ML 势筛选（预留框架）

> 需要服务器安装 MACE/CHGNet。如果将来配置了 ML 势环境，
> 可在此处集成以下流程：
>
> 1. pymatgen 读取 POSCAR + 枚举替换
> 2. MACE-MP-0 / CHGNet 快速弛豫
> 3. phonopy 有限位移法计算声子
> 4. 检查虚频 → 排序输出
>
> 典型耗时: 数十秒/体系（vs. DFT 数小时），准确率 80-90%。

---

## 方案 C: 服务器流水线 (server_pipeline.py)

从启发式筛选结果出发，自动生成完整的 VASP 弛豫 + phonopy 声子计算作业。

### 典型工作流

```bash
# ===== 第一步：本机启发式筛选 =====
python replace.py POSCAR --element V --candidates Nb,Ta,Mo,W \
    --ox-state 4 --output-dir ./candidates

# ===== 第二步：生成候选的 VASP 作业 =====
# -r 指向已有参考弛豫目录（含 POTCAR）
python server_pipeline.py setup ./candidates -r ./ref_opt -o ./vasp_jobs

# ===== 第三步：上传到服务器，提交弛豫 =====
# 在服务器上：
cd vasp_jobs/*/relax
qsub run_relax.pbs

# ===== 第四步：弛豫完成后生成声子作业 =====
python server_pipeline.py phonon-setup ./vasp_jobs --species "S V Nb"

# ===== 第五步：提交声子计算 =====
cd vasp_jobs/*/phonon
qsub run_phonon.pbs

# ===== 第六步：分析结果 =====
python server_pipeline.py analyze ./vasp_jobs --json results.json
```

### 批量生成替换 POSCAR

```bash
# 从单个 POSCAR 批量生成所有替换组合
python server_pipeline.py bulk POSCAR \
    -p "V@Nb,Ta" "Cl@F,Br,I" \
    -o ./all_candidates
```

### analyze 输出示例

```
=================================================================
  Phonon Stability Analysis
=================================================================
  Scanned  : 3  |  STABLE: 1  |  ARTIFACT: 1  |  UNSTABLE: 1

  Gamma-point imaginary modes = correctable artifact (hiphive ASR /
  rotational sum rule). Only non-Gamma imaginary = real instability.

  System                     Min(THz)   Non-G     #Imag    Status
-----------------------------------------------------------------
  pho551                      -6.70      -6.70    684      UNSTABLE
  VSF-phonon1                 -6.39      --          8    STABLE(Gamma-artifact)
  VSCl                         0.50       --          0    STABLE
```

### Gamma 点伪影识别

判断标准（基于实际实验经验）：
- **仅 Gamma 点 (q=0) 有虚频** → 声学求和规则 / 旋转求和规则伪影，可安全忽略，无论虚频大小或数量
- **任何非 Gamma 点 (q≠0) 有虚频** → 真实声子不稳定性，该替换方案不可用
- 示例：VSF 在 Gamma 点有 -6.39 THz 虚频，但实际验证稳定；pho551 在非 Gamma 点有虚频，属于真实不稳定

---

## 声子不稳定的常见原因及对策

| 原因 | 表现 | 对策 |
|------|------|------|
| **尺寸失配** | 高频虚频、结构扭曲 | 用启发式筛选避免半径差异 >20% |
| **电荷失配** | 虚频 + 电子不收敛 | 匹配氧化态；考虑额外掺杂补偿 |
| **费米面嵌套** | 特定 q 点虚频（如 $(\pi,\pi)$） | 考虑磁性、自旋极化、U 值 |
| **Jahn-Teller** | 局域结构畸变 | 检查对称性破缺可能性 |
| **虚声子模** | 单个软模 | 增加超胞大小、检查 ISYM 设置 |

---

## 建议的完整工作流

```
                ┌─────────────┐
                │  基础 POSCAR │
                └──────┬──────┘
                       ↓
                ┌─────────────┐
                │启发式筛选(replace.py)│
                │  筛掉明显不合适   │
                └──────┬──────┘
                       ↓
                ┌─────────────┐
                │  DFT 弛豫    │
                │  ISIF=3     │
                └──────┬──────┘
                       ↓
                ┌─────────────┐
                │  phonopy    │
                │  声子计算    │
                └──────┬──────┘
                       ↓
                ┌─────────────┐
                │ 检查虚频    │←── 有虚频 → 弃用此方案
                │  (analyze)  │
                └──────┬──────┘
                       ↓ 无虚频
                ┌─────────────┐
                │  ✓ 稳定方案  │
                │  继续计算性质 │
                └─────────────┘
```

---

## 依赖

```bash
pip install pymatgen
```

服务器端还需要: VASP, phonopy。
