# 项目说明（给 Claude Code 的上下文）

## 我是谁、目标是什么

我是外汇新手，在澳洲，长期目标是逐步建立一套「研究 → 验证 → 模拟执行」的
AUD/USD 交易系统。我在边做边学，**请用中文解释每一步做了什么、怎么验证。**

**一次只推进一步，每步单独验证通过再进下一步。** 我明确要求一次做多步时照做，
但每一层写完单独验一次，不许攒到最后一起跑。

**这个文件直接改，不用问。** 改完在回复里说一句改了哪里就行。

---

## 项目现状（2026-08 通读代码后的评审结论）

AlphaFX 已经是一个结构良好的 AUD/USD 宏观因子**研究平台**
（约 2000 行核心代码 + 完整测试 + CI）。已具备：

| 层 | 位置 | 职责 |
|---|---|---|
| 数据 | `alphafx/data/` | yfinance（AUD/USD、DXY、VIX）+ FRED + RBA provider，SQLite 持久化 |
| 信号 | `alphafx/signals.py` | `QuantSignalAgent` 宏观因子打分，概率经历史命中率校准 |
| 验证 | `alphafx/walk_forward.py`、`diagnostics.py` | 走向前样本外验证、前向收益诊断 |
| 风控建议 | `alphafx/risk.py` | `RiskAgent`：波动率分级、极端波动 NO TRADE、杠杆上限 5、`MIN_CONFIDENCE=0.52` |
| 纸面交易 | `alphafx/trade/` | `PaperBroker`，20 天时间屏障退出；`scripts/paper_trade.py` |
| 展示 | `alphafx/dashboard/`、`app.py` | Streamlit 仪表盘 + LLM 解释/反方/裁判 agents（**只解释不决策**） |
| 无头运行 | `scripts/run_signal.py` | 输出结构化 JSON（signal / probability / action / stop_loss / take_profit / factors） |
| **执行（新）** | `alphafx/execution/` | IG Demo REST 客户端 + 执行侧硬风控 + 信号→执行桥 |

**关键事实：每日自动纸面交易目前是暂停的**（`.github/workflows/daily.yml`），
原因写在注释里：AUD/EUR/CHF 组合十年回测净负（Sharpe −0.42，−23%），
当前信号没有稳健的样本外优势。**这个闸门是本项目最有价值的纪律，必须保留。**

---

## 核心原则（已在代码中体现，不可退化）

1. **量化层拥有信号**；ML/LLM 层只对比、解释、质疑，永远不能推翻信号
2. **无未来数据**：宏观因子按发布时点滞后、扩展窗口校准、ML 只用样本外预测
3. **没有实现历史校准证据的信号（fallback 先验）不允许开仓**
   （`RiskAgent.EVIDENCE_SOURCES` 机制，这个设计很好）
4. **回测/验证先于纸面执行，纸面执行先于任何真实下单**
5. **信号质量闸门**：walk-forward 显示正期望之前，自动执行保持关闭

---

## 硬性约束（即使我要求修改，也请先提醒我再动）

- **代码和程序输出里不许出现中文**，一律英文。适用范围：
  - 源码的注释、docstring、变量名
  - **所有字符串字面量**，包括异常消息、错误码对照表、日志、`print` 出来的东西
  - 测试里的断言文案（`pytest.raises(match=...)` 也算）

  **例外只有两个**：**本文件**（CLAUDE.md，是给我看的交接说明）和**对话**，用中文。
  `README.md` / `ROADMAP.md` / `DESIGN.md` / `docs/` 本来就是英文，**保持英文**。

  由 CI 的 `lint` job 自动拦截，本地自查：
  `grep -rlP '[\x{4e00}-\x{9fff}]' --include="*.py" .` 应该没有输出。
- `daily.yml` 的自动交易**保持暂停**，直到走完下面路线图的第 4 步
- **执行层永远锁定 Demo 环境**；接真实环境不在本蓝图范围内
- 风控规则**只能收紧不能放宽**；`MIN_CONFIDENCE`、`EVIDENCE_SOURCES` 门槛不能绕过
- **LLM 相关代码不得进入任何交易决策路径**（只做解释/复盘/红队）
- `.env` 不入库、不打印；**不要让我在对话里粘贴任何凭证**
- 策略/信号改动必须留档：改了什么、为什么、样本外回测对比（git commit + tag），
  **禁止亏损后临场调参**
- 保持「每步完成后项目可运行 + 测试通过」（沿用 ROADMAP 的增量原则）

---

## 执行层：IG Demo 整合

### 目标结构

| 文件 | 职责 | 状态 |
|---|---|---|
| `alphafx/execution/ig_client.py` | IG REST 封装（认证 / 行情 / 持仓 / 开仓 / 平仓） | ✅ A.1 完成 |
| `alphafx/execution/risk_engine.py` | 执行侧硬风控（确定性规则，**永不智能化**） | 🔴 A.2 待做 |
| `alphafx/execution/bridge.py` | 信号→执行桥：读 `data/latest_signal.json`，经 risk_engine 校验后转 IG 订单 | 🔴 A.3 待做 |
| `scripts/execute_demo.py` | 入口，**默认 dry-run**，`--live` 才真实提交到 Demo | 🔴 A.3 待做 |

### 执行侧硬风控（`risk_engine.py`）

在 `RiskAgent` 的「**建议**」之上，执行侧再加一层**强制校验**：

- `BASE_URL` 写死 `https://demo-api.ig.com/gateway/deal`，禁止改为真实环境
- 下单必须带服务器端止损（`stopDistance`），缺失直接拒绝
- 单笔风险 ≤ 账户 1%（按止损距离反推手数），单笔手数上限 1.0
- 月度熔断：当月亏损 5% → 本月不再开新仓
- 账户熔断：净值从高点回撤 15% → 全面停止，等待人工复盘
- `RiskAgent` 返回 `NO TRADE` 时，执行层**无条件服从**
- 重大数据（RBA/美联储决议、CPI、非农）前后 2 小时不开新仓

> **重复检查是故意的，不是冗余。** `risk_engine.py` 是上层闸门，
> `ig_client.open_position()` 里的止损/手数检查是最后一道防线——
> 哪怕以后有人绕过 risk_engine 直接调 client，那两条也拦得住。

---

## ⚠️ IG API 踩过的坑（实测得来，别踩第二遍）

### 1. 认证必须用 v3 OAuth，不能用 v2

> **⚠️ 原交接文档在这一条上是错的。** 它写「`POST /session` 用 Version 2 头，
> token 在响应头里」——那是 v2 的 `CST` + `X-SECURITY-TOKEN` 老认证。
> 这个账户已迁移到 IG 新平台，走 v2 会直接被拒，返回
> `error.security.account-migrated`，**和凭证对不对无关**。

v3 的做法：

- 认证头是 `Authorization: Bearer <token>` + `IG-ACCOUNT-ID`
- **`access_token` 只有 60 秒寿命**，`_ensure_token()` 在每次请求前自动续期
- 所有业务请求走 `_request()` 统一入口（自动续 token + 统一查错）

### 2. 每个接口的 Version 号不一样，不能统一

改错会 404 且返回 HTML 而不是 JSON。已知正确的：

| 接口 | Version | 备注 |
|---|---|---|
| `POST /session` | **3** | OAuth |
| `POST /session/refresh-token` | 1 | 续期 |
| `GET /markets/{epic}` | 3 | |
| `GET /accounts` | 1 | 余额/净值，risk_engine 反推手数要用 |
| `GET /prices/{epic}/{res}/{num}` | **2** | v3 改成查询参数式 URL，路径式在 v3 下 404 |
| `GET /positions` | 2 | |
| `POST /positions/otc`（开仓） | 2 | |
| `GET /confirms/{ref}` | 1 | |
| `POST /positions/otc`（平仓） | 1 | 加 `_method: DELETE` 头 |

### 3. 下单是异步的

`POST /positions/otc` 只返回 `dealReference`（受理号），**不代表成交**。
必须再 `GET /confirms/{ref}` 查 `dealStatus`，可能是 `REJECTED`。
`open_position()` 已经把这两步绑在一起了。

### 4. 历史价格接口有每周配额 —— 所以本项目不用它

Demo 约 10,000 个数据点/周，**1 根 K 线 = 1 个数据点**。
真正糟蹋配额的不是正常运行，是调试时反复重跑。

**AlphaFX 的行情数据继续走现有 yfinance/SQLite 管线，IG 只用于执行和实时报价。**
因此 `ig_client.py` **刻意不提供 `get_history()`** —— 没有这个方法就不可能烧配额。

### 5. 其他

- AUD/USD 迷你合约 epic `CS.D.AUDUSD.MINI.IP`，1 手 = 10,000 AUD，约每 pip 1 美元
- 周末休市，`marketStatus` 非 `TRADEABLE` 时下单会被拒
- 出错时看 `ig_client.py` 里的 `ERROR_HINTS` 表，把 IG 的错误码翻成人话
  （按上面的纪律，表里的文案是英文）
- 登录用的**不是**平台用户名：IG 的 Web API demo 账户有独立凭证，
  在 My IG → Settings → Web API 里单独设置。填错会报 `error.security.invalid-details`，
  看起来像密码错，其实是用户名根本不对

### 6. ⚠️ Demo 账户余额是假的一亿，「1% 风险」规则会失效

2026-08-09 实测：本项目用的 Demo 账户余额是**九位数（约 1 亿 AUD）**，
不是真实账户会有的规模。所以「单笔风险 ≤ 账户 1%」≈ 100 万 AUD，
按 30 点止损反推出来的手数是天文数字，**这条规则在 Demo 上等于没有约束**。
真正拦得住的是 `MAX_SIZE = 1.0` 手的硬上限。

> 本仓库是 **public**。账户号、余额明细、持仓价位这类东西不要写进文档、
> 测试数据或提交信息里——测试里的账户号一律用 `TESTACC1` 这种明显是假的值。

A.2 写 `risk_engine.py` 时必须处理这一点：用一个**写死的名义账户规模**
（例如 10,000 AUD）来算 1% 风险，而不是直接读 IG 返回的余额。
否则等哪天接真实账户，这条规则从没被真正测试过。

### 7. 账上有一笔不归脚本管的手动持仓

`CS.D.AUDUSD.CFD.IP`（**标准**合约）上有 1.0 手 BUY，是手动开的。
脚本用的是 `CS.D.AUDUSD.MINI.IP`（**迷你**合约），两者 epic 不同。
风控按 epic 匹配，不会把它当成「已有持仓」而拦下 MINI 的单——这是对的，
但复盘统计 PnL 时要记得把它排除掉。

---

## 环境

**一律用 `.venv/bin/...`，不要用裸的 `python3`。** 裸 `python3` 是系统里那个
3.9，代码在它上面跑不起来（见下）。

```bash
cd ~/Documents/AlphaFX

# 测试：不联网、不用凭证、不耗配额，改完代码先跑这个
.venv/bin/python -m pytest
.venv/bin/python -m pytest tests/test_execution.py -v    # 只跑执行层

# lint：规则在 ruff.toml，跟 CI 跑的完全一致
.venv/bin/ruff check .
.venv/bin/ruff check . --fix                             # 能自动修的直接修

# 研究/展示
.venv/bin/streamlit run app.py                  # 仪表盘
.venv/bin/python scripts/run_signal.py --json   # 无头信号，输出 JSON
.venv/bin/python scripts/paper_trade.py         # 纸面交易

# 执行层（Demo）
.venv/bin/python -m alphafx.execution.ig_client   # 连通性自检：登录 + 行情 + 持仓，不下单
```

### Python 环境（2026-08-09 重建）

**三个地方的版本必须一致，改一个就要改另外两个**：`.venv`、`ruff.toml` 的
`target-version`、CI 的 `python-version`。现在都是 **3.12**。
（之前本机 3.9 / CI 3.12 的落差已经害过一次：CI 绿灯，本机 `zip(..., strict=)` 崩。）

重建 venv：

```bash
uv venv --python 3.12
uv pip install -r requirements.txt -r requirements-ml.txt
```

两个环境上的坑：

- **Homebrew 是坏的**，任何 `brew` 命令都直接崩：
  `unknown or unsupported macOS version: "26.5.2"`。它是 2020 年装的，太老了。
  **所以装 Python 用 `uv`，别去修 brew**（要修的话得重跑官方安装脚本，属于另一件事）。
- 系统里的 `/usr/local/bin/python3`（3.9.0）是 **x86_64 构建，跑在 Rosetta 转译下**；
  `.venv` 里的 3.12 是原生 arm64。所以 venv 不只是版本新，也快。

- 凭证在 `.env`（已 gitignore）：`IG_API_KEY` / `IG_USERNAME` / `IG_PASSWORD`
- `alphafx/config.py` 的 `load_local_env()` 负责加载，**不依赖 python-dotenv**
- 凭证在 `IGClient.__init__` 里读，不在模块级 —— 所以没有 `.env` 也能
  `import`，离线测试和回测不会被凭证绊住

---

## 开发路线（按顺序，每步单独验证）

### A. 执行层并入（不依赖信号质量，可立即开始）

1. ✅ 迁入 `ig_client.py` 到 `alphafx/execution/`，补单元测试（mock HTTP，24 条）
   —— 2026-08-09 已对真实 Demo 账户跑通登录 / 行情 / 持仓 / 账户余额（未下单）
2. 🔴 实现 `risk_engine.py`（含熔断状态持久化到 SQLite），补测试
3. 🔴 实现 `bridge.py` + `scripts/execute_demo.py`，dry-run 模式：
   每次运行记录「若执行会下什么单」到日志/数据库，与纸面交易并行对照

### B. 信号质量攻坚（决定闸门何时开，继续 ROADMAP V2 方向）

4. 🔴 按 `ROADMAP.md` 推进研究升级。**通过标准（三条同时满足才算过关）**：
   - walk-forward 样本外为正期望（**含点差成本后**）
   - 多窗口回测不再像当前组合那样净负
   - 纸面/dry-run 实录 ≥ 3 个月且与回测偏差在合理范围

> **未过关前，凡「要不要打开自动执行」的问题，答案一律是「还不行」，请如实提醒我。**

### C. 闸门开启后（需我明确确认）

5. 🔴 重新启用 `daily.yml` 的 cron（仍是 Demo），加每日运行摘要日志
6. 🔴 每周复盘物料自动生成：交易记录、熔断事件、实盘 vs 回测偏差，
   供我拿去做 AI 复盘分析

---

## 风格要求

- 代码简单直白，优先可读性；沿用仓库现有代码风格和测试习惯
- 每完成一步，**中文说明：改了什么文件、怎么验证、下一步是什么**
- 不确定的地方明说「这个我没法替你验证」，不要假装跑通了

---

## 交接来源

执行层代码来自另一个项目 `~/projects/ig-demo-bot`（GitHub `lvyongyu/ig-demo-bot`，private）。
注意该仓库已在 commit `61b054c` 移除全部代码转入产品设计阶段，
**原始 `ig_client.py` 保留在 `340bed4`**，需要对照时：

```bash
git -C ~/projects/ig-demo-bot show 340bed4:ig_client.py
```

该仓库的 `docs/` 里有完整的 IG API 中文教程（认证、取数、下单、平仓），值得参考。
