# CLAUDE.md — AlphaFX 项目交接与开发蓝图

> 给未来的 Claude Code 会话与维护者：本文件是本仓库的**总蓝图与操作纪律**，
> 每次会话开始请通读一遍。用户是外汇新手，在澳洲，边做边学——所有解释请用
> **中文**，每完成一步都要说明「改了什么文件、怎么验证、下一步是什么」。

---

## 1. 人与目标

我是外汇新手，在澳洲。长期目标是逐步建立一套「**研究 → 验证 → 模拟执行**」
的 AUD/USD 交易系统，在边做边学。请用中文解释每一步做了什么、怎么验证。

## 2. 项目现状（2026-08 通读代码后的评审结论）

AlphaFX 已是一个结构良好的 AUD/USD 宏观因子研究平台（约 2000 行核心代码 +
完整测试 + CI）。已具备：

- **数据层**：yfinance（AUD/USD、DXY、VIX）+ FRED + RBA provider，SQLite 持久化。
- **信号层**：`QuantSignalAgent`（`alphafx/signals.py`）宏观因子打分；概率经历史
  命中率校准（`probability_source` 标记 `historical_calibration` /
  `walkforward_calibration` / `fallback_score_map`）。
- **验证层**：walk-forward 样本外验证、前向收益诊断（`alphafx/diagnostics.py`）、
  因子 IC / 滚动 IC。
- **风控建议层**：`RiskAgent`（`alphafx/risk.py`）——波动率分级、极端波动
  `NO TRADE`、推荐杠杆上限 5x。
- **纸面交易**：`PaperBroker`（`alphafx/trade/`）；`scripts/paper_trade.py`。
  仓位名义 = `base_units × size_factor × leverage`（杠杆已接入，受风控上限约束）。
- **展示层**：Streamlit 仪表盘 + LLM 解释/反方/裁判 agents（`alphafx/llm/`，只解释
  不决策）。
- **无头运行**：`scripts/run_signal.py` 输出结构化 JSON（signal / probability /
  action / stop_loss / take_profit / factors）。

> ⚠️ 关于「信号质量闸门」：评审的原意是——在 walk-forward 显示（含成本后的）
> 正期望之前，自动执行必须保持关闭。**这条纪律必须保留**。但当前代码状态与
> 蓝图描述存在若干差异，见 **第 7 节「现状核对」**，务必先读。

## 3. 核心原则（不可退化）

1. **量化层拥有信号**；ML / LLM 层只对比、解释、质疑，**永远不能推翻信号**。
2. **无未来数据**：宏观因子按发布时点滞后、扩展窗口校准、ML 只用样本外预测。
3. **没有实现历史校准证据的信号（fallback 先验）不允许开仓。**
   （目标纪律——当前 `RiskAgent` 尚未按 `probability_source` 拦截，见第 7 节，
   须在执行层 `risk_engine.py` 强制实现。）
4. **回测/验证先于纸面执行，纸面执行先于任何真实下单。**
5. **信号质量闸门**：walk-forward 显示正期望之前，自动执行保持关闭。

## 4. 硬性约束（即使用户要求修改，也请先提醒再动）

- `daily.yml` 的自动交易**保持暂停**，直到走完第 6 节路线图的第 4 步。
- 执行层**永远锁定 Demo 环境**；接真实环境不在本蓝图范围内。
- **风控规则只能收紧不能放宽**；证据门槛 / 拦截机制不能被绕过。
- **LLM 相关代码不得进入任何交易决策路径**（只做解释 / 复盘 / 红队）。
- `.env` **不入库、不打印**；不要让用户在对话里粘贴任何凭证。
- 策略/信号改动**必须留档**：改了什么、为什么、样本外回测对比（git commit +
  tag）；**禁止亏损后临场调参**。
- 保持「每步完成后项目可运行 + 测试通过」（沿用 ROADMAP 的增量原则）。

## 5. 本阶段任务：并入 IG Demo 执行层

有一个最小 IG REST API 客户端（来自 `ig-demo-bot` 项目的 `ig_client.py`，纯
`requests` 实现：登录 CST / X-SECURITY-TOKEN、行情、持仓、市价开仓强制止损、
`/confirms/` 确认、平仓）。任务是把它作为**执行层**并入本仓库。

> 注：`ig_client.py` 源文件在另一个仓库 `ig-demo-bot`，**当前会话尚未包含**，
> 迁入前需要先把该文件提供进来。

### 目标结构

- `alphafx/execution/ig_client.py` — IG REST 封装（从 `ig-demo-bot` 迁入）。
- `alphafx/execution/risk_engine.py` — 执行侧硬风控（见下）。
- `alphafx/execution/bridge.py` — 信号→执行桥：读 `run_signal.py` 的 JSON 输出
  （或 `data/latest_signal.json`），经 `risk_engine` 校验后转为 IG 订单。
- `scripts/execute_demo.py` — 入口，**默认 dry-run**，`--live` 才真实提交到 Demo。

### 执行侧硬风控（`risk_engine.py`，确定性规则，永不智能化）

在 `RiskAgent` 的「建议」之上，执行侧再加一层强制校验：

- `BASE_URL` 写死 `https://demo-api.ig.com/gateway/deal`，**禁止**改为真实环境。
- 下单**必须带服务器端止损**（`stopDistance`），缺失直接拒绝。
- **单笔风险 ≤ 账户 1%**（按止损距离反推手数），**单笔手数上限 1.0**。
- **月度熔断**：当月亏损 5% → 本月不再开新仓。
- **账户熔断**：净值从高点回撤 15% → 全面停止，等待人工复盘。
- `RiskAgent` 返回 `NO TRADE` 时，执行层**无条件服从**。
- 重大数据（RBA / 美联储决议、CPI、非农）**前后 2 小时不开新仓**。
- （新增落实第 3 条原则）`probability_source == fallback_score_map` 的信号
  **不允许开仓**——执行侧必须拦截。

### IG API 技术要点（已踩过的坑）

- 认证 `POST /session` 用 **Version 2** 头，token 在**响应头**里
  （CST、X-SECURITY-TOKEN）。
- 下单**异步**：`POST /positions/otc` 只返回 `dealReference`，必须
  `GET /confirms/{ref}` 查 `dealStatus`（可能 `REJECTED`）。
- AUD/USD 迷你合约 epic：`CS.D.AUDUSD.MINI.IP`，**1 手 = 10,000 AUD**。
- 历史价格接口有**每周配额**；行情数据继续走现有 yfinance / SQLite 管线，
  IG 只用于**执行和实时报价**。
- 周末 `marketStatus` 非 `TRADEABLE`，下单会被拒。

## 6. 开发路线（按顺序，每步单独验证）

### A. 执行层并入（不依赖信号质量，可立即开始）

1. 迁入 `ig_client.py` 到 `alphafx/execution/`，补单元测试（**mock HTTP**）。
2. 实现 `risk_engine.py`（含熔断状态**持久化到 SQLite**），补测试。
3. 实现 `bridge.py` + `scripts/execute_demo.py`，**dry-run 模式**：每次运行记录
   「若执行会下什么单」到日志/数据库，与纸面交易并行对照。

### B. 信号质量攻坚（决定闸门何时开，继续 ROADMAP V2 方向）

4. 按 ROADMAP 推进研究升级；**通过标准（三条同时满足才算过关）**：
   - walk-forward 样本外为正期望（**含点差成本后**）。
   - 多窗口回测不再像当前组合那样净负。
   - 纸面 / dry-run 实录 **≥ 3 个月**且与回测偏差在合理范围。

   > 未过关前，凡「要不要打开自动执行」的问题，答案一律是「**还不行**」，
   > 请如实提醒用户。

### C. 闸门开启后（需用户明确确认）

6. 重新启用 `daily.yml` 的 cron（仍是 Demo），加每日运行摘要日志。
7. 每周复盘物料自动生成：交易记录、熔断事件、实盘 vs 回测偏差，供用户做
   AI 复盘分析。

## 7. 现状核对：蓝图与当前代码的差异（先补齐，勿当作已实现）

> 2026-08 逐文件核对结果。蓝图第 2/3 节的部分描述**与当前代码不一致**，
> 记录在此以免误导。这些是「待补齐」项，不是「已完成」项。

1. **`daily.yml` 当前并未暂停。** cron `0 22 * * 1-5` 处于**启用**状态，且没有
   写任何「因回测净负而暂停」的注释。蓝图（第 4 节约束 1）要求它暂停——**这是
   当前未满足的硬约束**，需尽快处理（注释掉 cron，仅保留 `workflow_dispatch`
   手动触发，并写明原因）。
2. **`RiskAgent` 没有 `EVIDENCE_SOURCES` 机制，也没有 `MIN_CONFIDENCE=0.52`
   常量。** 实际开仓门槛是 `probability >= 0.60`（方向置信度，多空同一阈值）；
   `0.52` 只是 `|score|=1` 时的 fallback 概率值，不是门槛。
3. **fallback 先验目前允许开仓。** `RiskAgent.suggest` 只看 `probability`，不看
   `probability_source`；一个 `fallback_score_map` 且概率 ≥0.60 的信号当前会
   产生 BUY/SELL（测试亦如此断言）。第 3 节原则「fallback 不得开仓」**尚未在
   代码中强制**——须在 `execution/risk_engine.py`（并考虑在 `RiskAgent`）落实。
4. **`PaperBroker` 没有「20 天时间屏障退出」。** 它只按止损/止盈退出
   （`alphafx/trade/paper.py::_exit_reason`）。20 天持有期是**回测**
   （`BacktestAgent.run(holding_period=20)`）的参数，不是纸面经纪的行为。

## 8. 风格要求

- 代码简单直白，优先可读性；沿用仓库现有代码风格和测试习惯。
- 每完成一步，用中文说明：**改了什么文件、怎么验证、下一步是什么**。
