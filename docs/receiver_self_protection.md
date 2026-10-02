# 吸热器自算降额保护设计说明

本文档对应代码：`cspplant/receiver/self_params.py`、`estimator.py`、
`policy.py`、`protection.py`、`replay.py`、`persist.py`。控制台接口见
`GET /api/self-params`、`GET /api/protection/verify`、
`POST /api/receiver/recalibrate`（仿真漂移注入：`POST /api/receiver/degrade`）。

## 1. 要解决的问题

原来的降额保护按固定热容与固定散热系数计算，流量计读数也默认无漂移。
长期运行后管屏结焦（等效热容变化）、保温老化（散热变强）、流量计漂移
（指示流量高估真实冷却能力），同样工况下保护有时提前动作、有时晚一步。

## 2. 自算什么

用三个现场可测量做能量平衡在线辨识：

- 实际入射热功率 `P`（镜场按已落盘辐照与对焦面积估算）
- 指示流量 `ṁ`（流量计读数）
- 出口温升 `dT/dt`（用传感器**滤波前**的原始温度，滤波会引入相位滞后、
  等效虚增热容）

模型：

```
C·dT/dt = P − g·cp·ṁ·(T−T_in) − UA·(T−T_amb)
```

辨识三个参数：等效热容 `C`、散热系数 `UA`、流量计增益 `g`
（真实流量 = 指示流量 × g）。算法为带遗忘因子的递推最小二乘
（RLS），回归量按标定点归一化。

**可辨识性按维度独立判定**：流量项与散热项都正比于温差，流量恒定时
二者严格共线，此时只采纳热容，g、UA 保持标定，绝不沿共线方向瞎猜。
激励不足、残差过大或系数跑出物理包络时 `fit_ok=False`，不得据此改保护。

## 3. 只往保守走（单向棘轮）

自算生效值相对标定只能：

| 参数 | 采纳方向 | 物理后果 |
| --- | --- | --- |
| 热容 C | 取更小 | 升温更快、缓冲更小 |
| 散热 UA | 取更大 | 净吸热更多 |
| 流量增益 g | 取更小 | 可信流量更少、冷却更弱 |

实现要点：

- `pessimistic()` 先把原始估计按标定夹一遍（不允许越过标定变宽松）；
- 再做保守向 EMA 平滑 + 连续 30 步确认，防止噪声来回刷版本；
- 每次收紧 `revision += 1`，单调递增、持久化、跨重启保留；
- **放松的唯一入口是检修后重新标定**（`recalibrate`），必须带操作人
  落 `self_params_recalibrated` 审计，同样升 revision。

## 4. 据此调整降额起点与上限

每个受热步在当前生效参数下外推净温升速率 `rate`，求不动点：

```
T_enter* = clamp(T_target, T_derate − rate(T_enter*)·lead, T_derate)
```

- 起点只能比标定降额线 `T_derate` 更早（温度更低），不会推迟；
- 出力上限取"自算锚点曲线"与"标定锚点曲线"中更保守的一条；
- 回差、跳闸上限 `T_max` 不变。

## 5. "这次比上次早/晚凭什么"

每次进入降额，判定记录里带结构化 `explanation`：

- `temperature_margin_c` / `earlier_vs_calibration_s`：相对纯标定提前
  多少度、多少秒（只可能 ≥ 0）；
- `factors`：热容、散热、流量增益三个因子各自的自算值 vs 标定值；
- `vs_previous`：相对上一次进入事件的提前量之差（earlier/later/same）、
- 两次的 revision 是否变化；
- 审计正文同步给出一句可读中文。

## 6. 自算与标定冲突时按部位定（来源定死）

策略表在 `policy.py` 里随代码冻结，带指纹 `POLICY_FINGERPRINT`：

| 部位 | 来源 | 理由 |
| --- | --- | --- |
| `panel` 受热管屏 | `self_pessimum`（自算悲观值） | 有独立功率/温升观测，可验证 |
| `header` 集箱 | `cal_only`（只按标定） | 无独立热工观测，自算不可辨识 |
| `support` 支撑 | `cal_only` | 按机械温度标定保护 |

不允许运行时切换；每条判定记录都带策略指纹。

## 7. 版本、迁移与一致性

- `schema`：自算参数**结构版本**。v1 只有 C/UA，v2 增加 flow_gain。
  改版必须在 `self_params.MIGRATIONS` 登记迁移函数；读盘时自动迁移、
  落审计 `self_params_migrated`。遇到未知（含更新的）schema：
  **fail-closed** 退回纯标定并报警，不猜测解释。
- `revision`：生效值的**代际**，收紧/重标定各加一，只能单调递增。
- `ENGINE_VERSION`：保护判定引擎版本（起点公式、上限合成、回差、
  来源仲裁）。判定记录自带引擎版本、schema、revision、策略指纹。

**实时保护、回放、审计共用同一套纯函数与版本**：
`replay.verify_protection_records()` 从审计流水按记录里的版本复算
active / ceiling / 起点，核对：

- 未知引擎 / 未知 schema → `unknown_engine` / `unknown_schema`；
- 策略指纹不一致 → `policy_drift`；
- 复算结果与落盘值不符 → `replay_mismatch`；
- revision 链条倒退 → `revision_chain_broken`。

## 8. 展示与审计

- 控制台：`GET /api/self-params` 返回标定值、生效值、原始估计、
  各维可辨识标志、版本、部位策略、进入降额历史；
- 审计事件：`self_params_committed`（收紧，warning）、
  `self_params_recalibrated`（重标定）、`self_params_migrated`、
  `self_params_load_failed`（alarm）、`derate_entered` /
  `derate_cleared`（带完整 decision 戳与生效参数快照）；
- 持久化：`receiver/self_params` 原子 JSON 文档。
