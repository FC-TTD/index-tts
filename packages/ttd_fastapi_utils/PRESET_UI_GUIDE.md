# Preset And Preset UI Split Guide

这份说明专门给实现者，帮助把 `Preset` 本体和 `Preset UI` 的职责拆清楚，避免业务调音时把“标准扫尾”和“风格设计”混在一起。

## 总原则

- `Standard Postprocess` 是基础扫尾链路，适合 TTS / ASR / 语音生成结果统一收口。
- `Style Preset` 是独立音色链路，适合做电话音、智能助手、心声独白、广播、对讲机等风格化效果。
- 不要让 preset 内部偷偷自动带上标准链。业务代码应显式决定两条链路是否串联。

## 推荐职责拆分

### 1. 标准链

标准链建议继续放在 [`postprocess.py`](./src/ttd_fastapi_utils/postprocess.py)：

- `trim_silence(...)`
- `loudnorm(...)`
- `eq(...)`
- `apply_postprocess(...)`

适用目标：

- 清理首尾静音
- 统一响度
- 给 TTS 类结果做轻量一致化扫尾

### 2. Preset

Preset 建议继续放在 [`preset.py`](./src/ttd_fastapi_utils/preset.py)：

- 只组合风格相关原语
- 只暴露真正影响风格的参数
- 不隐式调用 `apply_postprocess(...)`

常见风格原语：

- `bandpass(...)`
- `lowpass(...)`
- `highpass(...)`
- `saturate(...)`
- `delay(...)`
- `delay_tail(...)`
- `reverb(...)`
- `mix(...)`
- `limiter(...)`

## 推荐业务调用方式

```python
from ttd_fastapi_utils import postprocess, preset

# 1) 标准扫尾：适合 TTS 结果统一收口
wav = postprocess.apply_postprocess(
    wav,
    sr,
    target_loudness=-23.0,
    enable=True,
    trim_silence=False,
    enable_eq=True,
)

# 2) 风格化：按需单独叠加
wav = preset.apply_preset("smart_assistant", wav, sr)
```

如果业务默认已经统一应用标准链，那么 preset 就应保持纯风格逻辑，不要再在 preset 里重复做标准扫尾。

## 推荐 UI 设计方式

调试页建议明确拆成两个区块：

1. `Standard Postprocess / 标准后处理`
2. `Style Preset Controls / 预设风格参数`

不要把这些控件混成一块：

- `Trim Silence / 静音裁切`
- `Enable EQ / 启用 EQ`
- `Target LUFS`
- `Enable Saturation / 启用饱和`
- `Enable Delay / 启用 Delay`
- `Enable Reverb / 启用 Reverb`

原因是业务会自然把它们理解成“同一套 preset 参数”，但实际上前者是基础扫尾，后者是风格塑形。

## 推荐元数据结构

建议每个 preset 提供稳定的元数据，供 UI 和文档复用：

```python
{
    "display_name": "smart_assistant 模拟智能语音",
    "summary": "bandpass + light saturation + delay tail",
    "primary_controls": [...],
    "secondary_controls": [...],
    "recommended_standard_postprocess": {
        "enable": False,
        "target_loudness": -23.0,
        "trim_silence": False,
        "enable_eq": True,
    },
    "implementation_note": "...",
}
```

这样 UI 可以直接复用：

- 展示 preset 说明
- 标记主要参数
- 回填推荐标准链默认值

而不是在前端或调试页里另写一套知识。

## 日志建议

建议日志明确拆开记录：

- `standard_chain`
- `preset`
- `preset_style_params`
- `effective_params`

其中 `effective_params` 最有价值，因为它记录的是“这次真正参与计算的参数”，后续回写 preset 默认值时最稳。

## 维护建议

- 预设命名尽量表达风格，而不是表达业务流程。
- `preset.py` 只承载音色设计，不承载业务默认扫尾。
- 如果某个 preset 需要业务先做标准扫尾，用元数据或 README 说明，不要在实现里隐式执行。
- 调试页里与 preset 无关的控件，应明确标注“仅在 Custom 模式下生效”。
