# CanITrustU-Jev

Jev 类决策模型的六版本分类评测。仓库仅包含数据读取、模型适配、评测代码及配置、测试；数据、权重、密钥、调研资料、生成代码和运行结果保留在本地。

## 数据与读取

最终集为 2,000 道原题，每题原始版及 5 个扰动版，共 12,000 次评测。10 个来源覆盖工具调用、网页操作、轨迹判断、软件修复、长记忆、安全决策等场景。每条记录附带 `dataset`、`scenario`、`decision_type`、`tags`、`source`，通过 `base_id` 关联六版本。

版本依次为原始、选项逆序、选项 ID 替换、状态 JSON 格式化、无关上下文插入、指令释义。V4 正文已逐题由 gpt-5.6-luna 生成，无模板回退。运行时直接读取已准备好的版本，不重新生成扰动。

数据需另行放到本地，仓库不下载或提供数据。支持：

- `data/final/versions.jsonl`，同目录可放冻结数据的 `manifest.json`。
- `data/release/`，包含 `manifest.json` 和无损 gzip JSONL 分片，默认输入路径。清单格式为 `jev-release/gzip-jsonl-v1`，包含 `shards`（每片 path/records/sha256）、`original_samples`、`version_instances`、`stream_sha256`。

```python
from src.dataset.reader import iter_records
for record in iter_records('data/final/versions.jsonl'):
    print(record['base_id'], record['version'], record['gold'])
```

读取器流式解压，不需要将全部数据载入内存。分片哈希在评测前校验，完整遍历后校验解压内容哈希与记录数。旧 JSONL 有清单时也校验哈希。模型请求仅发送 state、instructions 与候选描述，不发送 gold、标签或来源注释。

## 运行

Python 3.9+，mock 和 HTTP 适配器仅用标准库。从仓库根目录运行：

```bash
python3 -m unittest discover -s tests/runtime -v
python3 -m src.eval --model mock --input data/final/versions.jsonl --limit-bases 10

export JEV_API_KEY='你的密钥'
python3 -m src.eval --model jev --input data/final/versions.jsonl --output results/jev-run-001
# 使用本地压缩分片：
python3 -m src.eval --model jev --input data/release --output results/jev-run-002
```

`python3 -m src.eval.final` 是等价入口。省略 `--limit-bases` 表示全量；每道入选题始终评测六版本。真实 API 调用可能计费；mock 仅用于验证链路。输出目录不能重复使用。

`configs/models.json` 配置截图中的 10 个目标。Jev 默认使用 `https://api.typesafe.ai/v1/systemone`，请求 `jev-1.13.0`，可通过 `JEV_ENDPOINT` 覆盖端点。其他 HTTP 目标需设置配置中指定的端点变量。SemIf/so1 原生适配器需自行安装其库与权重；kev 0.6B 通过用户提供的 JSON stdin/stdout bridge 接入。配置列出目标不代表其服务可用或已完成真实模型验证。

长轨迹可能超出某些模型上下文窗口；读取器不会截断。候选超限、调用或响应解析失败均记为错误并计入准确率分母，进程最终非零退出。当前入口为串行运行，不支持断点续跑。

## 指标与输出

每次运行产生 `run.json`（配置与输入/代码哈希）、`predictions.jsonl`（逐条预测）、`metrics.json`：

- 分版本准确率，以及按来源、场景和决策类型分组统计。
- 整体准确率：正确实例数 /（原题数 × 6）。
- 全版本正确率：六个版本全部正确的原题数 / 原题数。

缺失版本及失败调用不能算对。另提供概率校准、延迟与扰动前后配对统计；成本未测量，不复现截图排行榜的综合分。模拟结果不能代表模型能力。

## 目录

`src/dataset/`：统一 schema、读取与完整性校验；`src/model/`：模型适配及注册；`src/eval/`：评测入口和指标；`tests/runtime/`：不依赖真实数据、权重或网络的测试。
