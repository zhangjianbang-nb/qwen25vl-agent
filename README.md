# qwen25vl-agent

一个 **24 小时摄像头监控智能体**：从摄像头读视频、从麦克风读语音、做人脸与声纹识别，用一个
本地小模型（Qwen2.5-VL）理解画面，并把看到和听到的一切写进一个五段式多模态记忆库。

它不是一个"调用一次就结束"的脚本，而是一个长跑进程：掉帧、摄像头掉线、模型调用失败、
模型输出解析不出来，都只会让**当前这一 tick 降级**，下一 tick 照常继续。24/7 的前提是不因为
一次抖动就退出。

---

## 一、架构：四层

```
┌──────────────────────────────────────────────────────────────────────────┐
│  感知 perception                                                          │
│                                                                          │
│   video.py     V4L2 取帧，排空缓冲（避免"描述的是过去"），掉线自动重连     │
│   face.py      YuNet 检测 + SFace 128 维人脸特征（cv2.dnn，纯 CPU）        │
│   audio.py     arecord 连续采集 16k S16_LE                                │
│   vad.py       能量 VAD，把连续流切成"有人在说话"的片段                    │
│   asr.py       SenseVoice 转写，顺带给出语言/情绪/事件标签                  │
│   speaker.py   CAM++ 192 维声纹（说话人特征）                              │
└──────────────────────────────────────────────────────────────────────────┘
                                    │
                                    ▼
┌──────────────────────────────────────────────────────────────────────────┐
│  智能体 agent + llm                                                       │
│                                                                          │
│   不自己加载模型，HTTP 调用一个已经在跑的 qwen25vl-lite serve             │
│   提示词要求小模型只吐一个固定 schema 的 JSON：                            │
│   {scene, people[], events[], remember[], notes[]}                       │
│   解析是防御式的：模型把 JSON 包在散文/代码块里也能救回来，失败只降级不抛异常│
└──────────────────────────────────────────────────────────────────────────┘
                                    │
                                    ▼
┌──────────────────────────────────────────────────────────────────────────┐
│  记忆 memory（五类，一个 SQLite 文件，WAL 模式）                           │
│                                                                          │
│   人物记忆 person.py   人脸特征 + 声纹双模态，融合判定"这是谁"             │
│   工作记忆 working.py  最近几分钟的感知，进程内环形缓冲，重启即失           │
│   情景记忆 episodic.py "发生了什么"，带帧路径，保留 30 天                   │
│   语义记忆 semantic.py 持久事实 (subject, predicate, object)，去重 upsert   │
│   程序记忆 procedural.py 监控规则（触发器 → 动作），用户配置，运行时不变   │
└──────────────────────────────────────────────────────────────────────────┘
                                    │
                                    ▼
┌──────────────────────────────────────────────────────────────────────────┐
│  监控 monitor.py                                                          │
│                                                                          │
│   双线程：音频线程做 VAD→ASR→声纹；主线程做取帧→人脸→融合→推理→写记忆     │
│   →规则评估→归档。所有 SQLite 写都在主线程（连接不跨线程共享）             │
│   每小时一次归档：老数据落 JSONL，再按大小预算裁剪，24/7 不会撑爆磁盘       │
└──────────────────────────────────────────────────────────────────────────┘
```

### 1. 感知：视频 + 语音 + 人脸

三路并行，各自可独立缺省：

- **视频**：`cv2.VideoCapture` 封装，但关键在**排空缓冲**。30fps 摄像头配 0.5s 循环，
  不排空就会拿到几百毫秒前的帧，把"实时监控"变成"描述过去"。所以每次 `retrieve` 前先
  `grab` 两次，并把 `CAP_PROP_BUFFERSIZE` 设为 1。摄像头掉线时按指数退避重开，返回 `None`
  而不抛异常。
- **人脸**：YuNet（"这里有没有脸"，便宜，每 tick 都跑）+ SFace（"这是谁的脸"，128 维特征）。
  两个 ONNX 走 `cv2.dnn`，**不需要 onnxruntime / insightface**，且跑在 CPU 上——因为 8GB 显存
  已经被服务的 Qwen2.5-VL 占满了。小于 48px 的脸不做特征（不可信）。
- **语音**：`arecord` 直接读 raw PCM（机器上没有 sounddevice/pyaudio）。能量 VAD 按绝对 dBFS
  阈值切句，只在真的有人说话时才调 ASR——这是 24/7 不烧 CPU 转写环境噪声的关键。ASR 用
  SenseVoice，它**顺带报告听到的语言**，所以"语言识别"不需要第二个模型。声纹用 CAM++。

### 2. 智能体：本地小模型

**不自己加载模型。** 在 8GB 卡上，一个已经在跑的 Qwen2.5-VL 就占约 5.9GB，再加载第二份会 OOM，
复用服务还省掉约 20s 的模型加载。所以 agent 只通过 OpenAI 兼容接口 HTTP 调用
`qwen25vl-lite serve`。

提示词是按 3B 模型的能力写的，不是按"聪明"写的：记忆上下文只给一个小切片，输出契约是单个
JSON 对象、字段固定、每个字段都可选、空数组是明确合法答案。解析器会找第一个配平的 `{}`
（跳过字符串里的花括号），也能从代码块里提取；解析失败就把整段回复当 scene，绝不抛异常出循环。

### 3. 记忆：人物 + 工作 + 语义 + 情景 + 程序

**人物记忆**存两套独立生物特征库：SFace 128 维人脸 + CAM++ 192 维声纹，身份由两者融合决定。
为什么两套都要：人脸要求人可见且朝向镜头，声纹要求人说话——一天下来只依赖任一种都会大面积
失效，两种融合能让"谁在场"在更多时间里可用。融合模式由 `identity.mode` 控制：

| mode | 含义 |
| --- | --- |
| `prefer_face` | 有人脸特征时用人脸，否则退回声纹（默认） |
| `any` | 任一模态命中即可 |
| `both` | 要求两者一致（最严，适合两者都稳定存在时） |

**工作记忆**是进程内的 TTL 环形缓冲，只装最近几分钟（默认 40 条 / 300s），刻意不持久化——
重启后活下来的是情景/语义/人物记忆，工作记忆只是让"刚刚发生了什么"不必每 tick 查库的草稿纸。

**情景记忆**是"发生了什么"的摘要（"某人在书桌前坐下"），与逐 tick 的原始观测分开存、分开淘汰：
情景小而值得留 30 天，原始日志只对最近几天排障有用（留 7 天）。每条情景可指向产生它的帧文件，
所以数据库里的摘要永远能和摄像头实际看到的画面对照。

**语义记忆**是持久事实，按 `(subject, predicate, object)` 唯一键 upsert——同一件事听到两次是
提升置信度和新鲜度，而不是追加重复行。置信度显式存储（用户陈述 0.9，模型单帧猜测 0.4），
提示词可以据此加权。

**程序记忆**是"这个家怎么被监控"：一条规则 = 触发器 + 动作，存 JSON，新增触发类型不用改表。
**智能体不会在运行时自己发明规则**——它只评估用户配置好的规则，这让 24/7 监控可预测：
模型负责描述和总结，判断何时告警是确定性代码。

支持的触发器：`unknown_face`、`known_person`、`speech_keyword`、`person_absent`、`no_speech_for`。
任意触发器可带 `when` 门（如 `{"after":"22:00","before":"06:00"}`，跨午夜已处理）。
动作：`alert`（记录/外发）、`remember`（写语义事实）、`snapshot`（标记本 tick 帧值得保留）。
消息模板可用 `{time}` `{name}` `{text}` `{count}` `{rule}`。

### 4. 监控与记忆存档：24 小时

主循环双线程：**音频线程**（采集→VAD→ASR→声纹，通过队列交给主线程）和**主线程**（取帧→人脸→
身份融合→智能体调用→写记忆→规则评估→归档）。语音处理是突发且 CPU 重的，不能拖住 tick 循环。

`describe_interval`（默认 12s）控制多久调一次视觉模型（一帧约 0.6s），`describe_on_event`
允许新出现的未知人脸/语音立刻触发一次描述，`min_describe_gap`（2s）是硬下限。

**归档**每小时跑一次，幂等：
- 超过 `episode_retention_days`（30 天）的情景、超过 `observation_retention_days`（7 天）的
  原始观测，按天写进 `episodes-YYYY-MM-DD.jsonl` / `observations-YYYY-MM-DD.jsonl`，再从库里删；
- 帧文件按 `frame_retention_days`（3 天）删；
- 最后按 `archive_max_mb`（2048MB）预算删最老的归档文件，保证 24/7 跑不满磁盘。

JSONL 是刻意选的格式：追加式、一行一条、能直接用 `grep` 读。一年后这个包没了，归档还能用
`grep` 打开。

---

## 二、安装

```bash
# 1. 人脸模型（两个 ONNX，Git LFS 上游，约 37MB）
scripts/fetch_models.sh

# 2. 安装包（含 ASR/声纹需要的 funasr）
pip install -e '.[asr]'

# 3. 体检
qwen25vl-agent doctor
```

`doctor` 会逐项检查并给出可操作的结论：

```
[ok  ] llm            http://127.0.0.1:8081 -> Qwen2.5-VL-3B-Instruct
[ok  ] camera         /dev/video0 640x480
[ok  ] face           YuNet + SFace loaded
[ok  ] audio-devices  plughw:1,0, plughw:1,6
[ok  ] asr            funasr present
[ok  ] speaker        funasr present
[ok  ] disk           1553.8 GB free for ~/.qwen25vl-agent/memory.db
[ok  ] memory         1 people, 2 episodes, 1 facts
overall: ready
```

`llm`、`camera`、`memory` 是 critical；`face`、`audio-devices`、`asr`、`speaker` 缺失只是 warn，
agent 会降级到它能做的部分。

### 前置：一个在跑的推理引擎

agent 不自带模型，需要先起 `qwen25vl-lite serve`：

```bash
qwen25vl-lite serve --model /path/to/Qwen2.5-VL-3B-Instruct --device cuda \
  --dtype bfloat16 --weight-dtype int4 --host 127.0.0.1 --port 8081
```

地址用 `configs/default.yaml` 的 `llm.url` 或 `--url` 指定。

---

## 三、快速开始

```bash
# 1. 注册一个人（人脸 + 声纹）。也可以直接用照片：
qwen25vl-agent enroll --name 小明 --image ~/照片.jpg --audio-seconds 5

# 2. 看注册结果
qwen25vl-agent persons

# 3. 短跑一次端到端自检（取帧→人脸→模型→记忆，出报告）
qwen25vl-agent selftest --ticks 6

# 4. 正式 24/7 运行
qwen25vl-agent run

# 5. 另开终端看记忆
qwen25vl-agent mem --what stats
qwen25vl-agent mem --what episodes --limit 20
```

---

## 四、命令

| 命令 | 作用 |
| --- | --- |
| `run` | 启动 24/7 监控循环 |
| `selftest` | 短跑端到端并出报告 |
| `enroll` | 注册一个人的人脸（可选声纹），支持 `--image` 用照片 |
| `voiceprint` | 给一个 wav 提声纹，或两个 wav 比对相似度 |
| `transcribe` | 一次性语音识别（wav 文件或 `--mic N` 秒） |
| `persons` | 列出已注册的人（`--history` 带交互记录） |
| `mem` | 查看记忆：`--what stats/episodes/facts/rules/search/archive` |
| `rules` | 管理程序记忆规则：`list/add/enable/disable/remove` |
| `doctor` | 依赖与就绪检查 |

常用标志：`--config`（YAML 路径）、`--db`（记忆库路径）、`--json`（机器可读输出）、
`--no-audio`（完全关掉音频）、`--asr-backend none`、`--speaker-backend none`。

`run` 额外有 `--interval`、`--describe-interval`、`--max-vision-tokens`、`--no-save-frames`、
`--max-ticks` / `--duration`（测试用）、`--verbose`。

---

## 五、配置

全部默认值在 `configs/default.yaml`，每项都能被 CLI 覆盖。配置是**强校验**的：拼错的键会在
加载时报错，而不是悄悄回退到默认值——一个拼错的 `describe_intervall` 应该是错误，否则就是一个
以错误节奏运行的 agent。

几个值得知道的旋钮：

| 键 | 默认 | 说明 |
| --- | --- | --- |
| `llm.max_vision_tokens` | 128 | **请求体**字段，真正决定图像 token 预算；服务端 `--vision-tier` 只驱动 FastV 剪枝，不改像素 |
| `monitor.tick_interval` | 0.5 | 感知节奏（人脸检测 + 工作记忆） |
| `monitor.describe_interval` | 12.0 | 多久调一次视觉模型（一帧约 0.6s） |
| `audio.vad.start_db` | -32.0 | 绝对 dBFS 起说阈值。用绝对值而非相对噪声底，因为绝对值跨房间可预测，相对值会随噪声底估计漂移 |
| `face.match_threshold` | 0.363 | SFace 官方公布的余弦阈值 |
| `speaker.threshold` | 0.65 | 声纹余弦阈值 |
| `identity.mode` | prefer_face | 人脸/声纹融合策略 |
| `memory.archive_max_mb` | 2048 | 归档大小预算上限 |

---

## 六、设计取舍（为什么这么做）

- **不自己加载模型**：8GB 卡放不下第二份 Qwen2.5-VL。复用 HTTP 服务，还省 20s 加载。
- **人脸/语音全在 CPU**：显存已被服务的模型占满，任何要自己 GPU runtime 的东西都得抢显存。
  代价是 640x480 一帧几十毫秒，可以接受。
- **不用向量数据库**：家庭规模的生物特征库是几十行，numpy 点积精确、无依赖、每 tick 都够快。
- **不用 WebRTC/Silero VAD**：能量 VAD 无模型无依赖，职责很窄——把连续流切成句，让 ASR 只在
  有人说话时被调用。
- **归档用 JSONL 不用压实**：归档步就是一条 SQL `DELETE` 加一次 JSONL 追加，不是压缩作业。
- **SQLite + WAL**：24/7 写者加一个读者（CLI 在 agent 运行时查记忆）正是 WAL 的场景，读不阻塞写。
- **模型不发明规则**：判断何时告警是确定性代码，模型只描述和总结，24/7 才可预测。

---

## 七、已知限制

- **单摄像头、单麦克风**：`camera.device` / `audio.device` 各一个。
- **ASR/声纹首次调用要下模型**：SenseVoice 约 940MB + CAM++，走 modelscope（国内网络比
  huggingface 可靠得多）。首次约 10s 加载，所以是**懒加载**，不在 import 时加载——24/7 agent
  重启不该在能看摄像头之前先付 10s。
- **身份识别是余弦阈值判定**，不是聚类；未注册的人一律是 `unknown`，这是刻意的一等结果，
  因为未知人脸往往正是程序规则盯着的东西。
- **声纹只对真实语音可靠**：VAD 是按响度（能量）切句的，不判断"这是不是人声"。所以音乐、
  电视、装修噪声等足够响的非语音片段也会被送进 CAM++，而声纹模型在非语音输入上给出的
  192 维向量没有说话人含义——实测两个纯音（220Hz / 880Hz）的相似度高达 0.7252，超过 0.65
  阈值被判为"同一说话人"。也就是说**语音维度的身份判定可能被环境音误触发**；如果这个场景
  重要，需要换成会区分人声/非人声的 VAD（如 Silero），代价是引入模型依赖。
- **归档是删除式迁移**：超期的行写进 JSONL 后从库里删掉，CLI 的 `mem` 只看库，看历史要去
  `archive_dir` 里 `grep`。
- **原始观测日志量很大**：每个 tick 都会写一行 `frame` 观测，默认 0.5s 一个 tick 就是约
  17 万行/天。保留期是 7 天，所以数据库约 120 万行；超期后由归档搬到 JSONL。磁盘总体有界
  （`archive_max_mb` 只限制归档目录，数据库由保留期限制），但如果想进一步降量，把
  `monitor.tick_interval` 调大即可。

---

## 八、许可

Apache-2.0。推理引擎见 [qwen25vl-lite](https://github.com/zhangjianbang-nb/qwen25vl-lite)。
