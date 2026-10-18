# qwen25vl-agent

一个 **24 小时摄像头监控智能体**：从摄像头读视频、从麦克风读语音、做人脸与声纹识别，用一个
本地小模型（Qwen2.5-VL）理解画面，并把看到和听到的一切写进一个五段式多模态记忆库；
**每小时**再由第二个本地模型（Qwen3.5-4B）把这一小时总结成几句话，用 TTS 念出来。

它不是一个"调用一次就结束"的脚本，而是一个长跑进程：掉帧、摄像头掉线、模型调用失败、
模型输出解析不出来，都只会让**当前这一 tick 降级**，下一 tick 照常继续。24/7 的前提是不因为
一次抖动就退出。

---

## 一、架构：五层

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
                                    │
                                    ▼
┌──────────────────────────────────────────────────────────────────────────┐
│  每小时播报 summary.py + tts.py                                           │
│                                                                          │
│   summary.py  取最近一小时的情景/语音/告警，交给文本模型 Qwen3.5-4B 总结    │
│               结果写回情景记忆（kind=hourly_summary），下一窗口排除它       │
│   tts.py      edge-tts / piper / espeak 合成，pw-play/aplay 播放           │
│               合成与播放都在一个 worker 线程，两秒语音不阻塞 tick 循环       │
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

### 5. 每小时播报：第二个模型 + TTS

每满 `summary.interval_s`（默认 3600s）就跑一次播报，和归档是**两个独立时钟**，免得一次慢归档
把播报时间推歪：

1. `summary.py` 从记忆里取最近一小时的情景、语音、告警和出现的人，渲染成一份紧凑的中文清单
   （不是原始 JSON——小模型从整齐的清单里总结得更稳，整段提示词还压了 4000 字上限）；
2. 这份清单交给**文本模型** `llm.text_url`（默认 `http://127.0.0.1:8082` 上的 Qwen3.5-4B）；
3. 总结写回情景记忆，`kind=hourly_summary`；
4. 如果 `summary.speak` 打开且 TTS 可用，把这句话交给 `tts.py` 念出来。

三个关键设计：

- **排除上一次总结**：下一窗口读记忆时会跳过 `hourly_summary`，否则每小时都在总结上一小时的
  总结，安静的一小时永远读不成"安静"。
- **安静不播报**：窗口里条目少于 `summary.min_items`（默认 1）就直接跳过，连模型都不调——
  每小时播报一遍"没有异常"比不说话更烦。
- **降级不抛异常**：文本模型挂了就退回视觉引擎（它也能写散文）；两个都挂、TTS 挂了、播放器没有，
  都只让这次播报失败，下一小时照常。

TTS 后端按 `auto` 的顺序挑第一个可用的：`piper`（本地、离线、需要 voice `.onnx`）→ `edge`
（`edge-tts`，中文质量最好但**播报时需要联网**）→ `espeak`（离线、机器人腔）→ `spd-say`。
播放器挑第一个可用的 `pw-play`/`paplay`/`ffplay`/`aplay`；`aplay` 只认 WAV，所以 `edge` 产出的
MP3 会先经 `soundfile` 转成 WAV 再播。想用别的后端：`tts.backend: command` 加一个
`command` 模板（含 `{text}` 和 `{out}`）。

---

## 二、安装

```bash
# 1. 人脸模型（两个 ONNX，Git LFS 上游，约 37MB）
scripts/fetch_models.sh

# 2. 安装包（含 ASR/声纹需要的 funasr）
pip install -e '.[asr]'

# 3. 播报用的文本模型权重（GGUF，约 2.7GB）
modelscope download --model unsloth/Qwen3.5-4B-GGUF \
  --local-dir ~/models/Qwen3.5-4B-GGUF Qwen3.5-4B-Q4_K_M.gguf

# 4. TTS 后端（edge-tts 是 pip 包；piper/espeak 是系统二进制）
pip install edge-tts

# 5. 体检
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
[ok  ] summary-llm    http://127.0.0.1:8082 -> Qwen3.5-4B
[ok  ] tts            backend=edge player=pw-play voice=zh-CN-XiaoxiaoNeural
overall: ready
```

`llm`、`camera`、`memory` 是 critical；`face`、`audio-devices`、`asr`、`speaker`、`summary-llm`、`tts`
缺失只是 warn，agent 会降级到它能做的部分（没有 `summary-llm` 就用视觉引擎总结，没有 TTS 就只写
不念）。

### 前置：两个在跑的推理引擎

agent 不自带模型，需要先起两个服务：

**视觉引擎**（每 tick 描述画面）：

```bash
qwen25vl-lite serve --model /path/to/Qwen2.5-VL-3B-Instruct --device cuda \
  --dtype bfloat16 --weight-dtype int4 --host 127.0.0.1 --port 8081
```

地址用 `configs/default.yaml` 的 `llm.url` 或 `--url` 指定。

**文本引擎**（每小时写播报）。权重是 GGUF，直接用本机 llama.cpp 的 `llama-server`：

```bash
scripts/serve_qwen35.sh          # CPU，127.0.0.1:8082
```

为什么是 CPU：8GB 卡已经被视觉模型占了约 6GB，装不下 4B。而播报是**一小时一次、几百 token** 的
活，CPU 上实测约 2.7 tok/s，一次总结约 20-35 秒——完全够，且完全不碰显存。想让它上 GPU 就
`NGLAYERS=99 scripts/serve_qwen35.sh`（前提是显存真的空出来）。

脚本里的 `--reasoning-budget 0` 和 `--chat-template-kwargs '{"enable_thinking":false}'` 是必要的：
Qwen3.5 是**思考型**模型，不关思考时 `content` 会是空的、答案都落在 `reasoning_content` 里，而
token 预算会被思考过程吃光。客户端也做了兜底——`content` 为空时自动回退到 `reasoning_content`
（见 `llm.extract_text`），所以换成别的思考型端点也不会静默记下一条空播报。

地址用 `llm.text_url` / `--text-url` 指定；设成 `null` 就退回用视觉引擎做总结。

---

## 三、快速开始

```bash
# 1. 注册一个人（人脸 + 声纹）。也可以直接用照片：
qwen25vl-agent enroll --name 小明 --image ~/照片.jpg --audio-seconds 5

# 镜头前的人不在、或者 monitor 正占着摄像头时，用已保存的帧注册（无需摄像头）：
qwen25vl-agent enroll --name 小明 --from-frames 40

# 2. 看注册结果
qwen25vl-agent persons

# 3. 短跑一次端到端自检（取帧→人脸→模型→记忆，出报告）
qwen25vl-agent selftest --ticks 6

# 4. 正式 24/7 运行
qwen25vl-agent run

# 5. 另开终端看记忆
qwen25vl-agent mem --what stats
qwen25vl-agent mem --what episodes --limit 20

# 6. 手动跑一次播报（读现有记忆；--say 顺便念出来）
qwen25vl-agent summary --say

# 7. 只测音频通路（合成 + 播放）
qwen25vl-agent speak "测试播报：系统正常。"

# 8. 带一个网页前端跑（监控在后台线程，界面在主线程）
qwen25vl-agent web                     # http://127.0.0.1:8090
qwen25vl-agent web --no-monitor        # 只开界面，不占摄像头（监控已在别处跑时用）
```

---

## 四、命令

| 命令 | 作用 |
| --- | --- |
| `run` | 启动 24/7 监控循环 |
| `web` | 带本地网页前端跑监控（实时事件流 + 记忆浏览 + 操作按钮） |
| `selftest` | 短跑端到端并出报告 |
| `enroll` | 注册一个人的人脸（可选声纹），支持 `--image` 用照片、`--from-frames N` 用已保存的帧（无需摄像头） |
| `voiceprint` | 给一个 wav 提声纹，或两个 wav 比对相似度 |
| `transcribe` | 一次性语音识别（wav 文件或 `--mic N` 秒） |
| `persons` | 列出已注册的人（`--history` 带交互记录） |
| `mem` | 查看记忆：`--what stats/episodes/facts/rules/search/archive` |
| `rules` | 管理程序记忆规则：`list/add/enable/disable/remove` |
| `summary` | 立刻生成一次小时播报（`--window` 指定读多久，`--say` 念出来） |
| `speak` | 合成一句话并播放，用来单独验音频通路 |
| `doctor` | 依赖与就绪检查 |

常用标志：`--config`（YAML 路径）、`--db`（记忆库路径）、`--json`（机器可读输出）、
`--no-audio`（完全关掉音频）、`--asr-backend none`、`--speaker-backend none`。

`run` 额外有 `--interval`、`--describe-interval`、`--max-vision-tokens`、`--no-save-frames`、
`--max-ticks` / `--duration`（测试用）、`--verbose`，以及播报相关的 `--text-url`、`--text-model`、
`--summary-interval`、`--summary-window`、`--no-summary`、`--speak` / `--no-speak`、`--no-tts`、
`--tts-backend`、`--tts-voice`、`--tts-player`。

### Web 前端（`qwen25vl-agent web`）

一个本地网页监控台，**纯标准库**（`http.server` + 单个 `index.html`），不引入任何 web 框架，
和 `qwen25vl-lite` 自带 webui 的做法一致。默认监听 `127.0.0.1:8090`（`configs/default.yaml` 的
`web.host` / `web.port` 可改，也可用 `--host` / `--port`）。

```bash
qwen25vl-agent web                      # 监控 + 界面
qwen25vl-agent web --port 8090 --no-monitor   # 只开界面
```

**为什么监控跑在后台线程**：`Monitor.run()` 是阻塞的，且要独占摄像头和 SQLite 写连接。所以
`web` 把监控放在 daemon 线程、HTTP 服务放在主线程，两者共享同一个 `Monitor` 实例——这样
`/api/events` 才能把 `MonitorEvent` 实时推给浏览器。**代价是 `web` 会独占摄像头，此时不能再
同时跑 `qwen25vl-agent run`**；`--no-monitor` 就是给"监控已在别的进程跑"这种情况留的口子。

界面内容：

- **顶栏状态灯**：监控是否在跑、摄像头、视觉引擎（8081）、文本引擎（8082）、TTS、记忆库行数；
- **左栏实时事件流**：SSE 推送 `scene` / `speech` / `person` / `alert` / `summary` / `error`，带时间戳，按类型配色；
- **右栏标签页**：人物（含人脸/声纹计数）、情景、事实、规则（可直接启停）、统计、搜索；
- **右上最新帧缩略图**：每秒刷新 `/api/latest.jpg`（监控运行时是刚抓的那帧，按需编码；没监控时退回最新的存档帧）；
- **底栏操作**：说话（测试 TTS）、立即播报本小时、启停监控、上传照片注册人脸。

HTTP 接口（都在同一进程内，无鉴权，仅本机）：

| 方法 | 路径 | 作用 |
| --- | --- | --- |
| GET | `/` | 前端页面 |
| GET | `/api/state` | 监控状态、计数、记忆库统计、两个引擎的健康 |
| GET | `/api/events` | SSE 实时事件流（15s 心跳） |
| GET | `/api/persons`、`/api/persons/<id>/history` | 人物与其交互记录 |
| GET | `/api/episodes?limit&kind&since`、`/api/facts`、`/api/rules`、`/api/search?q` | 记忆查询 |
| GET | `/api/latest.jpg`、`/api/frame/<name>` | 最新帧 / 存档帧（只接受裸文件名，防路径穿越） |
| POST | `/api/actions/speak`、`/summary`、`/enroll`、`/rules`、`/monitor` | 动作 |

读查询各自开一个短命 `Memory` 连接：WAL 允许它和监控的写连接并存，连接的 30s timeout 吸收
两者短暂的写重叠。写动作（注册、改规则）是单个短事务 + 一把锁。

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
| `monitor.face_presence_merge_s` | 300.0 | 同一组人脸在窗口内重复出现就延长上一条 `face` episode 的 `t_end`，而不是再写一行（检测器在模糊帧上抖一下不该留下十几条一模一样的"1张未知人脸"） |
| `audio.vad.start_db` | -32.0 | 绝对 dBFS 起说阈值。用绝对值而非相对噪声底，因为绝对值跨房间可预测，相对值会随噪声底估计漂移 |
| `face.match_threshold` | 0.363 | SFace 官方公布的余弦阈值 |
| `speaker.threshold` | 0.65 | 声纹余弦阈值 |
| `identity.mode` | prefer_face | 人脸/声纹融合策略 |
| `memory.archive_max_mb` | 2048 | 归档大小预算上限 |
| `llm.text_url` | http://127.0.0.1:8082 | 播报用的文本模型端点；设 `null` 就退回用视觉引擎总结 |
| `summary.interval_s` | 3600.0 | 多久播报一次（与归档时钟独立） |
| `summary.retry_s` | 300.0 | 播报**失败**后多久重试。失败不该白等一整个小时：失败时把播报时钟回拨到 5 分钟后，而不是直接推进一小时 |
| `summary.window_s` | 3600.0 | 每次读多久的记忆 |
| `summary.min_items` | 1 | 窗口低于这个条数就跳过，不播报"没有异常" |
| `summary.speak` | true | 是否把总结念出来 |
| `tts.backend` | auto | 按 piper → edge → espeak → spd-say 挑第一个可用的；也可 `command`/`none` |
| `tts.voice` | zh-CN-XiaoxiaoNeural | edge 音色名（`backend: piper` 时这里填 voice `.onnx` 路径） |
| `tts.player` | auto | 按 pw-play → paplay → ffplay → aplay 挑第一个可用的 |
| `tts.max_chars` | 400 | 单次播报字数上限，避免一小时的内容变成一小时音频 |
| `web.host` | 127.0.0.1 | 前端绑定地址；默认只回环，因为它能注册人物和触发播报 |
| `web.port` | 8090 | 前端端口，避开视觉 8081 / 文本 8082 |

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
- **播报用第二个模型，且跑 CPU**：总结是"文本进、文本出"，不需要视觉；把它放在 CPU 上，就
  不必和占满显存的视觉模型抢资源。一小时一次、几百 token 的活，CPU 的 2.7 tok/s 够用。
- **播报结果写回记忆但被下一窗口排除**：这样"每小时说了什么"有据可查，又不会让总结滚成
  总结的总结。
- **TTS 走系统播放器而不是 python 音频库**：机器上只有 PipeWire 的 `pw-play` 和 ALSA 的 `aplay`，
  没有 sounddevice/pyaudio；用现成的播放器意味着不需要为放一句话再装一层音频栈。

---

## 七、已知限制

- **单摄像头、单麦克风**：`camera.device` / `audio.device` 各一个。因此 `qwen25vl-agent web`
  会独占摄像头，**不能**和 `qwen25vl-agent run` 同时跑；要么用 `web` 一个命令代替 `run`，要么
  用 `web --no-monitor` 只开界面。
- **Web 前端无鉴权**：接口能注册人物、改规则、触发播报。默认只绑回环（`web.host`），
  不要随手改成 `0.0.0.0` 暴露到局域网。
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
- **播报依赖两个外部服务**：文本模型（默认 8082）挂了会退回视觉引擎；两个都挂，这一小时的
  播报就丢了——但记忆照写，事后可以用 `summary --window N` 补跑。TTS 挂了只是不念，总结照样
  写进记忆。
- **`auto` 会选到需要联网的 edge-tts**：机器上没装 piper 时，`auto` 挑的是 `edge`，播报时
  需要访问微软的在线 TTS；网络不通就不出声。要完全离线，装 `piper` 并把 `tts.voice` 指向本地
  voice `.onnx`。

---

## 八、开发与测试

测试是**自足**的：每个用例都把数据库、归档和告警日志指向 pytest 的临时目录，音频默认关闭，
摄像头、人脸模型和推理服务要么被 fake 替换、要么跳过——不需要硬件，也不需要两个引擎在跑。

```bash
pip install -e '.[dev]'
pytest
```

CI（`.github/workflows/ci.yml`）在 Python 3.10 / 3.12 上跑同样的路径：安装（`.[dev]`）、
`compileall`、`pytest`，再加一条 CLI 冒烟。webui 的 `index.html` 由 `pyproject.toml` 的
`package-data` 随包发布，安装后不需要额外的静态文件路径。

---

## 九、许可

Apache-2.0（见 [LICENSE](LICENSE)）。推理引擎见
[qwen25vl-lite](https://github.com/zhangjianbang-nb/qwen25vl-lite)。
