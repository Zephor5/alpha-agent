# 通用 Bash Tool 落地方案

## 状态

Proposed，待实现。

## 目标

为 Alpha Agent 增加一个通用的 `bash` tool，让模型可以在受控边界内执行本地 shell 命令，用于构建、测试、包管理、Git 操作、诊断脚本和必要的系统命令。

这个工具的核心目标不是“把 `subprocess.run` 暴露给模型”，而是建立一条可审计、可取消、可限权、可控输出大小的命令执行通道，并和现有同步 LLM/tool loop、daemon turn guard、SQLite trace 体系保持一致。

## 参考结论

本方案参考两个现有实现的设计取舍，但不照搬内部结构：

- Claude Code `BashTool`：重点吸收严格参数 schema、命令语义解释、输出截断/持久化、权限检查、前后台任务边界。
- Hermes `terminal_tool.py`：重点吸收 backend 抽象、本地 process group kill、workdir 校验、环境变量清洗、长任务提示和输出治理。

不纳入第一阶段：

- 多云 sandbox/backend，例如 Modal、Vercel Sandbox、SSH、Daytona、Singularity。
- sudo 密码缓存和交互式提权。
- UI 进度组件。
- 宽泛 plugin marketplace 或第三方 tool gateway。

## 当前项目约束

相关现状：

- `src/alpha_agent/tools/base.py` 定义最小 `Tool` 协议，目前只有 `run(arguments)`。
- `src/alpha_agent/runtime/tools.py` 的 `ToolExecutor` 在工具调用前后记录 `tool.started`、`tool.completed`、`tool.failed`。
- `AlphaAgent.respond()` 的 tool loop 是同步、有界、provider-neutral 的 OpenAI-compatible tool call 流程。
- session 级取消目前是 cooperative cancel，只在 LLM/tool 边界检查，不能中断一个正在阻塞的工具调用。
- 默认工具注册位于 `src/alpha_agent/tools/default.py`，当前只在 Tavily key 存在时注册 `web_search`。
- 配置由 `src/alpha_agent/config.py`、`config.example.toml`、README 共同维护。

因此 bash tool 必须从全局角度处理以下问题：

- 执行过程必须能响应 `/stop` 或 session cancel。
- 工具结果不能无限写入 `session_messages` 或下一轮 LLM context。
- `tool.started` 不能无条件保存完整命令参数，避免泄漏 secret 或大型 heredoc。
- 默认不能启用本地命令执行，必须显式配置开启。
- 命令失败不等于工具失败；非零退出码应作为结构化结果返回。

## 非目标

- 不实现远程容器或云 sandbox。
- 不实现交互式 shell 会话。
- 不实现后台进程管理的完整版本，除非同时实现 process registry。
- 不新增独立 agent framework。
- 不保证兼容历史数据库内容；按项目规则直接面向目标架构调整。

## 总体架构

新增三层：

1. Tool 契约层：`BashTool` 暴露模型可调用的 schema，做参数校验和结果格式化。
2. Shell 执行层：`ShellBackend` 抽象具体执行机制，v1 只实现 `LocalShellBackend`。
3. Policy/治理层：统一处理配置、workdir 范围、危险命令、取消、timeout、输出截断、ANSI 清理、secret redaction、退出码语义。

建议文件结构：

```text
src/alpha_agent/tools/bash.py
src/alpha_agent/tools/shell/__init__.py
src/alpha_agent/tools/shell/backend.py
src/alpha_agent/tools/shell/local.py
src/alpha_agent/tools/shell/policy.py
src/alpha_agent/tools/shell/output.py
src/alpha_agent/tools/shell/semantics.py
tests/test_bash_tool.py
```

## Tool 接口调整

当前 `Tool.run(arguments)` 不足以支持可取消 bash。建议引入执行上下文：

```python
@dataclass(frozen=True)
class ToolExecutionContext:
    session_id: str
    tool_call_id: str | None
    output_dir: Path
    check_canceled: Callable[[str], None]
```

将协议调整为：

```python
class Tool(Protocol):
    name: str
    description: str

    def run(
        self,
        arguments: dict[str, Any],
        context: ToolExecutionContext,
    ) -> ToolResult:
        ...
```

同步更新现有工具实现，不保留旧式 `run(arguments)` 兼容分支：

- `ToolExecutor` 总是创建 `ToolExecutionContext`。
- `TavilyWebSearchTool.run()` 一次性调整为新签名；不使用 context 的工具可以忽略该参数。
- 新增 `trace_arguments(arguments)` 可选方法，用于工具自定义 `tool.started` 记录内容。

`BashTool.trace_arguments()` 必须截断或摘要化 `command`，不把完整大型 heredoc 或明显 secret 写入 trace metadata。

## Bash Tool 模型契约

工具名：`bash`

描述原则：让模型知道 bash 适用于构建、测试、包管理、Git、脚本、进程诊断；不鼓励用它读写文件，因为未来可以提供更专门的 file/search/patch tool。

参数 schema：

```json
{
  "type": "object",
  "properties": {
    "command": {
      "type": "string",
      "description": "要执行的 shell 命令。"
    },
    "description": {
      "type": "string",
      "description": "简短说明命令目的，用于 trace 和可读日志。"
    },
    "workdir": {
      "type": "string",
      "description": "命令工作目录，必须位于允许的工作区内。默认使用配置的 default_workdir。"
    },
    "timeout_seconds": {
      "type": "integer",
      "minimum": 1,
      "description": "前台命令超时时间，受 max_timeout_seconds 限制。"
    }
  },
  "required": ["command"],
  "additionalProperties": false
}
```

第一阶段不向模型暴露 `background`。原因是没有 process registry 时，后台命令会变成不可观测副作用。模型如使用 `nohup`、`disown`、`setsid` 或尾随 `&`，v1 应直接拒绝并提示未来使用受管后台任务。

## 返回契约

`ToolResult.output` 使用 JSON object，不用纯文本：

```json
{
  "status": "completed",
  "exit_code": 0,
  "stdout": "...",
  "stderr": "...",
  "elapsed_ms": 1234,
  "workdir": ".",
  "truncated": false,
  "omitted_chars": 0,
  "return_code_interpretation": null
}
```

状态枚举：

- `completed`：命令执行完成，包括非零退出码。
- `timeout`：超过 timeout，被工具杀掉。
- `canceled`：session cancel 触发，被工具杀掉。
- `blocked`：policy 拒绝执行。
- `error`：工具内部错误，例如 shell 不存在、workdir 无法解析。

工具异常只用于实现层故障，不用于表达命令退出码。这样 provider 后续轮次可以读取 stdout/stderr/exit_code 自行修正命令。

## 配置设计

新增配置段：

```toml
[tools.bash]
enabled = false
default_workdir = "."
allowed_workdirs = ["."]
default_timeout_seconds = 120
max_timeout_seconds = 600
max_output_chars = 30000
env_passthrough = []
```

对应环境变量：

```text
ALPHA_BASH_TOOL_ENABLED
ALPHA_BASH_TOOL_DEFAULT_WORKDIR
ALPHA_BASH_TOOL_ALLOWED_WORKDIRS
ALPHA_BASH_TOOL_DEFAULT_TIMEOUT_SECONDS
ALPHA_BASH_TOOL_MAX_TIMEOUT_SECONDS
ALPHA_BASH_TOOL_MAX_OUTPUT_CHARS
ALPHA_BASH_TOOL_ENV_PASSTHROUGH
```

解析规则：

- `enabled` 默认 false，避免本地命令执行被意外暴露给远程 gateway。
- `allowed_workdirs` 支持项目相对路径和 `~`，加载后解析为真实路径并去重。
- `workdir` 每次调用都必须 resolve 后确认位于允许目录内。
- `env_passthrough` 默认空；不把 provider secrets 传给子进程。

需要更新：

- `src/alpha_agent/config.py`
- `config.example.toml`
- README 配置说明和工具说明
- `tests/test_config.py`

## 本地执行策略

`LocalShellBackend` 使用：

```python
subprocess.Popen(
    [bash_path, "-lc", command],
    cwd=resolved_workdir,
    env=sanitized_env,
    stdin=subprocess.DEVNULL,
    stdout=subprocess.PIPE,
    stderr=subprocess.PIPE,
    text=True,
    start_new_session=True,
)
```

关键点：

- 不使用 `shell=True`。
- POSIX 下用 `start_new_session=True`，timeout/cancel 时 kill process group。
- stdout/stderr 分开捕获，但最终都进入结构化返回。
- 用轮询循环读取输出并定期调用 `context.check_canceled("during_tool")`。
- 超时后先 SIGTERM，再短等待，再 SIGKILL。
- bash 路径优先 `shutil.which("bash")`，找不到时退到 `sh`，并在 metadata 记录 shell。

## Workdir 策略

`workdir` 校验在 policy 层完成：

- 空值使用 `default_workdir`。
- 展开 `~` 和相对路径。
- `resolve()` 后必须在某个 `allowed_workdirs` 下。
- 拒绝包含 NUL 字符的路径。
- 路径不存在时返回 `blocked` 或 `error`，不自动创建。
- 返回给模型和 trace 的路径使用项目相对路径或 `~` 形式，不写本机绝对路径。

## 环境变量与 Secret 治理

默认环境：

- 保留最小运行环境：`PATH`、`HOME`、`LANG`、`LC_ALL`、`SHELL`、`TMPDIR`。
- 注入配置允许的 `env_passthrough`。
- 不传递 Alpha 自己的 provider token、API key、gateway secret。

默认阻断变量名模式：

```text
*_API_KEY
*_TOKEN
*_SECRET
*_PASSWORD
ALPHA_CODEX_ACCESS_TOKEN
ALPHA_COMPATIBLE_API_KEY
ALPHA_DEEPSEEK_API_KEY
ALPHA_TAVILY_API_KEY
TAVILY_API_KEY
```

输出 redaction：

- 从当前进程环境和 config 中收集已知 secret value。
- 对长度足够的 secret 做精确替换为 `[REDACTED]`。
- 不做复杂正则猜测，避免误伤正常输出。

## 危险命令与交互命令策略

v1 不实现用户审批流，直接拒绝高风险命令。拒绝结果返回 `status=blocked`。

拒绝或要求未来审批的类别：

- 明确破坏性：`rm -rf /`、`git reset --hard`、`git clean -fd`、`chmod -R 777`、`chown -R`。
- 提权：`sudo`、`doas`、`pkexec`。
- shell 级后台逃逸：`nohup`、`disown`、`setsid`、尾随 `&`。
- 交互式编辑器或 TUI：`vim`、`vi`、`nano`、`less`、`more`、`top`、`htop`。
- 明显凭证输入命令：例如 `gh auth login --with-token`。

注意：这不是完整安全沙箱，只是本地工具边界。真正的高风险远程使用应在后续引入 sandbox/approval。

## 输出治理

输出处理顺序：

1. 按字节/字符上限保护内存。
2. strip ANSI escape。
3. secret redaction。
4. head/tail 截断。
5. 附加 truncation metadata。

`max_output_chars` 默认 30000。截断策略：

- 保留前 40% 和后 60%。
- 中间插入固定提示：`[output truncated: N chars omitted]`。
- `omitted_chars` 记录被省略数量。

第一阶段不做大输出文件持久化。若后续要持久化，应使用 `config.log_dir` 下的 tool-results 目录，并返回项目无关的可读描述，不能把本机绝对路径放进仓库文档或 prompt 固化文本。

## 退出码语义

新增 `semantics.py`，只解释常见命令的非错误退出码，不改变 `exit_code`：

- `grep` / `rg` / `ag` / `ack`: `1` 表示无匹配。
- `diff`: `1` 表示存在差异。
- `find`: `1` 表示部分路径不可访问。
- `test` / `[`: `1` 表示条件为 false。

结果字段：

```json
"return_code_interpretation": "No matches found"
```

## 与 ToolExecutor 的集成

需要调整 `ToolExecutor.execute()`：

- 创建 `ToolExecutionContext`，传给工具。
- `write_trace("tool.started", ...)` 使用工具可选的 `trace_arguments()`。
- `check_canceled` 在工具运行中可被调用。
- recover_errors 下，工具内部异常仍转为 `tool.failed` 并返回模型。
- 命令 exit 非 0 不触发 `tool.failed`。

建议 metadata 增加：

```json
{
  "tool_name": "bash",
  "tool_call_id": "...",
  "tool_index": 0,
  "result": {
    "metadata": {
      "shell": "bash",
      "elapsed_ms": 1234,
      "status": "completed",
      "exit_code": 1
    }
  }
}
```

## 默认注册策略

`build_default_tool_registry(config)`：

- `tools.bash.enabled = true` 时注册 `BashTool`。
- `tavily.api_key` 仍独立控制 `web_search`。
- deterministic name order 由现有 registry 保证。

测试期可以构造 `AlphaConfig(bash_tool=...)` 直接验证 registry names。

## 后台任务的第二阶段设计

只有在实现 process registry 后才开放模型参数：

```json
{
  "background": true,
  "notify_on_complete": true
}
```

同时新增一个 `process` tool：

- `list`
- `poll`
- `wait`
- `terminate`

后台任务必须由 daemon 持有生命周期，不能由 agent turn 的临时 Python 对象持有。否则 daemon 重启、session 结束、工具异常都会留下不可追踪进程。

## 实施任务

### Phase 1：执行上下文与配置基础

**任务 1：扩展 ToolExecutionContext**

说明：让工具可以拿到 session、tool_call、输出目录和取消检查函数。

验收：

- `ToolExecutor` 能向新式工具传递 context。
- `web_search` 已更新为新协议，现有 web search 行为不破。
- `tool.started` 支持工具自定义 trace 参数摘要。

验证：

```bash
PYTHONPATH=. uv run pytest tests/test_web_search_tool.py tests/test_agent_loop.py -q
```

涉及文件：

- `src/alpha_agent/tools/base.py`
- `src/alpha_agent/runtime/tools.py`
- `tests/test_agent_loop.py`

**任务 2：新增 bash 配置模型**

说明：在配置层加入 `[tools.bash]`，默认关闭。

验收：

- 默认配置不注册 bash。
- TOML 和环境变量都可开启 bash。
- timeout/output/workdir 配置有边界校验。

验证：

```bash
PYTHONPATH=. uv run pytest tests/test_config.py tests/test_daemon_manager.py -q
```

涉及文件：

- `src/alpha_agent/config.py`
- `config.example.toml`
- `README.md`
- `tests/test_config.py`
- `tests/test_daemon_manager.py`

### Phase 2：本地前台 Bash MVP

**任务 3：实现 shell policy 和 local backend**

说明：完成 workdir 校验、环境清洗、超时、取消、process group kill。

验收：

- `echo ok` 返回 `completed` 和 stdout。
- 非零退出码返回 `completed`，不抛工具异常。
- timeout 返回 `timeout`，并终止子进程组。
- cancel 返回 `canceled`，并终止子进程组。
- workdir 越界返回 `blocked`。

验证：

```bash
PYTHONPATH=. uv run pytest tests/test_bash_tool.py -q
```

涉及文件：

- `src/alpha_agent/tools/shell/backend.py`
- `src/alpha_agent/tools/shell/local.py`
- `src/alpha_agent/tools/shell/policy.py`
- `tests/test_bash_tool.py`

**任务 4：实现 BashTool schema 和返回格式**

说明：将 backend 接入 `BashTool`，提供严格模型契约和结构化 JSON 输出。

验收：

- schema required 只有 `command`。
- unknown argument 被 strict schema 拒绝或在工具内明确拒绝。
- `ToolResult.output` 是 JSON object。
- trace metadata 不包含完整大型命令内容。

验证：

```bash
PYTHONPATH=. uv run pytest tests/test_bash_tool.py tests/test_agent_loop.py -q
```

涉及文件：

- `src/alpha_agent/tools/bash.py`
- `src/alpha_agent/tools/default.py`
- `tests/test_bash_tool.py`

### Phase 3：治理能力

**任务 5：实现输出治理**

说明：完成 ANSI stripping、secret redaction、head/tail 截断。

验收：

- ANSI escape 不出现在 stdout/stderr。
- 已知 secret value 被替换为 `[REDACTED]`。
- 超长输出被截断，并设置 `truncated=true` 和 `omitted_chars`。

验证：

```bash
PYTHONPATH=. uv run pytest tests/test_bash_tool.py -q
```

涉及文件：

- `src/alpha_agent/tools/shell/output.py`
- `tests/test_bash_tool.py`

**任务 6：实现危险命令和交互命令拒绝**

说明：在执行前拒绝高风险或会挂住的命令。

验收：

- `sudo ...` 返回 `blocked`。
- `nohup ... &` 返回 `blocked`。
- `vim file` 返回 `blocked`。
- 普通 `git status`、`uv run pytest` 不被误拦。

验证：

```bash
PYTHONPATH=. uv run pytest tests/test_bash_tool.py -q
```

涉及文件：

- `src/alpha_agent/tools/shell/policy.py`
- `tests/test_bash_tool.py`

**任务 7：实现退出码语义解释**

说明：让常见命令的特殊退出码变得可读。

验收：

- `rg pattern missing-dir` 的真实错误仍保留 exit code。
- `rg no-match file` 返回 `No matches found` 语义。
- `diff a b` exit 1 返回 `Files differ` 语义。

验证：

```bash
PYTHONPATH=. uv run pytest tests/test_bash_tool.py -q
```

涉及文件：

- `src/alpha_agent/tools/shell/semantics.py`
- `tests/test_bash_tool.py`

### Phase 4：集成与文档

**任务 8：接入默认 registry 和 daemon agent factory**

说明：通过配置开关让 daemon-owned agent 自动获得 bash tool。

验收：

- 默认 registry names 不包含 `bash`。
- 开启配置后 registry names 包含 `bash`。
- Tavily 和 bash 可同时注册。

验证：

```bash
PYTHONPATH=. uv run pytest tests/test_daemon_manager.py tests/test_web_search_tool.py -q
```

涉及文件：

- `src/alpha_agent/tools/default.py`
- `tests/test_daemon_manager.py`

**任务 9：更新 README 和配置示例**

说明：记录启用方式、风险边界、适用场景和非目标。

验收：

- README 说明默认关闭。
- README 说明 bash 不是安全 sandbox。
- `config.example.toml` 包含 `[tools.bash]`。
- 文档不包含本机绝对路径。

验证：

```bash
rg -n "$PWD|$HOME" README.md config.example.toml docs
PYTHONPATH=. uv run pytest tests/test_config.py -q
```

涉及文件：

- `README.md`
- `config.example.toml`

## 全量验收标准

- bash 默认关闭，显式开启后可注册。
- 所有 bash 调用都在允许 workdir 内执行。
- 命令超时和 session cancel 都会终止子进程组。
- 非零退出码作为结构化结果返回，不触发工具异常。
- 输出经过 ANSI 清理、secret redaction 和大小限制。
- `tool.started`、`tool.completed`、`tool.failed` trace 可审计，且不泄露完整 secret。
- README 和配置示例完整说明使用方式和边界。
- 测试覆盖 config、registry、tool executor、bash tool、agent loop。

全量验证：

```bash
PYTHONPATH=. uv run pytest -q
```

## 风险与缓解

| 风险 | 影响 | 缓解 |
| --- | --- | --- |
| 本地命令执行被远程 gateway 误启用 | 高 | 默认关闭；README 明确风险；配置开关显式命名 |
| 子进程无法被取消 | 高 | ToolExecutionContext + poll loop + process group kill |
| secret 泄漏进 trace 或 LLM context | 高 | 环境清洗、输出 redaction、trace 参数摘要 |
| 输出过大导致 context 膨胀 | 中 | max_output_chars + head/tail 截断 |
| 危险命令误执行 | 高 | v1 无 approval 时直接 blocked |
| policy 误拦正常命令 | 中 | 拒绝规则保持窄集合；测试覆盖常见开发命令 |
| 后台任务失控 | 高 | v1 不开放 background；后续必须配套 process registry |

## 开放问题

- 是否允许 `git` 破坏性命令在用户明确请求后执行？如果需要，需要先设计 approval 机制。
- 是否要引入大输出持久化文件？如果需要，应先设计 tool-results 生命周期和清理策略。
- 是否要把 `bash` 拆成 `terminal` 命名？建议用 `bash`，因为当前工具 registry 更偏 provider-neutral function tool，且执行语义就是 shell command。
- 是否要提供专门的 file/search/patch tools？建议后续提供，减少模型滥用 bash 做文件读写。

## 推荐实施顺序

严格按 Phase 1 到 Phase 4 顺序实现。不要先写 `BashTool` 再补取消和安全边界；那会把最危险的行为先暴露出来。每个 phase 完成后运行对应测试，再进入下一阶段。
