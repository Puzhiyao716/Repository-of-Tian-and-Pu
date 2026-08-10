# Repository-of-Tian-and-Pu

## 四维四子棋（4D Connect Four）

基于 **FastAPI + WebSocket** 的多人在线四维四子棋对弈平台，支持人类玩家与 AI 电脑玩家混合对弈。

---

### 项目概述

在 4×4×4×4 的四维空间中连四子即获胜。棋盘共 256 个位置，通过 16×16 的二维网格呈现给玩家。三名玩家分别使用 **○ / ✕ / △** 三种符号对弈。

### 项目结构

```
Repository-of-Tian-and-Pu/
└── FourInFour/
    ├── main.py                 # FastAPI + WebSocket 服务端入口
    ├── AI/
    │   └── mcts.py            # 蒙特卡洛树搜索 (MCTS) + UCB1 AI 引擎
    ├── GameCore/
    │   ├── __init__.py         # 统一导出所有对外接口
    │   ├── board.py            # 棋盘核心：4D↔2D 坐标映射、获胜线预计算
    │   ├── game.py             # 游戏房间状态管理（玩家注册、落子校验、胜负判定）
    │   └── player.py           # 玩家类层次（Human / Random / MCTS-Normal）
    ├── static/
    │   └── game.js             # 前端 WebSocket 交互逻辑
    └── templates/
        └── index.html          # 前端游戏界面
```

### 技术栈

| 层级 | 技术 |
|------|------|
| 后端框架 | Python + [FastAPI](https://fastapi.tiangolo.com/) + WebSocket |
| 前端 | 原生 HTML / CSS / JavaScript |
| AI 算法 | 蒙特卡洛树搜索 (MCTS) + UCB1 |
| 通信协议 | WebSocket + JSON 消息 |
| 运行方式 | uvicorn (默认端口 6006) |

### AI 玩家

提供两种电脑难度：

- **随机型（Random）** — 在所有空位中随机落子
- **普通型（Normal）** — 基于 MCTS + UCB1 搜索，可配置时间限制和迭代上限

### 蒙特卡洛树搜索 (MCTS) 算法

普通型 AI 实现了完整的 MCTS 算法，每次落子前在后台运行搜索。下面是完整的逻辑链条：

#### 1. 调用入口与循环控制

```
用户点击"开始游戏" / 人类落子后自动触发
    ↓
main.py  _launch_computer_turns()
    ↓ asyncio.create_task()
main.py  process_computer_turns()
    ↓  while not game_over and is_computer_turn():
player.py ComputerPlayerNormal.choose_move(board)
    ↓  惰性初始化 MCTSEngine(time_limit, max_iters)
mcts.py  MCTSEngine.choose_move(board, turn)
```

**终止条件**（满足任一即停止迭代）：
- 迭代次数达到 `max_iters`（默认 10000，UI 可调）
- 用时超过 `time_limit` 秒（默认 5.0 秒，UI 可调）

#### 2. 单次迭代四阶段

每一轮迭代执行 Selection → Expansion → Simulation → Backpropagation 四个阶段：

```
for _ in range(max_iters):          ← 迭代循环（受时间/次数上限约束）
    if timeout: break
    
    leaf = _select(root)            ← ① Selection + Expansion
    winner = _simulate(leaf)        ← ② Simulation
    _backprop(leaf, winner)         ← ③ Backpropagation
```

##### ① Selection（选择）— `_select(node)`

从根节点出发，沿树向下走到叶节点。每层由"当前回合玩家"选择对自己 UCB1 值最高的子节点（体现"每个对手都在为自己赢而奋斗"）。

此外，在每个节点扩展之前，先检查两步优化：

| 优化 | 触发条件 | 行为 |
|------|----------|------|
| **必胜一步检测** | 首次进入该节点时 (`losing_checked == False`) | 扫描所有获胜线，若存在"本回合玩家 3 子 + 1 空"，直接扩展该必胜落子并返回 |
| **必败防御** | 节点记录了 `losing_moves` | 若上一轮模拟发现对手存在必胜落子，强制扩展该防御位置 |

##### ② Expansion（扩展）— `_expand(node, untried)`

从 `untried` 列表中弹出一个随机未尝试的落子，创建新棋盘状态和新子节点：

```
untried.pop() → (row, col)
    ↓ pos_2d_to_4d → pos_4d_to_index
new_board[idx] = parent.turn
    ↓
child = _Node(new_board, next_turn, move, parent)
parent.children.append(child)
```

若来自必胜一步检测，则走 `_expand_specific()`，同时设置 `child.winner = parent.turn` 标记必胜。

##### ③ Simulation（模拟）— `_simulate(node)`

从新节点开始，随机模拟至终局：

- 若节点已标记胜者（`node.winner`），直接返回
- **前两步优化**：检查当前玩家是否存在一步获胜，是则直接落子并返回胜者
- 之后纯随机：从所有空位中随机选一个落子，每次落子后用 `check_win_at()` 增量判断胜负
- 棋盘满且无胜者 → 返回 0（平局）

##### ④ Backpropagation（回传）— `_backprop(node, winner)`

从当前节点沿 `parent` 链向上走到根节点，对路径上每个节点：
- `visits += 1`
- 若有胜者（非 0），`wins[winner] += 1`

#### 3. 最终决策与输出

迭代结束后，从根节点的所有子节点中选 `visits` 最多者作为最终落子：

```
best = max(root.children, key=lambda c: c.visits)
return best.move, elapsed_time, actual_iters
```

返回的 `(row, col)` 经以下链路最终在前端日志显示：

```
MCTSEngine.choose_move → (move, elapsed, iters)
    ↓
ComputerPlayerNormal.choose_move → 存储到 _last_thinking_*
    ↓ thinking_stats property
main.py process_computer_turns → thinking_stats → make_move(thinking_time, thinking_iters)
    ↓
game.py _build_result → result["thinking"] = {time, iters}
    ↓ WebSocket broadcast move_result
game.js → 日志追加 "思考 X.XXXs · N 次迭代"
```

#### 4. UCB1 公式

$$
UCB1 = \frac{wins[player]}{visits} + \sqrt{2} \times \sqrt{\frac{\ln(parent.visits)}{visits}}
$$

- 第一项（**利用项**）：该节点历史上 player 获胜的比例
- 第二项（**探索项**）：鼓励访问次数少的节点，C = √2 是理论最优值
- 未访问过的节点 → 返回 `inf`，确保每个子节点至少被访问一次

#### 5. 关键类结构

```
_Node                              ← MCTS 树节点
├── board: List[int]               ← 棋盘快照（256 长度）
├── turn: int                      ← 当前回合玩家
├── move: (row, col)               ← 到达此节点的落子
├── parent: _Node                  ← 父节点
├── children: List[_Node]          ← 已扩展的子节点
├── visits: int                    ← 被访问次数
├── wins: {1: int, 2: int, 3: int} ← 各玩家获胜次数
├── winner: int|None               ← 必胜节点标记（优化用）
└── _untried: List[(row, col)]     ← 尚未尝试的落子（惰性初始化）

MCTSEngine                         ← 搜索引擎
├── player_id: int                 ← AI 自己的玩家编号
├── time_limit: float              ← 时间上限（秒）
├── max_iters: int                 ← 迭代次数上限
├── choose_move(board, turn)       ← 入口
├── _select(node)                  ← Selection + Expansion
├── _expand(node, untried)         ← 随机扩展
├── _expand_specific(node, idx)    ← 指定必胜落子扩展
├── _simulate(node)                ← 随机模拟
├── _backprop(node, winner)        ← 结果回传
├── _is_terminal(node)             ← 终局判断
└── _find_winning_move(board, p)   ← 必胜一步检测（O(线数)）
```

### 快速开始

1. **安装依赖**

```bash
pip install fastapi uvicorn
```

2. **启动服务**

```bash
cd Repository-of-Tian-and-Pu
python FourInFour/main.py
```

3. **打开浏览器访问**

```
http://localhost:6006
```

打开多个浏览器窗口即可多人对弈，或在座位面板添加电脑玩家。

### 游戏规则

- **棋盘**：4 维空间 (w, x, y, z)，每个维度 4 格，映射为 16×16 的二维面板
- **玩家**：3 人（○ 玩家1 / ✕ 玩家2 / △ 玩家3），轮流落子
- **获胜**：在任意一条 4D 直线上连成四子即获胜
- **操作**：观战者先点击「坐下」占用座位，再点击棋盘空位落子；可随时添加或移除电脑玩家

### 通信协议

前端与后端通过 WebSocket 发送 JSON 消息，主要消息类型：

| type | 方向 | 说明 |
|------|------|------|
| `sit` | 客户端→服务端 | 人类坐下 |
| `stand` | 客户端→服务端 | 人类站起 |
| `add_robot` | 客户端→服务端 | 添加电脑玩家 |
| `remove_robot` | 客户端→服务端 | 移除电脑玩家 |
| `move` | 客户端→服务端 | 人类落子 |
| `start_game` | 客户端→服务端 | 开始游戏 |
| `reset` | 客户端→服务端 | 重置游戏 |
| `state` | 服务端→客户端 | 游戏状态同步 |
| `chat` | 服务端→客户端 | 聊天/日志消息 |

### 设计原则

- 游戏逻辑层（`GameCore/`）不依赖任何网络层，保持纯函数式设计
- 所有公开方法返回值可 JSON 序列化
- 棋盘 256 个获胜线在模块加载时一次性预计算，之后 O(1) 查表判定胜负
- 支持回合制：人类落子 → 连续电脑回合 → 回到人类回合
