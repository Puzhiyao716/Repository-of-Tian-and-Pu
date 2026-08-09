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
- **普通型（Normal）** — 基于 MCTS + UCB1 搜索，通过大量随机模拟评估每个落子的胜率，在探索与利用之间自动平衡

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
