"""
四维四子棋 - WebSocket 服务端
===========================

基于 FastAPI + WebSocket 的多人在线对弈服务。
支持人类玩家与电脑玩家混合对弈。
"""

import json
import asyncio
import sys
from pathlib import Path
from typing import Optional

# 确保项目根目录在 Python 路径中（支持 python FourInFour/main.py 直接运行）
_PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles

from FourInFour.GameCore import GameRoom, PLAYER_SYMBOLS, EMPTY, pos_4d_to_2d

# ============================================================================
# 应用初始化
# ============================================================================

app = FastAPI(title="四维四子棋")

BASE_DIR = Path(__file__).resolve().parent
STATIC_DIR = BASE_DIR / "static"
TEMPLATES_DIR = BASE_DIR / "templates"
app.mount("/static", StaticFiles(directory=str(STATIC_DIR)), name="static")

# 全局游戏房间（不预注册玩家，由客户端动态加入）
game = GameRoom()


# ============================================================================
# 连接管理器
# ============================================================================

class ConnectionManager:
    """
    WebSocket 连接管理器（槽位制）。

    所有新连接默认为观战者；
    人类玩家通过「坐下」占用槽位，「站起」释放；
    机器人由任意客户端指定槽位添加/删除。
    """

    def __init__(self) -> None:
        # 槽位 → WebSocket 映射
        self._slot_ws: dict[int, WebSocket] = {}
        # WebSocket → 槽位 反向映射
        self._ws_slot: dict[WebSocket, int] = {}
        # 观战者列表
        self.spectators: list[WebSocket] = []

    async def accept(self, websocket: WebSocket) -> None:
        """接受新连接（一律先作为观战者）。"""
        await websocket.accept()
        self.spectators.append(websocket)

    def disconnect(self, websocket: WebSocket) -> Optional[int]:
        """断开连接，若该连接在某个槽位上则自动站起。"""
        slot = self._ws_slot.pop(websocket, None)
        if slot is not None:
            self._slot_ws.pop(slot, None)
            return slot
        if websocket in self.spectators:
            self.spectators.remove(websocket)
        return None

    def sit(self, websocket: WebSocket, slot: int) -> bool:
        """
        观战者坐到指定槽位。

        返回 True 表示成功；False 表示已坐在其他位置或槽位无效。
        """
        if websocket in self._ws_slot:
            return False  # 已经坐着
        if slot not in (1, 2, 3) or slot in self._slot_ws:
            return False  # 槽位无效或已被占用
        if websocket in self.spectators:
            self.spectators.remove(websocket)
        self._slot_ws[slot] = websocket
        self._ws_slot[websocket] = slot
        return True

    def stand(self, websocket: WebSocket) -> Optional[int]:
        """
        站起，释放槽位。

        返回被释放的槽位号，若本来就没坐着则返回 None。
        """
        slot = self._ws_slot.pop(websocket, None)
        if slot is not None:
            self._slot_ws.pop(slot, None)
            self.spectators.append(websocket)
        return slot

    def get_player_ws(self, slot: int) -> Optional[WebSocket]:
        """获取指定槽位的 WebSocket（人类玩家）。"""
        return self._slot_ws.get(slot)

    def get_slot(self, websocket: WebSocket) -> Optional[int]:
        """获取某 WebSocket 当前坐的槽位号。"""
        return self._ws_slot.get(websocket)

    def slot_has_human(self, slot: int) -> bool:
        """指定槽位是否有人类坐着。"""
        return slot in self._slot_ws

    async def broadcast(self, message: dict) -> None:
        """向所有连接广播。"""
        payload = json.dumps(message, ensure_ascii=False)
        targets = list(self._slot_ws.values()) + self.spectators
        for ws in targets:
            try:
                await ws.send_text(payload)
            except Exception:
                pass

    async def send_to(self, websocket: WebSocket, message: dict) -> None:
        """向单个连接发送。"""
        try:
            await websocket.send_text(json.dumps(message, ensure_ascii=False))
        except Exception:
            pass



# 全局连接管理器
manager = ConnectionManager()


# ============================================================================
# 电脑玩家自动落子
# ============================================================================

COMPUTER_MOVE_DELAY = 0.1  # 电脑落子间隔（秒）

# 后台电脑落子 Task（支持重置中断）
_computer_task: Optional[asyncio.Task] = None


def _launch_computer_turns() -> None:
    """启动后台电脑落子 Task（取消旧的以防重复）。"""
    global _computer_task
    if _computer_task and not _computer_task.done():
        _computer_task.cancel()
    _computer_task = asyncio.create_task(process_computer_turns())


async def process_computer_turns() -> None:
    """
    检查当前回合是否为电脑玩家，若是则自动执行落子。

    支持连续执行（例如玩家2和玩家3都是电脑时，连续自动落子）。
    每个电脑落子之间有固定延迟，以便人类观看。
    暂停期间不执行任何操作。
    """
    while not game.game_over and game.is_computer_turn():
        # 暂停期间等待，不执行落子
        if game.paused:
            await asyncio.sleep(0.2)
            continue

        await asyncio.sleep(COMPUTER_MOVE_DELAY)

        # 再次检查暂停状态（睡眠期间可能被暂停）
        if game.paused:
            continue

        player = game.get_player(game.current_turn)
        if player is None:
            break

        # 电脑选择落子位置（传入所有玩家最近一步棋的索引）
        move_data = player.choose_move(game.board, game._last_move_index)
        # 引擎可能返回四维 (w,x,y,z) 或二维 (row,col)，统一转为二维供 game.make_move
        if len(move_data) == 4:
            row, col = pos_4d_to_2d(*move_data)
        else:
            row, col = move_data

        # 提取思考统计（普通型 AI 才有效）
        thinking_time = 0.0
        thinking_iters = 0
        if hasattr(player, 'thinking_stats'):
            stats = player.thinking_stats
            thinking_time = stats["time"]
            thinking_iters = stats["iters"]

        # 执行落子
        result = game.make_move(player.player_id, row, col,
                                thinking_time, thinking_iters)

        # 广播结果
        await manager.broadcast({"type": "move_result", **result})
        await manager.broadcast({"type": "state", **game.get_state()})

        if result.get("game_over"):
            break


# ============================================================================
# HTTP 路由
# ============================================================================

@app.get("/", response_class=HTMLResponse)
async def get_index():
    """返回游戏前端页面。"""
    html_path = TEMPLATES_DIR / "index.html"
    if html_path.exists():
        return HTMLResponse(content=html_path.read_text(encoding="utf-8"))
    return HTMLResponse(content="<h1>index.html 尚未创建</h1>", status_code=200)


# ============================================================================
# WebSocket 路由
# ============================================================================

@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    """WebSocket 核心端点（槽位制）。"""
    await manager.accept(websocket)
    my_slot = None

    await manager.send_to(websocket, {
        "type": "welcome", "role": "spectator",
        "symbols": {"1": PLAYER_SYMBOLS[1], "2": PLAYER_SYMBOLS[2], "3": PLAYER_SYMBOLS[3]},
    })
    await manager.send_to(websocket, {"type": "state", **game.get_state()})

    try:
        while True:
            raw = await websocket.receive_text()
            try:
                data = json.loads(raw)
            except json.JSONDecodeError:
                await manager.send_to(websocket, {"type": "error", "message": "无效的 JSON"})
                continue
            msg_type = data.get("type", "")

            if msg_type == "sit":
                slot = data.get("slot")
                if slot not in (1, 2, 3):
                    await manager.send_to(websocket, {"type": "error", "message": "无效的座位号"})
                    continue
                if not game.sit_human(slot):
                    await manager.send_to(websocket, {"type": "error", "message": f"座位 {slot} 不可用"})
                    continue
                if not manager.sit(websocket, slot):
                    game.stand_up(slot)
                    await manager.send_to(websocket, {"type": "error", "message": "无法坐下"})
                    continue
                my_slot = slot
                await manager.send_to(websocket, {"type": "sit_confirm", "slot": slot})
                await manager.broadcast({"type": "chat", "message": f"玩家坐到了座位 {slot} ({PLAYER_SYMBOLS[slot]})"})
                await manager.broadcast({"type": "state", **game.get_state()})

            elif msg_type == "stand":
                if manager.stand(websocket) is not None:
                    game.stand_up(my_slot)
                    await manager.send_to(websocket, {"type": "stand_confirm"})
                    await manager.broadcast({"type": "chat", "message": f"座位 {my_slot} 的玩家站起了"})
                    my_slot = None
                    await manager.broadcast({"type": "state", **game.get_state()})
                else:
                    await manager.send_to(websocket, {"type": "error", "message": "你没有坐在任何座位上"})

            elif msg_type == "add_robot":
                slot = data.get("slot")
                strategy = data.get("strategy", "random")
                time_limit = float(data.get("time_limit", 5.0))
                max_iters = int(data.get("max_iters", 10000))
                if slot not in (1, 2, 3):
                    await manager.send_to(websocket, {"type": "error", "message": "无效的座位号"})
                    continue
                if not game.add_robot(slot, strategy, time_limit, max_iters):
                    await manager.send_to(websocket, {"type": "error", "message": f"座位 {slot} 不可用"})
                    continue
                await manager.broadcast({"type": "chat", "message": f"🤖 机器人[{strategy}]放置在座位 {slot}"})
                await manager.broadcast({"type": "state", **game.get_state()})

            elif msg_type == "remove_robot":
                slot = data.get("slot")
                if not game.remove_robot(slot):
                    await manager.send_to(websocket, {"type": "error", "message": f"座位 {slot} 没有机器人"})
                    continue
                await manager.broadcast({"type": "chat", "message": f"座位 {slot} 的机器人已被移除"})
                await manager.broadcast({"type": "state", **game.get_state()})

            elif msg_type == "move":
                await _handle_move(websocket, data, my_slot)
            elif msg_type == "pause":
                await _handle_pause()
            elif msg_type == "resume":
                await _handle_resume()
            elif msg_type == "undo":
                await _handle_undo(websocket, my_slot)
            elif msg_type == "start_game":
                await _handle_start_game()
            elif msg_type == "reset":
                await _handle_reset()
            elif msg_type == "ping":
                await manager.send_to(websocket, {"type": "pong"})
            else:
                await manager.send_to(websocket, {"type": "error", "message": f"未知消息类型：{msg_type}"})

    except WebSocketDisconnect:
        pass
    except Exception as e:
        print(f"[WS] 异常：{e}")
    finally:
        disconnected_slot = manager.disconnect(websocket)
        if disconnected_slot is not None:
            game.stand_up(disconnected_slot)
            await manager.broadcast({"type": "chat", "message": f"座位 {disconnected_slot} 的玩家断开连接"})
            await manager.broadcast({"type": "state", **game.get_state()})
        else:
            await manager.broadcast({"type": "chat", "message": "一位观战者离开了"})


async def _handle_move(websocket: WebSocket, data: dict, slot: Optional[int]) -> None:
    """处理人类落子（需已坐下）。"""
    if slot is None:
        await manager.send_to(websocket, {"type": "error", "message": "请先坐下再落子"})
        return
    row = data.get("row"); col = data.get("col")
    if row is None or col is None:
        await manager.send_to(websocket, {"type": "error", "message": "缺少 row 或 col"})
        return
    result = game.make_move(slot, row, col)
    if result["success"]:
        await manager.broadcast({"type": "move_result", **result})
        await manager.broadcast({"type": "state", **game.get_state()})
        _launch_computer_turns()
    else:
        await manager.send_to(websocket, {"type": "error", "message": result["message"]})


async def _handle_start_game() -> None:
    """开始游戏。"""
    result = game.start_game()
    if result["success"]:
        await manager.broadcast({"type": "chat", "message": "游戏开始！"})
        await manager.broadcast({"type": "state", **game.get_state()})
        _launch_computer_turns()
    else:
        await manager.broadcast({"type": "chat", "message": result["message"]})


async def _handle_reset() -> None:
    """重置游戏（先中断电脑回合）。"""
    global _computer_task
    if _computer_task and not _computer_task.done():
        _computer_task.cancel()
        _computer_task = None
    new_state = game.reset()
    await manager.broadcast({"type": "state", **new_state})
    await manager.broadcast({"type": "chat", "message": "游戏已重置，新一局开始！"})


async def _handle_pause() -> None:
    """
    暂停游戏。

    - 若当前是电脑思考时间，取消电脑 task（不落子）
    - 设置 paused 状态，广播更新
    """
    if not game.started:
        await manager.broadcast({"type": "chat", "message": "游戏尚未开始"})
        return
    if game.game_over:
        await manager.broadcast({"type": "chat", "message": "游戏已结束"})
        return
    if game.paused:
        await manager.broadcast({"type": "chat", "message": "游戏已经在暂停中"})
        return

    # 如果正在电脑思考，取消电脑 task
    global _computer_task
    if _computer_task and not _computer_task.done():
        _computer_task.cancel()
        _computer_task = None

    game.paused = True
    await manager.broadcast({"type": "state", **game.get_state()})
    await manager.broadcast({"type": "chat", "message": "⏸ 游戏已暂停"})


async def _handle_resume() -> None:
    """
    继续游戏。

    - 清除 paused 状态
    - 若当前回合是电脑，重新启动电脑思考
    """
    if not game.started:
        await manager.broadcast({"type": "chat", "message": "游戏尚未开始"})
        return
    if game.game_over:
        await manager.broadcast({"type": "chat", "message": "游戏已结束"})
        return
    if not game.paused:
        await manager.broadcast({"type": "chat", "message": "游戏未暂停"})
        return

    game.paused = False
    await manager.broadcast({"type": "state", **game.get_state()})
    await manager.broadcast({"type": "chat", "message": "▶ 游戏继续"})

    # 若当前回合是电脑，启动思考
    if game.is_computer_turn():
        _launch_computer_turns()


async def _handle_undo(websocket: WebSocket, slot: Optional[int]) -> None:
    """
    悔棋：撤销上一步棋。

    限制：
    - 仅在玩家思考时间（人类玩家的回合）或暂停时可用
    - 电脑思考时不可悔棋
    - 悔棋后若轮到电脑，自动进入暂停模式
    """
    if not game.started:
        await manager.send_to(websocket, {"type": "error", "message": "游戏尚未开始"})
        return
    if game.game_over:
        await manager.send_to(websocket, {"type": "error", "message": "游戏已结束"})
        return

    # 检查是否处于可悔棋状态：
    # 允许：暂停中、或当前是人类玩家的回合
    # 不允许：电脑正在思考（即非暂停且当前是电脑回合）
    can_undo = game.paused or not game.is_computer_turn()
    if not can_undo:
        await manager.send_to(websocket, {
            "type": "error",
            "message": "电脑思考中，请先暂停再悔棋"
        })
        return

    # 执行悔棋
    result = game.undo()
    if not result["success"]:
        await manager.send_to(websocket, {"type": "error", "message": result["message"]})
        return

    await manager.broadcast({"type": "state", **game.get_state()})
    await manager.broadcast({
        "type": "chat",
        "message": f"↩ 悔棋：撤销玩家 {result['undone_player']} 的落子 "
                   f"[{result['undone_row']+1},{result['undone_col']+1}]"
    })

    # 悔棋后，若当前回合是电脑，自动进入暂停模式
    if game.is_computer_turn():
        global _computer_task
        if _computer_task and not _computer_task.done():
            _computer_task.cancel()
            _computer_task = None
        game.paused = True
        await manager.broadcast({"type": "state", **game.get_state()})
        await manager.broadcast({
            "type": "chat",
            "message": "⏸ 悔棋后轮到电脑，自动暂停。按继续键开始电脑思考"
        })


# ============================================================================
# 启动入口
# ============================================================================

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=6006)
