"""
智慧型 AI —— 蒙特卡洛树搜索 (MCTS) + UCB1 + Alphabet剪枝树
========================================

核心思想：
    1. 以自己赢为第一目的
    2. 假设所有对手也各自为赢而奋斗
    3. 通过"必胜点"来减少对"必输"策略的探索
    4. 用大量随机模拟来评估每个位置落子的胜率
    5. UCB1 公式在「利用已知好着」与「探索未知着」之间平衡

算法流程：
    1. 从根节点出发，先检查必胜点
    2. 通过必胜点，使用Alphabet剪枝树展开，直到不存在任何必胜点，即无法剪枝
    3. 对所有根节点进行评分，评分方式采用双方随机落子，过程中如果出现必胜点，按照必胜点优先落子
    4. 每个节点维护三份分数，代表三个玩家的胜率。
    5. 所有叶节点至少探索一次之后，使用UCB1方法随机选择节点探索
    6. 时间到或达到迭代次数上限后，选择访问次数最多的一级子节点作为落子方案

AB 剪枝核心思想（point_to_try 驱动）：
    - 自己有必胜点 → 只走必胜点（其它分支不可能更好）
    - 下家恰好有 1 个必胜点 → 必须堵（否则下家直接赢）
    - 上家有 2+ 个必胜点 → 必须堵一个（否则上家下回合必赢）
    - 否则 → 不剪枝，所有空位均可探索
"""
import math
import random
import time
from typing import List, Tuple, Optional, Dict

from FourInFour.GameCore import (
    EMPTY, TOTAL_CELLS, pos_4d_to_index, pos_2d_to_4d, check_win_at,
)

# UCB1 探索参数（√2 是理论最优值）
UCB_C = 1.414


# ============================================================================
# MCTS 节点
# ============================================================================

class _AB_Node:
    """MCTS 搜索树节点（带 Alphabet 剪枝）。"""

    def __init__(
        self,
        board: List[int],
        turn: int,
        move: Optional[Tuple[int, int]] = None,
        parent: Optional["_AB_Node"] = None,
        move_idx: int = -1,
        key_points: Optional[Dict[int, List[Tuple[int, int]]]] = None,
    ) -> None:
        # ---- 棋盘状态 ----
        self.board: List[int] = board.copy()          # 当前棋盘快照（256 长度）
        self.turn: int = turn                          # 当前回合玩家编号（1/2/3）
        self.move: Optional[Tuple[int, int]] = move    # 到达此节点的落子 (row, col)
        self.move_idx: int = move_idx                  # 一维索引（root 为 -1）
        self.parent: Optional["_AB_Node"] = parent     # 父节点

        # ---- 关联玩家 ----
        self.up_player: int = (turn - 2) % 3 + 1       # 上一回合玩家
        self.down_player: int = turn % 3 + 1            # 下一回合玩家

        # ---- MCTS 统计 ----
        self.children: List["_AB_Node"] = []            # 子节点列表
        self.visits: int = 0                            # 被访问次数
        self.wins: Dict[int, int] = {1: 0, 2: 0, 3: 0} # 各玩家视角的获胜次数
        self._untried: Optional[List[Tuple[int, int]]] = None  # 尚未尝试的落子（惰性初始化）
        self.winner: Optional[int] = None               # 终局标记（None=未确定, 0=平局）

        # ---- 必胜点剪枝 ----
        # Key_Points: {player_id: [(row, col), ...]}
        # 记录各玩家当前可一步获胜的位置
        self.Key_Points: Dict[int, List[Tuple[int, int]]] = (
            key_points.copy() if key_points else {1: [], 2: [], 3: []}
        )
        # ---- 节点状态 ----
        self.is_root = self.parent is None
        self.is_leaf = self.parent is not None 

    @classmethod
    def from_parent(
        cls, parent: "_AB_Node", Parent_move: Tuple[int, int]
    ) -> "_AB_Node":
        """由父节点扩展新节点，并更新必胜点状态。"""
        assert parent.is_root == True ,\
            "父节点必须是根节点才能扩展新节点"
        
        new_board = list(parent.board)
        new_board[Parent_move] = parent.turn

        # 深拷贝 Key_Points，移除已被占据的必胜点
        KeyPoint: Dict[int, Tuple[list[int, int]]] = {}
        for pid, pts in parent.Key_Points.items():
            KeyPoint[pid] = [pt for pt in pts if pt != Parent_move]

        # 检测新落子是否产生了新的必胜点（对刚落子的玩家）
        Parent_move_idx = pos_4d_to_index(*pos_2d_to_4d(*Parent_move))
        _won, new_key_point = check_win_at(new_board, Parent_move_idx, parent.turn,
                                     potential_win=True)
        if new_key_point:
            for pt in new_key_point:
                key = (pt[0], pt[1])
                if key not in KeyPoint[parent.turn]:
                    KeyPoint[parent.turn].append(key)

        next_turn = parent.turn % 3 + 1

        child = cls(
            board=new_board,
            turn=next_turn,
            move=Parent_move,
            parent=parent,
            move_idx=Parent_move,
            key_points=KeyPoint,
        )
        return child

    @staticmethod
    def _search_all_moves(
        board: List[int], turn: int,
        KeyPoint: Dict[int, List[Tuple[int, int]]]
    ) -> List[Tuple[int, int]]:
        """
        根据必胜点规则搜索所有候选落子，统一以 (row, col) 格式返回。
        随机选择或进一步过滤由调用方负责。

        规则：
            1. 自己有必胜点 → 只返回必胜点
            2. 自己无，下家有且仅有一个 → 返回那一个
            3. 自己和下家无，上家有 2+ → 返回上家必胜点列表
            4. 否则 → 返回所有空位
        """
        down_player = turn % 3 + 1
        up_player = (turn - 2) % 3 + 1

        own_keypoint = KeyPoint.get(turn, [])
        down_keypoint = KeyPoint.get(down_player, [])
        up_keypoint = KeyPoint.get(up_player, [])

        # 规则 1：自己有必胜点
        if own_keypoint:
            return [pt for pt in own_keypoint]

        # 规则 2：下家恰好一个必胜点，且上家无
        if len(down_keypoint) == 1 and len(up_keypoint) == 0:
            return [pt for pt in down_keypoint]

        # 规则 3：自己和下家无，上家 2+
        if len(down_keypoint) == 0 and len(up_keypoint) >= 2:
            return [pt for pt in up_keypoint]

        # 规则 4：所有空位，统一转为 (row, col)
        result: List[Tuple[int, int]] = []
        for i, v in enumerate(board):
            if v == EMPTY:
                result.append(_AB_Node._index_to_2d(i))
        return result

    def Make_All_Children(self) -> None:
        """为当前节点生成所有可能的子节点（考虑必胜点）。"""
        candidates = self._search_all_moves(
            self.board, self.turn, self.Key_Points
        )
        for move in candidates:
            child = _AB_Node.from_parent(self, move)
            self.children.append(child)
        if self.children:
            self.is_leaf = False

    def _simulate(self) -> int:
        """从当前节点出发进行随机模拟至终局，遵循必胜点优先落子规则。

        每次落子后切换上家/下家身份，动态更新必胜点列表。
        全程操作在副本上进行，不修改节点自身的任何数据。

        返回胜者编号（1/2/3），或 0 表示平局。
        """
        # ---- 复制状态，避免修改 self ----
        board = self.board.copy()
        turn = self.turn
        Key_Points: Dict[int, List[Tuple[int, int]]] = {
            pid: pts.copy() for pid, pts in self.Key_Points.items()
        }

        # ---- 步骤1：检查上一步落子是否已获胜（仅非 root 节点） ----
        if self.move_idx >= 0:
            last_player = self.up_player
            won, new_key_points = check_win_at(
                board, self.move_idx, last_player, potential_win=True
            )
            if won:
                return last_player

            # 更新必胜点：移除被上一步占据的，加入新产生的
            for pid in Key_Points:
                Key_Points[pid] = [
                    pt for pt in Key_Points[pid] if pt != self.move
                ]
            if new_key_points:
                for new_key_point in new_key_points:
                    key = (new_key_point[0], new_key_point[1])
                    if key not in Key_Points[last_player]:
                        Key_Points[last_player].append(key)

        # ---- 步骤2：模拟循环 ----
        while True:
            # 2a. 按必胜点规则获取所有候选落子，随机选一个
            candidates = _AB_Node._search_all_moves(board, turn, Key_Points)
            if not candidates:
                return 0  # 无可落子，平局
            row, col = random.choice(candidates)
            move_idx = pos_4d_to_index(*pos_2d_to_4d(row, col))

            # 2c. 落子
            board[move_idx] = turn

            # 2d. 检测胜负与必胜点
            won, new_key_points = check_win_at(
                board, move_idx, turn, potential_win=True
            )
            if won:
                return turn

            # 2e. 更新必胜点：移除被占据的，加入新产生的
            move_2d = _AB_Node._index_to_2d(move_idx)
            for pid in Key_Points:
                Key_Points[pid] = [
                    pt for pt in Key_Points[pid] if pt != move_2d
                ]
            if new_key_points:
                for new_key_point in new_key_points:
                    key = (new_key_point[0], new_key_point[1])
                    if key not in Key_Points[turn % 3 + 1] \
                    and key not in Key_Points[(turn + 1) % 3]:
                        # 如果新增的必胜点不是上下家的必胜点，则加入当前玩家的必胜点列表
                        Key_Points[turn].append(key)

            # 2f. 切换到下一玩家
            turn = turn % 3 + 1
   
    @staticmethod
    def _index_to_2d(idx: int) -> Tuple[int, int]:
        """一维索引 → (row, col)。内联避免重复代码。"""
        _w = idx // 64
        _r = idx % 64
        _x = _r // 16
        _r = _r % 16
        _y = _r // 4
        _z = _r % 4
        return (4 * _w + _x, 4 * _y + _z)

    def ucb1(self, player: int) -> float:
        """本节点在指定玩家视角下的 UCB1 值。未访问过 → 无穷大。"""
        if self.visits == 0:
            return float("inf")
        return (self.wins[player] / self.visits) + UCB_C * math.sqrt(
            math.log(self.parent.visits) / self.visits
        )

    def best_child(self, player: int) -> "_AB_Node":
        """返回该玩家视角下 UCB1 最高的子节点。"""
        return max(self.children, key=lambda c: c.ucb1(player))


# ============================================================================
# MCTS 搜索引擎
# ============================================================================

class ABMCTSEngine:
    """
    A(lphabet)B(eta) MCTS 搜索引擎。

    在标准 MCTS 基础上通过 point_to_try 实现剪枝：
    - 展开时：有必胜点时只展开必胜点分支
    - 模拟时：有必胜点时优先走必胜点

    player_id : AI 自己的玩家编号（1/2/3）
    time_limit: 每次思考时间上限（秒）
    max_iters : 每次搜索迭代上限
    """

    def __init__(self, player_id: int, time_limit: float = 5.0,
                 max_iters: int = 10000) -> None:
        self.player_id: int = player_id
        self.time_limit: float = time_limit
        self.max_iters: int = max_iters

    # ==========================================================================
    # 对外接口
    # ==========================================================================

    def choose_move(
        self, board: List[int], turn: int,
        last_moves: Dict[int, int] = None
    ) -> Tuple[Tuple[int, int], float, int]:
        """
        给定棋盘和当前回合，返回 AI 选择的落子与思考统计。

        last_moves: {player_id: 一维索引}，所有玩家最近一步棋。
                    用于 root 节点计算 Key_Point（必胜点优先探索）。

        返回: ((row, col), elapsed_time_seconds, iterations_count)
        """
        # ---- 计算 root 节点的 Key_Point ----
        key_points = self._cal_key_points(board, last_moves or {})

        root = _AB_Node(board, turn, key_points=key_points)

        t0 = time.time()
        actual_iters = 0
        for _ in range(self.max_iters):
            if time.time() - t0 > self.time_limit:
                break

            # 开始蒙特卡洛
            leaf = self._select(root)               # 1. Selection + Expansion
            winner = leaf._simulate()                # 2. Simulation
            self._backprop(leaf, winner)             # 3. Backpropagation
            actual_iters += 1

        if not root.children:
            raise RuntimeError("无可落子位置")

        best = max(root.children, key=lambda c: c.visits)
        elapsed = time.time() - t0
        print("[AB-MCTS] Best move:", best.move,
              "Visits:", best.visits, "Wins:", best.wins,
              "Time:", f"{elapsed:.3f}s", "Iters:", actual_iters)
        return best.move, elapsed, actual_iters

    # ==========================================================================
    # Selection & Expansion
    # ==========================================================================

    def _select(self, root: _AB_Node) -> _AB_Node:
        """选择下一个需要模拟的叶节点。

        仅在 root 层展开子节点（含必胜点剪枝），子节点不再继续展开。
        策略：
            1. 首次调用时展开 root 的所有子节点
            2. 优先随机探索未访问过的子节点
            3. 全部访问过之后用 UCB1 公式选择最优子节点
        """
        # 首次调用：展开 root 的所有子节点（利用必胜点剪枝）
        if not root.children:
            root.Make_All_Children()
            if not root.children:
                raise RuntimeError("无可落子位置")

        # 优先随机探索尚未访问过的子节点
        unvisited = [c for c in root.children if c.visits == 0]
        if unvisited:
            return random.choice(unvisited)

        # 所有子节点都至少访问过一次，用 UCB1 选最优
        return root.best_child(self.player_id)


    # ==========================================================================
    # Backpropagation
    # ==========================================================================

    @staticmethod
    def _backprop(node: _AB_Node, winner: int) -> None:
        """结果向上传播。"""
        while node is not None:
            node.visits += 1
            if winner != 0:
                node.wins[winner] += 1
            node = node.parent

    # ==========================================================================
    # 辅助
    # ==========================================================================

    def _cal_key_points(
        self, board: List[int], last_moves: Dict[int, int]
    ) -> Dict[int, List[Tuple[int, int]]]:
        """
        root 节点专用：根据各玩家最近落子位置，收集每个玩家自己的必胜点。

        对每个玩家自己的最后落子位置，用 check_win_at 检测该玩家是否
        形成了"三子一空"的必胜局面，收集空位坐标。

        返回:
            {player_id: [(row, col), ...]} — 各玩家当前可一步获胜的位置
        """
        result: Dict[int, List[Tuple[int, int]]] = {1: [], 2: [], 3: []}
        if not last_moves:
            return result

        seen: set = set()

        for pid, idx in last_moves.items():
            # 只检查 pid 自己：idx 上的棋子就是 pid 放的
            _won, pot_points = check_win_at(board, idx, pid,
                                            potential_win=True)
            if pot_points:
                for pt in pot_points:
                    key = (pt[0], pt[1])
                    if key not in seen:
                        seen.add(key)
                        result[pid].append(key)

        return result

    @staticmethod
    def _is_terminal(node: _AB_Node) -> bool:
        """检查节点是否为终局状态。"""
        if node.winner is not None:
            return True
        if not any(v == EMPTY for v in node.board):
            node.winner = 0
            return True
        return False