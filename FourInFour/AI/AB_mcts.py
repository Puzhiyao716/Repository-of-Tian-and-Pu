"""
Tian型 AI —— 蒙特卡洛树搜索 (MCTS) + UCB1 + Alphabet剪枝树
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
from typing import List, Tuple, Optional, Dict, Set

from FourInFour.GameCore import (
    EMPTY, TOTAL_CELLS,
    pos_4d_to_index, pos_4d_to_2d,
    pos_index_to_4d, check_win_at,
)

# UCB1 探索参数（√2 是理论最优值）
UCB_C = 1.414

# ---------------------------------------------------------------------------
# 预计算映射表：一维索引 → 四维坐标 (w, x, y, z)
# 避免每次需要时重复计算除法/取模
# ---------------------------------------------------------------------------
_INDEX_TO_4D: List[Tuple[int, int, int, int]] = [
    pos_index_to_4d(i) for i in range(TOTAL_CELLS)
]


# ============================================================================
# MCTS 节点
# ============================================================================

class _AB_Node:
    """MCTS 搜索树节点（带 Alphabet 剪枝）。"""

    def __init__(
        self,
        board: List[int],
        turn: int,
        move: Optional[Tuple[int, int, int, int]] = None,
        parent: Optional["_AB_Node"] = None,
        move_idx: int = -1,
        key_points: Optional[Dict[int, Set[Tuple[int, int, int, int]]]] = None,
    ) -> None:
        # ---- 棋盘状态 ----
        self.board: List[int] = board.copy()                     # 当前棋盘快照（256 长度）
        self.turn: int = turn                                     # 当前回合玩家编号（1/2/3）
        self.move: Optional[Tuple[int, int, int, int]] = move    # 到达此节点的落子 (w, x, y, z)
        self.move_idx: int = move_idx                             # 一维索引（root 为 -1）
        self.parent: Optional["_AB_Node"] = parent                # 父节点

        # ---- 关联玩家 ----
        self.up_player: int = (turn - 2) % 3 + 1                  # 上一回合玩家
        self.down_player: int = turn % 3 + 1                       # 下一回合玩家

        # ---- MCTS 统计 ----
        self.children: List["_AB_Node"] = []                       # 子节点列表
        self.visits: int = 0                                       # 被访问次数
        self.wins: Dict[int, int] = {1: 0, 2: 0, 3: 0}            # 各玩家视角的获胜次数
        self._untried: Optional[List[Tuple[int, int, int, int]]] = None  # 尚未尝试的落子（惰性初始化）
        self.winner: Optional[int] = None                          # 终局标记（None=未确定, 0=平局）

        # ---- 必胜点剪枝 ----
        # Key_Points: {player_id: {(w, x, y, z), ...}}
        # 记录各玩家当前可一步获胜的位置（Set 以 O(1) 去重和移除）
        self.Key_Points: Dict[int, Set[Tuple[int, int, int, int]]] = (
            {pid: set(pts) for pid, pts in key_points.items()}
            if key_points else {1: set(), 2: set(), 3: set()}
        )
        # ---- 节点状态 ----
        self.is_root = self.parent is None
        self.is_leaf = self.parent is not None 

    @classmethod
    def from_parent(
        cls, parent: "_AB_Node", Parent_move: Tuple[int, int, int, int]
    ) -> "_AB_Node":
        """由父节点扩展新节点，并更新必胜点状态。Parent_move 为四维坐标。"""
        assert parent.is_root == True ,\
            "父节点必须是根节点才能扩展新节点"
        
        # 四维 → 一维索引，用于 board 操作和 check_win_at
        Parent_move_idx = pos_4d_to_index(*Parent_move)

        new_board = list(parent.board)
        new_board[Parent_move_idx] = parent.turn

        # 深拷贝 Key_Points，移除已被占据的必胜点（Set.discard O(1)）
        KeyPoint: Dict[int, Set[Tuple[int, int, int, int]]] = {}
        for pid, pts in parent.Key_Points.items():
            KeyPoint[pid] = pts - {Parent_move}  # set 差集，移除被占据的

        next_turn = parent.turn % 3 + 1

        child = cls(
            board=new_board,
            turn=next_turn,
            move=Parent_move,
            parent=parent,
            move_idx=Parent_move_idx,
            key_points=KeyPoint,
        )
        return child

    def best_child(self, player: int) -> "_AB_Node":
        """返回该玩家视角下 UCB1 最高的子节点。"""
        return max(self.children, key=lambda c: c._ucb1(player))

    def Make_All_Children(self) -> None:
        """为当前节点生成所有可能的子节点（考虑必胜点）。"""
        candidates = self._search_all_moves(
            self.board, self.turn, self.Key_Points
        )
        random.shuffle(candidates)
        for move in candidates:
            child = _AB_Node.from_parent(self, move)
            self.children.append(child)
        if self.children:
            self.is_leaf = False

    @staticmethod
    def _check_win_and_renew_keypoint(
            board: List[int], 
            last_move_idx: int, 
            last_player: int,
            Key_Points: Dict[int, Set[Tuple[int, int, int, int]]]
    )->bool:
        """
        检查 last_player 是否已获胜，并更新必胜点列表。

        1. 调用 check_win_at 检测 last_player 是否在 last_move_idx 获胜
        2. 若获胜，返回 True（Key_Points 不更新，因为游戏已结束）
        3. 若未获胜，更新 Key_Points：
           - 从所有玩家的必胜点中移除被占据的位置（O(1) Set.discard）
           - 若产生了新的必胜点，归入 last_player（排除已被其他玩家占据的）

        返回 True 表示 last_player 获胜，False 表示未获胜。
        """
        # 将一维索引转换为四维坐标，用于 Key_Points 的 discard 操作
        move_4d = _INDEX_TO_4D[last_move_idx]

        # 检测 last_player 是否在 last_move_idx 处获胜
        won, new_key_points = check_win_at(
            board, last_move_idx, last_player, potential_win=True
        )
        if won:
            return True

        # 更新必胜点：移除被占据的位置
        for pid in Key_Points:
            Key_Points[pid].discard(move_4d)
        # 加入新产生的必胜点（归入 last_player）
        if new_key_points:
            for pt in new_key_points:  # pt 已是四维 (w, x, y, z)
                if pt not in Key_Points[last_player % 3 + 1] \
                and pt not in Key_Points[(last_player - 2) % 3 + 1]:
                    Key_Points[last_player].add(pt)

        return False
        
    def _simulate(self) -> int:
        """从当前节点出发进行随机模拟至终局，委托 Simulator 执行。

        返回胜者编号（1/2/3），或 0 表示平局。
        """
        return Simulator(
            board=self.board,
            turn=self.turn,
            key_points=self.Key_Points,
            last_player=self.up_player,
            last_move_idx=self.move_idx,
        ).run()
   
    def _ucb1(self, player: int) -> float:
        """本节点在指定玩家视角下的 UCB1 值。未访问过 → 无穷大。"""
        if self.visits == 0:
            return float("inf")
        return (self.wins[player] / self.visits) + UCB_C * math.sqrt(
            math.log(self.parent.visits) / self.visits
        )

    @staticmethod
    def _search_all_moves(
        board: List[int], turn: int,
        KeyPoint: Dict[int, Set[Tuple[int, int, int, int]]]
    ) -> List[Tuple[int, int, int, int]]:
        """
        根据必胜点规则搜索所有候选落子，统一以 (w, x, y, z) 格式返回。
        随机选择或进一步过滤由调用方负责。

        规则：
            1. 自己有必胜点 → 只返回必胜点
            2. 自己无，下家有且仅有一个 → 返回那一个
            3. 自己和下家无，上家有 2+ → 返回上家必胜点列表
            4. 否则 → 返回所有空位
        """
        down_player = turn % 3 + 1
        up_player = (turn - 2) % 3 + 1

        own_keypoint = KeyPoint.get(turn, set())
        down_keypoint = KeyPoint.get(down_player, set())
        up_keypoint = KeyPoint.get(up_player, set())

        # 规则 1：自己有必胜点
        if own_keypoint:
            return list(own_keypoint)

        # 规则 2：下家恰好一个必胜点，且上家不超过一个
        if len(down_keypoint) == 1 and len(up_keypoint) <= 1:
            return list(down_keypoint)

        # 规则 3：自己和下家无，上家 2+
        if len(down_keypoint) == 0 and len(up_keypoint) >= 2:
            return list(up_keypoint)

        # 规则 4：所有空位，使用预计算映射表转为四维坐标
        result = [(_INDEX_TO_4D[i]) for i,v in enumerate(board) if v == EMPTY]
        # result: List[Tuple[int, int, int, int]] = []
        # for i, v in enumerate(board):
        #     if v == EMPTY:
        #         result.append(_INDEX_TO_4D[i])
        return result


# ============================================================================
# 模拟器
# ============================================================================

class Simulator:
    """从给定状态出发进行随机模拟至终局，遵循必胜点优先落子规则。

    全程操作在副本上进行，不修改外部任何数据。
    内部维护空位集合，每步落子后增量更新，避免重复扫描整个棋盘。
    可独立于 _AB_Node 使用，便于测试和复用。
    """

    def __init__(
        self,
        board: List[int],
        turn: int,
        key_points: Dict[int, Set[Tuple[int, int, int, int]]],
        last_player: int,
        last_move_idx: int,
    ) -> None:
        """初始化模拟器。

        board:        当前棋盘快照（会被内部拷贝）
        turn:         当前回合玩家编号（1/2/3）
        key_points:   各玩家必胜点集合
        last_player:  上一手落子玩家编号
        last_move_idx:上一手落子的一维索引
        """
        self._board = board.copy()
        self._turn = turn
        self._key_points = {
            pid: set(pts) for pid, pts in key_points.items()
        }
        self._last_player = last_player
        self._last_move_idx = last_move_idx

        # 维护空位集合（一维索引），每步落子后增量更新，避免重复扫描 256 格棋盘
        self._empty_point: Set[int] = {
            i for i, v in enumerate(board) if v == EMPTY
        }
        print(f"[INFO TIAN] 模拟器初始化：空位数 {len(self._empty_point)}")

    # ------------------------------------------------------------------
    # 对外接口
    # ------------------------------------------------------------------

    def run(self) -> int:
        """执行模拟至终局，返回胜者编号（1/2/3）或 0 表示平局。"""
        while True:
            # 1. 检测上一手是否已获胜并更新必胜点
            if _AB_Node._check_win_and_renew_keypoint(
                self._board, self._last_move_idx,
                self._last_player, self._key_points
            ):
                return self._last_player

            # 2. 获取候选落子：优先必胜点策略，否则从空位集合随机选取
            priority = self._try_keypoint_moves(self._turn)
            if priority is not None:
                # 规则 1-3 命中：从 KeyPoint 列表中随机选
                w, x, y, z = random.choice(priority)
                move_idx = pos_4d_to_index(w, x, y, z)
            else:
                # 规则 4：从维护的空位集合中随机选取（O(1) 采样）
                if not self._empty_point:
                    # 棋盘满，平局
                    return 0
                else:
                    move_idx = random.choice(tuple(self._empty_point))

            # 3. 落子并更新空位集合
            self._board[move_idx] = self._turn
            self._empty_point.remove(move_idx)

            # 4. 为下一轮准备
            self._last_player = self._turn
            self._last_move_idx = move_idx
            self._turn = self._turn % 3 + 1

    # ------------------------------------------------------------------
    # 候选落子策略（替代 _search_all_moves）
    # ------------------------------------------------------------------

    def _try_keypoint_moves(
        self, turn: int
    ) -> Optional[List[Tuple[int, int, int, int]]]:
        """尝试按必胜点优先级获取候选落子，不命中时返回 None。

        规则（与 _search_all_moves 完全一致）：
        1. 自己有必胜点         → 返回自己的必胜点列表
        2. 下家恰好 1 个必胜点，且上家不超过一个必胜点  → 返回下家的那一个（堵截）
        3. 上家有 2+ 个必胜点   → 返回上家的必胜点列表（破坏）
        4. 否则                 → 返回 None，由调用方使用空位集合

        返回:
            命中规则时返回四维坐标列表，否则返回 None。
        """
        down = turn % 3 + 1
        up = (turn - 2) % 3 + 1
        own_kp = self._key_points[turn]
        down_kp = self._key_points[down]
        up_kp = self._key_points[up]

        # 规则 1：自己有必胜点
        if own_kp:
            return list(own_kp)

        # 规则 2：下家恰好一个必胜点，且上家不超过一个
        if len(down_kp) == 1 and len(up_kp) <= 1:
            return list(down_kp)

        # 规则 3：自己和下家无必胜点，上家 2+
        if len(down_kp) == 0 and len(up_kp) >= 2:
            return list(up_kp)

        # 规则 4：不适用，交给空位集合处理
        return None


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

    def engine_move(
        self, board: List[int], turn: int,
        three_last_moves: Dict[int, int] = None
    ) -> Tuple[Tuple[int, int, int, int], float, int]:
        """
        给定棋盘和当前回合，返回 AI 选择的落子与思考统计。

        last_moves: {player_id: 一维索引}，所有玩家最近一步棋。
                    用于 root 节点计算 Key_Point（必胜点优先探索）。

        返回: ((w, x, y, z), elapsed_time_seconds, iterations_count)
        """
        # ---- 计算 root 节点的 Key_Point ----
        key_points = self._cal_key_points(board, three_last_moves or {})

        # ---- 输出初始信息 ----
        total_kp = sum(len(v) for v in key_points.values())
        print(f"[INFO TIAN] player{self.player_id}开始搜索：")
        print(f"[INFO TIAN] 总共{total_kp}个必胜点", \
              *(f"player{p}:{len(kp)}" for p, kp in key_points.items()))

        root = _AB_Node(board, turn, key_points=key_points)

        # ---- 展开 root 的所有子节点（利用必胜点剪枝，只需一次） ----
        root.Make_All_Children()
        print(f"[INFO TIAN] root 层展开 {len(root.children)} 个子节点")
        if not root.children:
            raise RuntimeError("无可落子位置")

        # ---- 短路：唯一子节点直接返回，无需模拟 ----
        if len(root.children) == 1:
            only_child = root.children[0]
            print(f"[INFO TIAN] 唯一候选落子: {only_child.move}，直接返回（跳过模拟）")
            print()
            return only_child.move, 0.0, 1

        t0 = time.perf_counter()
        actual_iters = 0
        # 每 N 轮检查一次时间，减少系统调用
        _TIME_CHECK_INTERVAL = 20
        for _ in range(self.max_iters):
            actual_iters += 1
            if actual_iters % _TIME_CHECK_INTERVAL == 0:
                if time.perf_counter() - t0 > self.time_limit:
                    break

            # 1. Selection + Expansion：选择叶节点
            #    优先随机探索未访问过的子节点，全部访问过后用 UCB1 选最优
            unvisited = [c for c in root.children if c.visits == 0]
            if unvisited:
                leaf = random.choice(unvisited)
            else:
                leaf = root.best_child(self.player_id)

            score = leaf._simulate()                # 2. Simulation
            self._backprop(leaf, score)             # 3. Backpropagation

        best = max(root.children, key=lambda c: c.visits)
        elapsed = time.perf_counter() - t0
        # ---- 输出最终决策 (best.move 是四维坐标，日志中转为 2D 显示) ----
        print(f"[INFO TIAN] ====== 搜索完成 ======")
        print(f"[INFO TIAN] 总迭代: {actual_iters}, 耗时: {elapsed:.3f}s")
        print(f"[INFO TIAN] 最佳落子: {best.move}, "
              f"Visits: {best.visits}, "
              f"胜率: P1={best.wins[1]/best.visits:.3f} "
              f"P2={best.wins[2]/best.visits:.3f} "
              f"P3={best.wins[3]/best.visits:.3f}")
        print()
        return best.move, elapsed, actual_iters

    # # [DEPRECATED] _select 已内联到 engine_move 中，请勿使用
    # def _select(self, root: _AB_Node) -> _AB_Node:
    #     """选择下一个需要模拟的叶节点。

    #     仅在 root 层展开子节点（含必胜点剪枝），子节点不再继续展开。
    #     策略：
    #         1. 首次调用时展开 root 的所有子节点
    #         2. 优先随机探索未访问过的子节点
    #         3. 全部访问过之后用 UCB1 公式选择最优子节点
    #     """
    #     # 首次调用：展开 root 的所有子节点（利用必胜点剪枝）
    #     if not root.children:
    #         root.Make_All_Children()
    #         print(f"[INFO TIAN] root 层展开 {len(root.children)} 个子节点")
    #         if not root.children:
    #             raise RuntimeError("无可落子位置")

    #     # 优先随机探索尚未访问过的子节点
    #     unvisited = [c for c in root.children if c.visits == 0]
    #     if unvisited:
    #         return random.choice(unvisited)

    #     # 所有子节点都至少访问过一次，用 UCB1 选最优
    #     return root.best_child(self.player_id)


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
        self, board: List[int], three_last_moves: Dict[int, int]
    ) -> Dict[int, Set[Tuple[int, int, int, int]]]:
        """
        root 节点专用：根据各玩家最近落子位置，收集每个玩家自己的必胜点。

        对每个玩家自己的最后落子位置，用 check_win_at 检测该玩家是否
        形成了"三子一空"的必胜局面，收集空位坐标（四维格式）。

        返回:
            {player_id: {(w, x, y, z), ...}} — 各玩家当前可一步获胜的位置
        """
        result: Dict[int, Set[Tuple[int, int, int, int]]] = {
            1: set(), 2: set(), 3: set()
        }
        if len(three_last_moves) < 3:
            return result

        seen: set = set()

        for i in range(2, 5):
            playerid = (self.player_id + i) % 3 + 1
            idx = three_last_moves[playerid]
            # 只检查 pid 自己：idx 上的棋子就是 pid 放的
            _won, pot_points = check_win_at(board = board, 
                                            last_move_index = idx, 
                                            player = playerid,
                                            potential_win=True)
            if pot_points:
                for pt in pot_points:  # pt 已是四维 (w, x, y, z)
                    if pt not in seen:
                        seen.add(pt)
                        result[playerid].add(pt)

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