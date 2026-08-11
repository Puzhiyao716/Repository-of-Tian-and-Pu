"""
四维四子棋 - AI 算法包
====================

- MCTS + UCB1：稳健型 AI（mcts.py）
- MCTS + UCB1 + Alphabet 剪枝：智慧型 AI（AB_mcts.py）
"""
from .mcts import MCTSEngine
from .AB_mcts import ABMCTSEngine

__all__ = ["MCTSEngine", "ABMCTSEngine"]