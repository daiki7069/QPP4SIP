"""
NSPベースのcoherencyグラフ構築とNC/ANC計算
"""
from __future__ import annotations

import time
from dataclasses import dataclass
from typing import TYPE_CHECKING, List, Optional, Tuple

import networkx as nx
import numpy as np
import pandas as pd
from tqdm import tqdm

from .nsp import NSP

if TYPE_CHECKING:
    from ..post_retrieval.data_loader import TurnData  # pragma: no cover


@dataclass
class GraphMetrics:
    conv_id: str
    turn_id: str
    question: str
    num_docs: int
    num_edges: int
    node_connectivity: float
    average_node_connectivity: float
    density: float


class NSPGraphAnalyzer:
    """NSPによるcoherencyグラフを構築してNC/ANCを計算する"""

    def __init__(
        self,
        turn_data_list: List["TurnData"],
        model_name: str = "bert-base-uncased",
        device: Optional[str] = None,
        threshold: float = 0.5,
        top_k: int = 20,
        batch_size: int = 64,
        use_multi_gpu: bool = True,
    ):
        self.turn_data_list = turn_data_list
        self.threshold = threshold
        self.top_k = top_k
        self.nsp = NSP(
            model_name=model_name, 
            device=device,
            batch_size=batch_size,
            use_multi_gpu=use_multi_gpu
        )

    def analyze(self, show_progress: bool = True) -> pd.DataFrame:
        rows: List[GraphMetrics] = []
        iterator = self.turn_data_list
        total_turns = len(self.turn_data_list)
        
        if show_progress:
            iterator = tqdm(
                iterator, 
                desc="Processing turns", 
                unit="turn",
                total=total_turns,
                bar_format='{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}, {rate_fmt}]'
            )

        for turn_idx, turn in enumerate(iterator, 1):
            start_time = time.time()
            metrics = self._analyze_turn(turn, show_progress=show_progress, turn_idx=turn_idx, total_turns=total_turns)
            elapsed_time = time.time() - start_time
            
            if show_progress:
                # 進捗バーのpostfixに処理時間と詳細情報を表示
                iterator.set_postfix({
                    'time': f'{elapsed_time:.1f}s',
                    'docs': metrics.num_docs,
                    'pairs': metrics.num_docs * (metrics.num_docs - 1),
                    'edges': metrics.num_edges
                })
            
            rows.append(metrics)

        df = pd.DataFrame([metric.__dict__ for metric in rows])
        return df

    def _analyze_turn(
        self, 
        turn: "TurnData", 
        show_progress: bool = False,
        turn_idx: int = 0,
        total_turns: int = 0
    ) -> GraphMetrics:
        documents = turn.documents[: self.top_k]
        if len(documents) < 2:
            return GraphMetrics(
                conv_id=str(turn.conv_id),
                turn_id=str(turn.turn_id),
                question=turn.question,
                num_docs=len(documents),
                num_edges=0,
                node_connectivity=0.0,
                average_node_connectivity=0.0,
                density=0.0,
            )

        graph = nx.DiGraph()
        for doc in documents:
            graph.add_node(doc.id, title=doc.title, score=doc.score)

        sentence_pairs: List[Tuple[str, str]] = []
        pair_indices: List[Tuple[int, int]] = []
        for i, doc_i in enumerate(documents):
            for j, doc_j in enumerate(documents):
                if i == j:
                    continue
                sentence_pairs.append((doc_i.text, doc_j.text))
                pair_indices.append((i, j))

        # バッチ処理で高速化（進捗表示を有効化）
        # ターン情報を進捗バーの説明に含める
        progress_desc = None
        if show_progress and total_turns > 0:
            progress_desc = f"Turn {turn_idx}/{total_turns} (NSP)"
        
        predictions = self.nsp.predicts(sentence_pairs, show_progress=show_progress, desc=progress_desc)

        for (i, j), pred in zip(pair_indices, predictions):
            if pred["prediction"] == "IsNext" and pred["is_next_prob"] >= self.threshold:
                doc_i = documents[i]
                doc_j = documents[j]
                graph.add_edge(doc_i.id, doc_j.id, weight=pred["is_next_prob"])

        nc = self._compute_node_connectivity(graph)
        anc = self._compute_average_node_connectivity(graph)
        density = nx.density(graph) if graph.number_of_nodes() > 1 else 0.0

        return GraphMetrics(
            conv_id=str(turn.conv_id),
            turn_id=str(turn.turn_id),
            question=turn.question,
            num_docs=len(documents),
            num_edges=graph.number_of_edges(),
            node_connectivity=nc,
            average_node_connectivity=anc,
            density=density,
        )

    def _compute_node_connectivity(self, graph: nx.DiGraph) -> float:
        if graph.number_of_nodes() < 2 or graph.number_of_edges() == 0:
            return 0.0
        try:
            return float(nx.node_connectivity(graph))
        except nx.NetworkXError:
            return 0.0

    def _compute_average_node_connectivity(self, graph: nx.DiGraph) -> float:
        if graph.number_of_nodes() < 2 or graph.number_of_edges() == 0:
            return 0.0
        try:
            anc = nx.average_node_connectivity(graph)
            if isinstance(anc, np.ndarray):
                return float(np.mean(anc))
            return float(anc)
        except nx.NetworkXError:
            return 0.0

