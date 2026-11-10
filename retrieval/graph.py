import re
import networkx as nx


class DependencyGraph:
    def __init__(self):
        self.graph = nx.DiGraph()

    def build(self, chunks: list):
        self.graph.clear()
        for c in chunks:
            self.graph.add_node(c.id, chunk=c)

        # Map symbol -> list of chunk IDs (avoid overwriting colliding names like __init__)
        # and filter short symbols (< 3 chars) to prevent substring false edges.
        symbols: dict[str, list[str]] = {}
        for c in chunks:
            if c.symbol and c.symbol != "module" and len(c.symbol) >= 3:
                symbols.setdefault(c.symbol, []).append(c.id)

        if not symbols:
            return

        sym_re = re.compile(r"\b(" + "|".join(re.escape(s) for s in symbols) + r")\b")
        for c in chunks:
            for sym in set(sym_re.findall(c.content)):
                for t_id in symbols[sym]:
                    if t_id != c.id:
                        self.graph.add_edge(c.id, t_id)

    def symbol_score(self, query: str, chunk) -> float:
        score = 0.0
        if chunk.symbol and chunk.symbol.lower() in query.lower():
            score += 0.5
        return score

    def expand(self, chunks: list, max_extra: int = 4) -> list:
        seen_ids = {c.id for c in chunks}
        expanded = []
        for c in chunks:
            if c.id in self.graph:
                for n in list(self.graph.successors(c.id)) + list(self.graph.predecessors(c.id)):
                    if n not in seen_ids:
                        seen_ids.add(n)
                        expanded.append(n)
                        if len(expanded) >= max_extra:
                            break
            if len(expanded) >= max_extra:
                break

        return [
            self.graph.nodes[n_id]["chunk"]
            for n_id in expanded
            if self.graph.nodes[n_id].get("chunk") is not None
        ]
