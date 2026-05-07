import networkx as nx


class CVECAPECScoringSystem:
    def __init__(self):
        self.graph = nx.DiGraph()

        # Edge weights for the different structural relations.
        # Higher weight = stronger / more reliable signal.
        self.weights = {
            'cve_cwe_truth': 1.0,      # Verified ground-truth CVE -> CWE mapping
            'cwe_capec_direct': 1.0,   # Official CWE -> CAPEC mapping
            'upward_abstraction': 0.9, # Child -> Parent (generalization)
            'downward_specify': 0.6,   # Parent -> Child (specialization, more uncertain)
            'peer_of': 0.5,            # Sibling variant relationship
            'can_precede': 0.3,        # Temporal precedence (one step may lead to another)
        }

    def add_cve_cwe_edge(self, cve_id, cwe_id, is_ground_truth=False, bert_prob=0.0):
        """
        Add a CVE -> CWE edge.
        For V2W-BERT predictions, the predicted probability is used directly as the edge weight.
        """
        weight = self.weights['cve_cwe_truth'] if is_ground_truth else bert_prob
        if weight > 0:
            self.graph.add_edge(cve_id, cwe_id, weight=weight, rel_type='cve_to_cwe')

    def add_cwe_hierarchy(self, parent_cwe, child_cwe):
        """
        Add a CWE parent-child relationship as two directed edges with different weights.
        Going up (child -> parent) is more reliable than going down (parent -> child).
        """
        self.graph.add_edge(child_cwe, parent_cwe, weight=self.weights['upward_abstraction'], rel_type='cwe_upward')
        self.graph.add_edge(parent_cwe, child_cwe, weight=self.weights['downward_specify'], rel_type='cwe_downward')

    def add_cwe_capec_edge(self, cwe_id, capec_id):
        """Add a direct CWE -> CAPEC mapping edge."""
        self.graph.add_edge(cwe_id, capec_id, weight=self.weights['cwe_capec_direct'], rel_type='cwe_to_capec')

    def add_capec_relationship(self, capec_1, capec_2, rel_type):
        """Add an internal CAPEC-to-CAPEC relationship."""
        if rel_type == 'ChildOf':
            # capec_1 is a child of capec_2: 1 -> 2 is upward abstraction, 2 -> 1 is downward specialization
            self.graph.add_edge(capec_1, capec_2, weight=self.weights['upward_abstraction'], rel_type='capec_upward')
            self.graph.add_edge(capec_2, capec_1, weight=self.weights['downward_specify'], rel_type='capec_downward')
        elif rel_type == 'PeerOf':
            self.graph.add_edge(capec_1, capec_2, weight=self.weights['peer_of'], rel_type='capec_peer')
            self.graph.add_edge(capec_2, capec_1, weight=self.weights['peer_of'], rel_type='capec_peer')
        elif rel_type == 'CanPrecede':
            # capec_1 can occur before capec_2
            self.graph.add_edge(capec_1, capec_2, weight=self.weights['can_precede'], rel_type='capec_precede')

    def calculate_path_score(self, path):
        """Score for a single path is the product of all edge weights along it."""
        score = 1.0
        for i in range(len(path) - 1):
            u, v = path[i], path[i + 1]
            score *= self.graph[u][v]['weight']
        return score

    def evaluate_connection(self, cve_id, capec_id, max_depth=4):
        """
        Evaluate the structural connection probability between a CVE and a CAPEC.

        Multiple paths are aggregated with Noisy-OR: 1 - prod(1 - P_i).
        max_depth caps the search depth to prevent combinatorial explosion;
        depth 4 is enough to cover cve -> cwe_child -> cwe_parent -> capec.
        """
        if not self.graph.has_node(cve_id) or not self.graph.has_node(capec_id):
            return 0.0, []

        try:
            paths = list(nx.all_simple_paths(self.graph, source=cve_id, target=capec_id, cutoff=max_depth))
        except nx.NetworkXNoPath:
            return 0.0, []

        if not paths:
            return 0.0, []

        path_details = []
        noisy_or_complement = 1.0

        for p in paths:
            p_score = self.calculate_path_score(p)
            path_details.append({'path': p, 'score': p_score})
            noisy_or_complement *= (1.0 - p_score)

        final_score = 1.0 - noisy_or_complement

        # Sort by per-path score so the top contributing paths are easy to inspect
        path_details.sort(key=lambda x: x['score'], reverse=True)
        return final_score, path_details

    def get_top_k_capecs_for_cve(self, cve_id, k=5, max_depth=4):
        """Return the Top-K most likely CAPECs for a given CVE based on graph reasoning."""
        capec_nodes = [n for n, d in self.graph.out_degree() if 'CAPEC' in str(n)]
        results = []
        for capec in capec_nodes:
            score, _ = self.evaluate_connection(cve_id, capec, max_depth)
            if score > 0:
                results.append((capec, score))

        results.sort(key=lambda x: x[1], reverse=True)
        return results[:k]