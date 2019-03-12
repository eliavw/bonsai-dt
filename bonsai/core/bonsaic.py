"""
This module defines the Bonsai class and its basic templates.
The Bonsai class implements splitting data and constructing decision rules.
User need to provide two additional functions to complete the Bonsai class:
- find_split()
- is_leaf()
"""
# Authors: Yubin Park <yubin.park@gmail.com>
# License: Apache License 2.0
import copy
import numpy as np
import json
from scipy.special import expit
import time

from bonsai.core._bonsaic import reorder, sketch, apply_tree
from bonsai.core._utils import (
    reconstruct_tree,
    get_canvas_dim,
    setup_canvas_na,
    setup_canvas,
    get_child_branch,
    tunnel,
    swap,
)


class Bonsai:
    """
    Bonsai Class.

    Attributes
    ----------
    tree_ind    np.ndarray, (n_nodes, 6), dtype = int
                Array that contains integer information for each node of the
                tree. Every row is corresponds to a node in the tree structure.

                I.e.:
                tree_ind[0] = [is_leaf, svar, missing, l_idx, r_idx, leaf_idx]
                    is_leaf     -1:     NA, not a leaf.
                                1:      A leaf
                    svar        -1:     NA, no split happens here
                                int:    Refers to attribute used in split
                    missing     0:      np.nan (missing value) is sent to left
                                1:      np.nan (missing value) is sent to right
                    l_idx       -1:     NA, no left child node
                                int:    index of left child node
                    r_idx       -1:     NA, no right child node
                                int:    index of right child node
                    leaf_idx    -1:     NA, node is not a leaf
                                int:    Leaf index, distinct from node index

    tree_val    np.ndarray, (n_nodes, 2), dtype = np.float
                Array that contains float information for each node of the
                tree. Every row is corresponds to a node in the tree structure.

                I.e.:
                tree_val[0] = [sval, out]
                    sval        -1:     NA, no split happens here
                                float:  Split value of this node
                    out         -1:     NA, not a leaf node => no prediction
                                float:  Predicted value by this node

    """

    def __init__(
        self,
        find_split,
        is_leaf,
        randomize_node=None,
        n_hist_max=256,
        subsample=1.0,
        random_state=None,
        z_type="M2",
    ):
        self.find_split = find_split  # user-defined
        self.is_leaf = is_leaf  # user-defined

        if randomize_node is not None:  # overwrite if necessary
            self.randomize_node = randomize_node

        self.n_hist_max = n_hist_max
        self.subsample = np.clip(subsample, 0.0, 1.0)
        self.random_state = random_state
        self.z_type = z_type

        self.leaves = []
        self.feature_importances_ = None
        self.n_features_ = 0
        self.tree_ind = np.zeros((1, 6), dtype=np.int)
        self.tree_val = np.zeros((1, 2), dtype=np.float)
        self.tree_ind_header = (
            "is_leaf",
            "svar",
            "missing",
            "l_idx",
            "r_idx",
            "leaf_idx",
        )
        self.tree_val_header = ("sval", "out")
        self.mask = None
        self.canvas_dim = None
        self.canvas_na = None
        self.canvas = None

        self.counts = None
        self.ratios = None
        return

    def fit(self, X, y, init_canvas=True):
        """Fit a tree to the data (X, y)."""

        n, m = X.shape
        X = X.astype(np.float, order="C", copy=True)
        y = y.astype(np.float, order="C", copy=True)
        if self.z_type == "M2":
            z = np.square(y)
        elif self.z_type == "Hessian":  # bernoulli hessian
            p = expit(y)
            z = p * (1.0 - p)
        else:
            z = np.zeros(n)

        if self.subsample < 1.0:
            np.random.seed(self.random_state)
            self.mask = np.random.rand(n) < self.subsample
            X = X[self.mask, :]
            y = y[self.mask]
            z = z[self.mask]
            n, m = X.shape
        else:
            self.mask = np.full(n, True, dtype=np.bool)

        self.n_features_ = m

        branches = [
            {
                "_id": "ROOT",
                "is_leaf": False,
                "depth": 0,
                "eqs": [],
                "i_start": 0,
                "i_end": n,
                "y": np.mean(y),
                "y_lst": [],
                "n_samples": n,
            }
        ]

        if init_canvas:
            self.canvas_dim = get_canvas_dim(X, self.n_hist_max)
            self.canvas_na = setup_canvas_na(self.canvas_dim.shape[0])
            self.canvas = setup_canvas(self.canvas_dim)

        self.leaves = []
        if self.canvas_dim is not None and self.canvas is not None:
            while len(branches) > 0:
                branches, leaves_new = self.grow_tree(X, y, z, branches)
                self.leaves += leaves_new

        # integer index for leaves (from 0 to len(leaves))
        for i, leaf in enumerate(self.leaves):
            leaf["index"] = i
        self.update_feature_importances()
        self.tree_ind, self.tree_val = reconstruct_tree(self.leaves)
        return

    def predict(self, X, output_type="response"):
        """Predict y by applying the trained tree to X."""
        X = X.astype(np.float)
        n, m = X.shape
        y = np.zeros(n, dtype=np.float)
        out = apply_tree(self.tree_ind, self.tree_val, X, y, output_type)
        return out

    # Fitting
    def split_branch(self, X, y, z, branch):
        """Splits the data (X, y) into two children based on 
           the selected splitting variable and value pair.
        """

        i_start = branch["i_start"]
        i_end = branch["i_end"]

        # Get AVC-GROUP
        avc = sketch(
            X, y, z, self.canvas, self.canvas_dim, self.canvas_na, i_start, i_end
        )

        if avc.shape[0] < 2:
            branch["is_leaf"] = True
            return [branch]

        # Find a split SS: selected split
        ss = self.find_split(avc)
        if not isinstance(ss, dict) or "selected" not in ss:
            branch["is_leaf"] = True
            return [branch]

        svar = ss["selected"][1]
        sval = ss["selected"][2]
        missing = ss["selected"][9]
        i_split = reorder(X, y, z, i_start, i_end, svar, sval, missing)

        if i_split == i_start or i_split == i_end:
            # NOTE: this condition may rarely happen due to
            #       Python's floating point treatments.
            #       We just ignore this case, and stop the tree growth
            branch["is_leaf"] = True
            return [branch]

        left_branch = get_child_branch(ss, branch, i_split, "@l")
        left_branch["is_leaf"] = self.is_leaf(left_branch, branch)

        right_branch = get_child_branch(ss, branch, i_split, "@r")
        right_branch["is_leaf"] = self.is_leaf(right_branch, branch)

        return [left_branch, right_branch]

    def grow_tree(self, X, y, z, branches):
        """Grows a tree by recursively partitioning the data (X, y)."""

        branches_new = []
        leaves_new = []
        for branch in branches:
            for child in self.split_branch(X, y, z, branch):
                if child["is_leaf"]:
                    leaves_new.append(child)
                else:
                    branches_new.append(child)
        return branches_new, leaves_new

    # Canvas
    def init_canvas(self, X):
        self.canvas_dim = get_canvas_dim(X, self.n_hist_max)
        self.canvas = setup_canvas(self.canvas_dim)
        self.canvas_na = setup_canvas_na(self.canvas_dim.shape[0])

    def set_canvas(self, canvas_dim, canvas):
        self.canvas_dim = canvas_dim
        self.canvas = canvas
        self.canvas_na = setup_canvas_na(self.canvas_dim.shape[0])

    def get_canvas(self):
        return self.canvas_dim, self.canvas

    def is_stochastic(self):
        return self.subsample < 1.0

    def get_mask(self):
        return self.mask

    def get_oob_mask(self):
        """Returns a mask array for OOB samples"""
        return ~self.mask

    def get_ttab(self):
        """Returns tree tables (ttab). 
            ttab consists of tree_ind (np_array) and tree_val (np_array).
            tree_ind stores tree indices - integer array. 
            tree_val stores node values - float array.
        """
        return self.tree_ind, self.tree_val

    # Load-Save
    def dump(self, columns=None):
        """Dumps the trained tree in the form of array of leaves"""

        if columns is None:
            columns = []

        def default(o):
            if isinstance(o, np.int64):
                return int(o)
            raise TypeError

        n_col = len(columns)
        for leaf in self.leaves:
            for eq in leaf["eqs"]:
                if eq["svar"] < n_col:
                    eq["name"] = columns[int(eq["svar"])]
        return json.loads(json.dumps(self.leaves, default=default))

    def load(self, leaves):
        """Loads a new tree in the form of array of leaves"""
        self.leaves = leaves
        self.tree_ind, self.tree_val = reconstruct_tree(self.leaves)
        return None

    def get_sibling_id(self, leaf_id):
        """Returns a sibling ID for the given leaf_id.
           Siblings are the nodes that are at the same level 
            with the same parent node.
        """
        sibling_id = None
        if leaf_id[-1] == "L":
            sibling_id = leaf_id[:-1] + "R"
        elif leaf_id[-1] == "R":
            sibling_id = leaf_id[:-1] + "L"
        sibling_leaf = [leaf for leaf in self.leaves if leaf["_id"] == sibling_id]
        if len(sibling_leaf) == 0:
            sibling_id = None
        return sibling_id

    def get_sibling_pairs(self):
        """Returns an array of sibling pairs. 
            For more info, see the get_sibling_id
        """
        id2index = {leaf["_id"]: i for i, leaf in enumerate(self.leaves)}
        leaf_ids = [k for k in id2index.keys()]
        sibling_pairs = []
        while len(leaf_ids) > 0:
            leaf_id = leaf_ids.pop()
            sibling_id = self.get_sibling_id(leaf_id)
            if sibling_id is not None:
                if sibling_id in leaf_ids:
                    leaf_ids.remove(sibling_id)
                sibling_pairs.append((id2index[leaf_id], id2index[sibling_id]))
            else:
                sibling_pairs.append((id2index[leaf_id], None))
        return sibling_pairs

    # Calculate quantities
    def get_feature_importances(self):
        return self.feature_importances_

    def update_feature_importances(self):
        """Returns a modified feature importance.
            This formula takes into account of node coverage and leaf value.
            NOTE: This is different from regular feature importances that
                are used in RandomForests or GBM.
            For more info, please see the PaloBoost paper.
        """
        if self.n_features_ == 0:
            return None
        self.feature_importances_ = np.zeros(self.n_features_)
        cov = 0
        J = len(self.leaves)
        for j, leaf in enumerate(self.leaves):
            gamma_j = np.abs(leaf["y"])
            cov_j = leaf["n_samples"]
            cov += cov_j
            eff_j = cov_j * gamma_j
            for eq in leaf["eqs"]:
                self.feature_importances_[eq["svar"]] += eff_j
        self.feature_importances_ /= J
        self.feature_importances_ /= cov
        return self.feature_importances_

    def get_counts(self, persist=True):
        """
        See how many samples are sent left and right by a node.

        Returns
        -------

        """

        n, _ = self.tree_ind.shape
        counts = np.zeros((n, 3), dtype=int)

        for node_idx in range(n):
            leaf = self.check_leaf(node_idx)

            if leaf:
                counts[node_idx, 0] = self.count_samples_node(node_idx)
                counts[node_idx, 1] = -1
                counts[node_idx, 2] = -1
            else:
                counts[node_idx, 1] = self.count_samples_node(
                    self.tree_ind[node_idx][3]
                )
                counts[node_idx, 2] = self.count_samples_node(
                    self.tree_ind[node_idx][4]
                )
                counts[node_idx, 0] = counts[node_idx, 1] + counts[node_idx, 2]

        if persist:
            self.counts = counts
        return counts

    def get_ratios(self, persist=True):
        """
        Calculate the ratio between samples sent left and right by a dnode

        Returns
        -------

        """

        n, _ = self.tree_ind.shape
        ratios = np.zeros((n, 2), dtype=float)

        assert self.counts is not None  # You first have to have counts.

        for node_idx in range(n):
            leaf = self.check_leaf(node_idx)
            counts = self.counts[node_idx]
            n_samples = counts[0]

            if leaf:
                ratios[node_idx, 0] = -1
                ratios[node_idx, 1] = -1
            else:
                ratios[node_idx, 0] = counts[1] / n_samples
                ratios[node_idx, 1] = counts[2] / n_samples

        if persist:
            self.ratios = ratios
        return ratios

    # Randomize tree
    def randomize_tree(self, kind="swap"):
        if kind in {"swap"}:
            return self.swap_randomization()
        elif kind in {"tunnel"}:
            return self.tunnel_randomization()
        else:
            msg = """
            Did not recognize randomization scheme: {}
            """.format(kind)
            ValueError(msg)

    def tunnel_randomization(self):
        if self.counts is None:
            self.get_counts()
            self.get_ratios()
        tree = copy.deepcopy(self)
        n, _ = tree.tree_ind.shape

        # Filter leaves
        internal_nodes = [node_idx for node_idx in range(n)
                          if tree.tree_ind[node_idx][0] != 1]

        l_ratios = tree.ratios[:, 0]
        r_ratios = tree.ratios[:, 1]
        relative_counts = tree.counts[:, 0]/tree.counts[0, 0]

        l_random_samples = self.tunnel_distribution_samples(n)
        r_random_samples = self.tunnel_distribution_samples(n)

        for node_idx in internal_nodes:
            l2r_tunnel = r_ratios[node_idx] * (1 - 0.5*relative_counts[node_idx])
            r2l_tunnel = l_ratios[node_idx] * (1 - 0.5*relative_counts[node_idx])

            l2r_tunnel = l_random_samples[node_idx] < l2r_tunnel
            r2l_tunnel = r_random_samples[node_idx] < r2l_tunnel

            if l2r_tunnel and r2l_tunnel:
                # This is a swap
                tree.tree_ind[node_idx] = tunnel(tree.tree_ind[node_idx], direction=-1)
            elif l2r_tunnel:
                tree.tree_ind[node_idx] = tunnel(tree.tree_ind[node_idx], direction=0)
            elif r2l_tunnel:
                tree.tree_ind[node_idx] = tunnel(tree.tree_ind[node_idx], direction=1)
            else:
                pass

        return tree

    def swap_randomization(self):
        if self.counts is None:
            self.get_counts()
            self.get_ratios()
        tree = copy.deepcopy(self)
        n, _ = tree.tree_ind.shape
        random_samples = self.swap_distribution_samples(n)

        # Filter leaves
        internal_nodes = [node_idx for node_idx in range(n)
                          if tree.tree_ind[node_idx][0] != 1]

        l_ratios = tree.ratios[:, 0]

        # Iterate over all nodes, swap if you must
        for node_idx in internal_nodes:
            s = random_samples[node_idx]
            d = 0.5 - l_ratios[node_idx]
            must_swap = np.abs(s) > np.abs(d)

            if must_swap:
                tree.tree_ind[node_idx] = swap(tree.tree_ind[node_idx])

        return tree

    # Helpers
    def check_leaf(self, node_idx):
        return self.tree_ind[node_idx][0] == 1

    def count_samples_node(self, node_idx):
        """
        Count samples for specific node in the tree.
        """

        tree_ind = self.tree_ind
        leaves = self.leaves

        def is_leaf(node_idx):
            return tree_ind[node_idx][0] == 1

        if is_leaf(node_idx):
            leaf_idx = tree_ind[node_idx][5]
            leaf = leaves[leaf_idx]
            return leaf["n_samples"]

        else:
            l_idx = tree_ind[node_idx][3]
            r_idx = tree_ind[node_idx][4]
            return self.count_samples_node(l_idx) + self.count_samples_node(r_idx)

    @staticmethod
    def swap_distribution_samples(n, mu=0., sigma=0.2):
        random_samples = np.random.normal(mu, sigma, n)
        #random_samples = np.round(random_samples, decimals=2)
        return random_samples

    @staticmethod
    def tunnel_distribution_samples(n):
        return np.random.rand(n)
