# -*- coding: utf-8 -*-
"""Federated BPR matrix factorisation (FedMF-style) with per-client clipping
and the private low-rank release of adaptrec.dp (revised Sections 5-6).

Each user is a client. User embeddings p_u stay on the client; the item
embedding matrix Q (n_items x d) is shared. In round t every participating
client draws B (positive, negative) pairs from its own training items,
updates p_u locally, and sends its item-gradient G_u. Because every row of
G_u is a multiple of p_u, G_u = c_u p_u^T is rank one, so its Frobenius norm
is ||c_u|| ||p_u|| and the clipped aggregate is
    G = C_mat^T (s * P),  s_u = min(1, C / (||c_u|| ||p_u||)),
computed with one sparse product (no per-client loop).
"""
import os
import numpy as np
import scipy.sparse as sp

from . import dp

DATA = os.environ.get("REC_DATA", os.path.expanduser("~/data"))


# ---------------------------------------------------------------------------
# data
# ---------------------------------------------------------------------------
def _lightgcn(name):
    def read(path):
        U, I = [], []
        with open(path) as f:
            for line in f:
                xs = line.split()
                if len(xs) < 2:
                    continue
                u = int(xs[0])
                for i in xs[1:]:
                    U.append(u); I.append(int(i))
        return np.asarray(U, np.int64), np.asarray(I, np.int64)
    tr = read(os.path.join(DATA, name + ".train.txt"))
    te = read(os.path.join(DATA, name + ".test.txt"))
    return tr, te


def _ml1m():
    def read(path):
        a = np.loadtxt(path, dtype=np.int64, usecols=(0, 1))
        return a[:, 0], a[:, 1]
    return read(os.path.join(DATA, "ml-1m.train.rating")), \
        read(os.path.join(DATA, "ml-1m.test.rating"))


def load(name):
    """Returns dict(train=(u,i), test=(u,i), n_user, n_item)."""
    (tu, ti), (eu, ei) = _ml1m() if name == "ml-1m" else _lightgcn(name)
    n_user = int(max(tu.max(), eu.max())) + 1
    n_item = int(max(ti.max(), ei.max())) + 1
    return dict(name=name, train=(tu, ti), test=(eu, ei),
                n_user=n_user, n_item=n_item)


class Data:
    def __init__(self, name):
        d = load(name)
        self.name = name
        self.n_user, self.n_item = d["n_user"], d["n_item"]
        tu, ti = d["train"]
        self.R = sp.csr_matrix((np.ones(len(tu), np.float32), (tu, ti)),
                               shape=(self.n_user, self.n_item))
        self.R.sum_duplicates(); self.R.data[:] = 1.0
        eu, ei = d["test"]
        self.T = sp.csr_matrix((np.ones(len(eu), np.float32), (eu, ei)),
                               shape=(self.n_user, self.n_item))
        self.T.sum_duplicates(); self.T.data[:] = 1.0
        self.n_train = self.R.nnz
        self.deg = np.diff(self.R.indptr)
        self.pop = np.asarray(self.R.sum(0)).ravel()

    def sample(self, users, B, rng):
        """B (pos, neg) pairs per user (users with >=1 training item)."""
        deg = self.deg[users]
        users = users[deg > 0]; deg = deg[deg > 0]
        uu = np.repeat(users, B)
        off = (rng.random(len(uu)) * np.repeat(deg, B)).astype(np.int64)
        pos = self.R.indices[self.R.indptr[uu] + off]
        neg = rng.integers(0, self.n_item, len(uu))
        return uu, pos, neg


# ---------------------------------------------------------------------------
# evaluation
# ---------------------------------------------------------------------------
def evaluate(data, P, Q, K=20, users=None, batch=None):
    """Full-ranking Recall@K and NDCG@K on the test split (train items masked)."""
    test_users = np.where(np.diff(data.T.indptr) > 0)[0]
    if users is not None:
        test_users = np.intersect1d(test_users, users)
    disc = 1.0 / np.log2(np.arange(2, K + 2))
    batch = batch or max(64, int(2.5e8 // (8 * data.n_item)))
    rec, ndcg = [], []
    for b0 in range(0, len(test_users), batch):
        us = test_users[b0:b0 + batch]
        S = P[us] @ Q.T
        Rb = data.R[us]
        S[Rb.nonzero()] = -np.inf
        top = np.argpartition(-S, K, axis=1)[:, :K]
        top = np.take_along_axis(top, np.argsort(-np.take_along_axis(S, top, 1), 1), 1)
        Tb = data.T[us]
        hit = np.asarray(Tb[np.arange(len(us))[:, None], top].todense()) > 0
        n_rel = np.diff(Tb.indptr)
        rec.append(hit.sum(1) / n_rel)
        idcg = np.array([disc[:min(n, K)].sum() for n in n_rel])
        ndcg.append((hit * disc).sum(1) / idcg)
    return float(np.concatenate(rec).mean()), float(np.concatenate(ndcg).mean())


def popularity(data, K=20):
    n = data.n_user
    P = np.ones((n, 1), np.float32)
    Q = data.pop[:, None].astype(np.float32)
    return evaluate(data, P, Q, K)


# ---------------------------------------------------------------------------
# federated training
# ---------------------------------------------------------------------------
def client_round(data, P, Q, users, B, lr_user, reg, rng):
    """Local step on every participating client; returns the per-pair
    coefficient vector needed to form the item gradients."""
    uu, pos, neg = data.sample(users, B, rng)
    pu, qi, qj = P[uu], Q[pos], Q[neg]
    x = np.einsum("bd,bd->b", pu, qi - qj)
    g = 1.0 / (1.0 + np.exp(np.clip(x, -30, 30)))       # d log sigma(x) / dx
    # client-side update of p_u (local, never transmitted)
    dP = g[:, None] * (qi - qj)
    np.add.at(P, uu, lr_user * (dP - reg * pu))
    # item-gradient coefficients: row i gets +g p_u, row j gets -g p_u
    rows = np.concatenate([uu, uu])
    cols = np.concatenate([pos, neg])
    vals = np.concatenate([g, -g]).astype(np.float64)
    Cm = sp.csr_matrix((vals, (rows, cols)), shape=(data.n_user, data.n_item))
    Cm.sum_duplicates()
    return Cm


def aggregate(Cm, P, clip):
    """Clipped aggregate of rank-one client updates G_u = c_u p_u^T.
    Returns (G, fraction of clients clipped)."""
    cn = np.sqrt(np.asarray(Cm.multiply(Cm).sum(1)).ravel())
    pn = np.linalg.norm(P, axis=1)
    nrm = cn * pn
    active = nrm > 0
    s = np.ones(len(nrm))
    s[active] = np.minimum(1.0, clip / nrm[active])
    G = np.asarray(Cm.T @ (s[:, None] * P))             # n_item x d
    frac = float((s[active] < 1.0).mean()) if active.any() else 0.0
    return G, frac, int(active.sum())


def train(data, d=64, rounds=200, B=16, lr_user=0.05, lr_item=None,
          reg=1e-4, clip=1.0, rank=None, sigma=0.0, basis="release",
          seed=0, eval_every=0, record_G=False, frac_users=1.0,
          callback=None, horizon=1.0):
    """Run federated BPR-MF.

    rank=None, sigma=0 : non-private full-rank FedSGD (reference).
    rank=r, sigma>0    : private low-rank release every round
                         (basis "release" = post-processing of the previous
                          release, "random" = fresh public random basis).
    Returns dict with final metrics and optional per-round G matrices.
    """
    rng = np.random.default_rng(seed)
    P = (rng.standard_normal((data.n_user, d)) * 0.1).astype(np.float64)
    Q = (rng.standard_normal((data.n_item, d)) * 0.1).astype(np.float64)
    users_all = np.where(data.deg > 0)[0]
    lr_item = lr_item if lr_item is not None else 1.0
    n_pub = len(users_all)              # public client count (normaliser)
    G_hat_prev, hist, Gs, ranks = None, [], [], []
    s_prev, r_cur = None, (d // 4 if rank == "auto" else rank)
    pub = np.random.default_rng([seed, 99])               # public randomness
    for t in range(rounds):
        if frac_users < 1.0:
            m = rng.random(len(users_all)) < frac_users
            users = users_all[m]
        else:
            users = users_all
        Cm = client_round(data, P, Q, users, B, lr_user, reg, rng)
        G, fclip, n_act = aggregate(Cm, P, clip)
        if record_G:
            Gs.append(G.astype(np.float32))
        if rank is None:
            G_hat = G + (rng.normal(0, sigma, G.shape) if sigma > 0 else 0.0)
        else:
            if rank == "auto" and s_prev is not None:
                # Proposition 1 applied to energies estimated from the
                # previous PUBLIC release (e_j ~ s_j(A_prev)^2 - d1 sigma^2):
                # keep j iff e_j > d1 sigma^2 / horizon, where horizon is the
                # number of rounds over which SGD averages the noise
                # (horizon = 1 is the per-round reconstruction criterion);
                # probe one more direction when all are kept.
                thr = G.shape[0] * sigma ** 2
                keep = int((s_prev ** 2 - thr > thr / horizon).sum())
                r_cur = min(d, r_cur + 1) if keep >= r_cur else max(1, keep)
            V = (dp.basis_from_release(G_hat_prev, d, pub) if basis == "release"
                 else dp.random_basis(d, pub))
            A, G_hat = dp.private_release(G, V, r_cur, sigma, rng)
            s_prev = np.linalg.svd(A, compute_uv=False)
            ranks.append(int(r_cur))
        G_hat_prev = G_hat
        Q += lr_item * G_hat / n_pub - lr_item * reg * Q
        if callback is not None:
            callback(t, P, Q, G, G_hat)
        if eval_every and ((t + 1) % eval_every == 0 or t == rounds - 1):
            r_, n_ = evaluate(data, P, Q)
            hist.append({"round": t + 1, "recall20": r_, "ndcg20": n_,
                         "clipped_frac": fclip})
    rec, nd = evaluate(data, P, Q)
    return {"recall20": rec, "ndcg20": nd, "hist": hist, "ranks": ranks,
            "Gs": Gs if record_G else None, "P": P, "Q": Q}
