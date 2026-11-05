# -*- coding: utf-8 -*-
"""Bosqich bo'yicha resurs profilini O'LCHASH."""
import sys, os, json, collections
import numpy as np
import os as _os, sys as _sys
# 1-maqola paketidagi real/ katalogi (Oqim, komponentlar) shu yerdan olinadi.
_M1 = _os.environ.get("MAQOLA1_REAL") or _os.path.abspath(
    _os.path.join(_os.path.dirname(_os.path.abspath(__file__)),
                  "..", "..", "Maqola1_kodlari", "real"))
if not _os.path.isdir(_os.path.join(_M1, "adaptreal")):
    raise SystemExit("1-maqola real/ katalogi topilmadi: " + _M1 +
                     "  (MAQOLA1_REAL muhit o'zgaruvchisi bilan ko'rsating)")
_sys.path.insert(0, _M1)
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from adaptreal.data import Oqim
from adaptreal.components import (Ommaboplik, ItemKNN, Kontent, P3Alpha,
                                  Ketmaket, CrossEncoderRerank)
from adaptreal2.bosqichlar import Hisob, NOMLAR

WARM = int(os.environ.get("WARM", "100000"))
N_SOROV = int(os.environ.get("N_SOROV", "400"))
M5_HAR = int(os.environ.get("M5_HAR", "1"))
MODEL = os.environ.get("MINILM", "/data/minilm")

s = Oqim(); md = s.meta
titles = [(md.get(j, ("", "", ""))[0] or "item") for j in range(s.n_item)]
pop = Ommaboplik(s.n_item); knn = ItemKNN(s.n_item); kon = Kontent(s.matn())
gr = P3Alpha(s.n_item); seq = Ketmaket(s.n_item)
m5 = CrossEncoderRerank(s.n_item, titles, MODEL, top_k=20, max_len=32, backend="onnx")

print("holat qurilmoqda...", flush=True)
hist = collections.defaultdict(list); oxirgi = {}
for e in range(WARM):
    u = int(s.u[e]); i = int(s.i[e]); h = hist[u]
    knn.yangilash(u, i, 0, oldingi=h); seq.yangilash(u, i, 0, oxirgi=oxirgi.get(u))
    gr.yangilash(u, i, 0); pop.yangilash(u, i, 0)
    h.append(i); oxirgi[u] = i
print("tayyor, o'lchov boshlanadi", flush=True)

H = Hisob()
F4 = 4  # float32/int32 bayt

def norm(x):
    m = x.max(); return x / m if m > 0 else x

n_done = 0
e = WARM
while n_done < N_SOROV and e < len(s):
    u = int(s.u[e]); i = int(s.i[e]); h = hist[u]
    if not h:
        e += 1; continue

    # 0 ingestion: hodisa yozuvi (user, item, rating, ts, verified, helpful)
    hodisa, b = H.qadam("ingestion", lambda: (
        {"u": u, "i": i, "t": int(s.t[e])}, 3 * 8 + 24), kirish_bayt=64)

    # 1 feature: tarix va holat vektori
    def _feat():
        hn = np.asarray(h[-20:], np.int32)
        holat = np.array([1.0, len(h), 0.0, 1.0, 0.5, 1.0], np.float32)
        return (hn, holat), hn.nbytes + holat.nbytes
    (hn, holat), b = H.qadam("feature", _feat, kirish_bayt=b)

    # 2 retrieval: to'liq katalog nomzod sifatida (arzon signal)
    def _ret():
        sc = pop.baholash(u, h)
        return sc, s.n_item * F4
    sc_pop, b = H.qadam("retrieval", _ret, kirish_bayt=b)

    # 3 pre-ranking: arzon skorerlar
    def _pre():
        a = knn.baholash(u, h); c = kon.baholash(u, h)
        return (a, c), 2 * s.n_item * F4
    (sc_knn, sc_kon), b = H.qadam("pre_ranking", _pre, kirish_bayt=b)

    # 4 ranking: asosiy skorerlar va fuzya
    def _rank():
        g = gr.baholash(u, h); q = seq.baholash(u, h)
        asos = norm(sc_knn) + norm(g) + norm(q)
        top = np.argpartition(-asos, 19)[:20]
        return (asos, top), 20 * (F4 + F4)
    (asos, top), b = H.qadam("ranking", _rank, kirish_bayt=b)

    # 5 re-ranking: cross-encoder (matn ham chegaradan o'tadi)
    if n_done % M5_HAR == 0:
        def _rr():
            sc5 = m5.baholash(u, h, asos)
            matn = sum(len(titles[int(c)][:120].encode()) for c in top)
            return sc5, 20 * F4 + matn
        _, b = H.qadam("re_ranking", _rr,
                       kirish_bayt=20 * F4 + sum(len(titles[int(c)][:120].encode()) for c in top))

    # 6 logging: qaror yozuvi
    def _log():
        rec = {"u": u, "top": [int(x) for x in top[:10]], "t": int(s.t[e])}
        return rec, len(json.dumps(rec).encode())
    rec, b = H.qadam("logging", _log, kirish_bayt=b)

    # 7 analytics: paketli agregatsiya (so'rov boshiga ulush)
    def _an():
        agg = {"kun": int(s.t[e]) // 86400000, "n": 1, "top1": int(top[0])}
        return agg, len(json.dumps(agg).encode())
    _, b = H.qadam("analytics", _an, kirish_bayt=b)

    knn.yangilash(u, i, 0, oldingi=h); seq.yangilash(u, i, 0, oxirgi=oxirgi.get(u))
    gr.yangilash(u, i, 0); pop.yangilash(u, i, 0)
    h.append(i); oxirgi[u] = i
    n_done += 1; e += 1

rep = H.hisobot(n_done)
out = {"n_sorov": n_done, "warm": WARM, "katalog": s.n_item,
       "m5_har": M5_HAR, "bosqichlar": rep}
os.makedirs("natijalar", exist_ok=True)
json.dump(out, open("natijalar/bosqich_profili.json", "w"), ensure_ascii=False, indent=1)
print("\n%-13s %-12s %-14s %-14s"%("bosqich","cpu ms/so'rov","cpu soat/1M","GB chiqish/1M"))
for n in NOMLAR:
    r = rep[n]
    print("%-13s %-12.4f %-14.3f %-14.3f"%(n, r["cpu_ms_sorov"], r["cpu_soat_1M_sorov"], r["gb_chiqish_1M_sorov"]))
