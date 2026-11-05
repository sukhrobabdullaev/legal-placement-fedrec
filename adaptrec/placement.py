# -*- coding: utf-8 -*-
"""A7 — huquqiy toifa bo'yicha quvur bosqichlarini joylashtirish
(dissertatsiya, 3.4-bo'lim).

Chiziqli quvur (N bosqich) uchun har bir bosqichni qatlamlardan biriga
(masalan: lokal DPC, mahalliy bulut, tashqi bulut) joylashtirish masalasi
dinamik programmalash bilan O(N * |L|^2) vaqtda aniq yechiladi.
Amaliy o'lchamlarda (N=8, |L|=3) bu 72 ta amal — huquqiy rejim
o'zgarganda reja tizimni to'xtatmasdan qayta hisoblanadi.
"""
import numpy as np


def joylashtirish_dp(narx, otish_narxi, ruxsat):
    """
    narx:        (N, L) — i-bosqichni l-qatlamda bajarish narxi.
    otish_narxi: (L, L) — ketma-ket bosqichlar orasidagi ma'lumot
                 ko'chirish narxi (qatlamdan qatlamga).
    ruxsat:      (N, L) — huquqiy jihatdan yo'l qo'yiladimi (bool):
                 masalan, LOKAL toifadagi bosqich tashqi bulutda taqiq.
    Qaytaradi: (reja, umumiy_narx). Yechim yo'q bo'lsa (None, inf).
    """
    narx = np.asarray(narx, float)
    otish = np.asarray(otish_narxi, float)
    ruxsat = np.asarray(ruxsat, bool)
    N, L = narx.shape
    INF = np.inf
    dp = np.full((N, L), INF)
    ota = np.full((N, L), -1, dtype=int)

    dp[0] = np.where(ruxsat[0], narx[0], INF)
    for i in range(1, N):
        for l in range(L):                      # joriy qatlam
            if not ruxsat[i, l]:
                continue
            eng, argeng = INF, -1
            for p in range(L):                  # oldingi qatlam
                v = dp[i - 1, p] + otish[p, l]
                if v < eng:
                    eng, argeng = v, p
            if np.isfinite(eng):
                dp[i, l] = eng + narx[i, l]
                ota[i, l] = argeng

    oxirgi = int(np.argmin(dp[-1]))
    if not np.isfinite(dp[-1, oxirgi]):
        return None, float("inf")
    reja = [oxirgi]
    for i in range(N - 1, 0, -1):
        reja.append(int(ota[i, reja[-1]]))
    reja.reverse()
    return reja, float(dp[-1, reja[-1]])
