# -*- coding: utf-8 -*-
"""Real narx modeli va qonuniyatga sezgir joylashtirish (2-maqola, 4 va 6.1).

Barcha narxlar ommaviy manbalardan, 2026-08-23 holatiga.
Valyuta kursi: 1 USD = 11 847,16 UZS (MB rasmiy kursi, 21.08.2026).

Qatlamlar:
  L0  o'z serveri (O'zbekistonda, operator nazoratida)
  L1  mahalliy bulut (UzCloud)
  L2  tashqi bulut (AWS eu-central-1)
"""
import json, numpy as np

USD_UZS = 11847.16

# --- hisoblash narxi, USD / vCPU-soat (RAM ichida) ---------------------
# L1: UzCloud rasmiy narx sahifasi, 118 UZS/vCPU-soat + 17 UZS/GB-soat,
#     4 GB/vCPU nisbatida
L1_VCPU = (118 + 4 * 17) / USD_UZS
# L2: AWS m7i.large eu-central-1 = 2 vCPU + 8 GiB, 0,1207 USD/soat
L2_VCPU = 0.1207 / 2
# L0: 37signals bulutdan chiqishi, 840 000 USD/yil / 4 000 vCPU / 8 760 soat
L0_VCPU_US_KOLO = 840000.0 / (4000 * 8760)
# L0 sezgirlik: mahalliy kolokatsiya narxi bilan (UzCloud 2 532-3 798 USD/rack/oy,
# AQSh ~15 000 USD/rack/oy) -> kolokatsiya ulushi (86%) shu nisbatda kamayadi
L0_VCPU_UZ_KOLO = L0_VCPU_US_KOLO * (0.14 + 0.86 * (3165.0 / 15000.0))

# --- uzatish narxi, USD / GB -------------------------------------------
# UzCloud: ichki va Tas-IX/UZ-IX trafigi bepul
# L1 -> xalqaro: aHOST majburiyatli Mbps zinasidan, kanal yuklamasiga
#   parametrlangan (100% da 0,0104, 30% da 0,035, 20% da 0,052)
# L2 -> tashqariga: AWS va GCP o'rta hajmda 0,085-0,09 da birlashadi
def uzatish_narxi(kanal_yuklamasi=0.30):
    L1_XALQARO = 0.0104 / kanal_yuklamasi
    return {
        (0, 0): 0.0, (1, 1): 0.0, (2, 2): 0.0,
        (0, 1): 0.0, (1, 0): 0.0,                 # Tas-IX, bepul
        (0, 2): L1_XALQARO, (1, 2): L1_XALQARO,   # mamlakatdan chiqish
        (2, 0): 0.085, (2, 1): 0.085,             # tashqi bulutdan qaytish
    }

VCPU = {0: L0_VCPU_US_KOLO, 1: L1_VCPU, 2: L2_VCPU}
QATLAM = {0: "o'z serveri (UZ)", 1: "mahalliy bulut (UZ)", 2: "tashqi bulut (EU)"}


def joylashtir(cpu_soat, gb_chiqish, mask, uzatish, vcpu=None):
    """Aniq joylashtirish, O(N|L|^2) dinamik dasturlash (Teorema 1).

    cpu_soat[i]    : i-bosqichning 1M so'rovga CPU-soati
    gb_chiqish[i]  : i-bosqichdan keyingisiga o'tadigan GB (1M so'rovga)
    mask[i][l]     : i-bosqich l qatlamda qonuniymi
    Qaytaradi: (narx_USD, reja)
    """
    vcpu = vcpu or VCPU
    N = len(cpu_soat); L = len(vcpu)
    INF = float("inf")
    dp = [[INF] * L for _ in range(N)]
    ota = [[-1] * L for _ in range(N)]
    for l in range(L):
        if mask[0][l]: dp[0][l] = cpu_soat[0] * vcpu[l]
    for i in range(1, N):
        for l in range(L):
            if not mask[i][l]: continue
            ijro = cpu_soat[i] * vcpu[l]
            eng, eng_l = INF, -1
            for lp in range(L):
                if dp[i-1][lp] == INF: continue
                c = dp[i-1][lp] + gb_chiqish[i-1] * uzatish[(lp, l)]
                if c < eng: eng, eng_l = c, lp
            if eng < INF:
                dp[i][l] = eng + ijro; ota[i][l] = eng_l
    son = min(range(L), key=lambda l: dp[N-1][l])
    if dp[N-1][son] == INF: return INF, None
    reja = [son]
    for i in range(N-1, 0, -1):
        son = ota[i][son]; reja.append(son)
    return dp[N-1][min(range(L), key=lambda l: dp[N-1][l])], list(reversed(reja))
