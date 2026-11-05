# -*- coding: utf-8 -*-
"""6.1-bo'lim davomi: sindirish (break-even) supurmasi va m5 ning narxga ta'siri.

run_joylashtirish.py dan keyin ishga tushiriladi va natijalar/joylashtirish.json
ga quyidagi kalitlarni qo'shadi: haqiqiy_L2_L1_nisbat, sindirish_sweep,
m5_tasiri, xulosa. Asosiy ssenariy: L0 = UZ kolokatsiya, kanal yuklamasi 30%.
"""
import sys, os, json
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from adaptreal2.narx import joylashtir, uzatish_narxi, L0_VCPU_UZ_KOLO, L1_VCPU, L2_VCPU
from adaptreal2.bosqichlar import NOMLAR
from run_joylashtirish import REJIMLAR, mask_qur, cpu, gb

F = "natijalar/joylashtirish.json"
R = json.load(open(F))
uz = uzatish_narxi(0.30)
ERKIN = mask_qur(REJIMLAR["A. Cheklovsiz"])
QATIY = mask_qur(REJIMLAR["B. 2021 tahriri (keng lokalizatsiya)"])

sweep = []
for r in [1.0, 0.8, 0.6, 0.5, 0.45, 0.4, 0.35, 0.3, 0.25, 0.2, 0.15, 0.1, 0.05]:
    v = {0: L0_VCPU_UZ_KOLO, 1: L1_VCPU, 2: r * L1_VCPU}
    e, _ = joylashtir(cpu, gb, ERKIN, uz, v)
    q, _ = joylashtir(cpu, gb, QATIY, uz, v)
    sweep.append({"L2_L1_nisbat": r, "L2_usd_soat": round(r * L1_VCPU, 6),
                  "erkin_usd_1M": round(e, 4), "qatiy_usd_1M": round(q, 4),
                  "komplayens_foiz": round((q / e - 1) * 100, 2)})

v = {0: L0_VCPU_UZ_KOLO, 1: L1_VCPU, 2: L2_VCPU}
bilan, _ = joylashtir(cpu, gb, ERKIN, uz, v)
cpu_siz = [0.0 if n == "re_ranking" else c for n, c in zip(NOMLAR, cpu)]
siz, _ = joylashtir(cpu_siz, gb, ERKIN, uz, v)

R["haqiqiy_L2_L1_nisbat"] = round(L2_VCPU / L1_VCPU, 3)
R["sindirish_sweep"] = sweep
R["m5_tasiri"] = {"m5_bilan_usd_1M": round(bilan, 4), "m5_siz_usd_1M": round(siz, 4),
                  "cpu_soat_m5_bilan": round(sum(cpu), 2), "cpu_soat_m5_siz": round(sum(cpu_siz), 2),
                  "tejash_foiz": round((1 - siz / bilan) * 100, 1)}
R["xulosa"] = ("Ommaviy narxlarda komplayens narxi NOL: cheklovsiz optimum ham to'liq mamlakat ichida "
               "qoladi, chunki AWS Frankfurt UzCloud dan vCPU-soat bo'yicha 3,84 barobar QIMMAT. "
               "Komplayens narxi faqat tashqi bulut mahalliydan ~60% arzon bo'lgandagina paydo bo'ladi.")
json.dump(R, open(F, "w"), ensure_ascii=False, indent=1)
print("break-even: komplayens > 0 birinchi nisbat =", [s["L2_L1_nisbat"] for s in sweep if s["komplayens_foiz"] > 0][0])
print("m5:", R["m5_tasiri"])
