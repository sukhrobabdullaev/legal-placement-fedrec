# -*- coding: utf-8 -*-
"""Uch huquqiy rejim uchun real USD/1M so'rov (2-maqola, 6.1-bo'lim)."""
import sys, os, json
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from adaptreal2.narx import (joylashtir, uzatish_narxi, VCPU, QATLAM,
                             L0_VCPU_US_KOLO, L0_VCPU_UZ_KOLO, L1_VCPU, L2_VCPU)
from adaptreal2.bosqichlar import NOMLAR

prof = json.load(open("natijalar/bosqich_profili.json"))["bosqichlar"]
cpu = [prof[n]["cpu_soat_1M_sorov"] for n in NOMLAR]
gb = [prof[n]["gb_chiqish_1M_sorov"] for n in NOMLAR]

# ---- huquqiy toifalar --------------------------------------------------
# LOKAL: xom shaxsga doir ma'lumot bilan bevosita ishlaydi
# SHARTLI: hosila (identifikatorlar, ballar), shartli rejimga tushadi
# OCHIQ: agregatsiya qilingan yoki katalog ma'lumoti
TOIFA = {
    "ingestion": "LOKAL", "feature": "LOKAL", "retrieval": "SHARTLI",
    "pre_ranking": "SHARTLI", "ranking": "SHARTLI", "re_ranking": "SHARTLI",
    "logging": "LOKAL", "analytics": "OCHIQ",
}

def mask_qur(ruxsat):
    """ruxsat: toifa -> ruxsat etilgan qatlamlar to'plami."""
    return [[l in ruxsat[TOIFA[n]] for l in (0, 1, 2)] for n in NOMLAR]

REJIMLAR = {
    "A. Cheklovsiz": {"LOKAL": {0,1,2}, "SHARTLI": {0,1,2}, "OCHIQ": {0,1,2}},
    "B. 2021 tahriri (keng lokalizatsiya)":
        {"LOKAL": {0,1}, "SHARTLI": {0,1}, "OCHIQ": {0,1,2}},
    "C. 2026 tahriri, shlyuz YO'Q":
        {"LOKAL": {0,1}, "SHARTLI": {0,1}, "OCHIQ": {0,1,2}},
    "C+. 2026 tahriri, shlyuz BOR":
        {"LOKAL": {0,1}, "SHARTLI": {0,1,2}, "OCHIQ": {0,1,2}},
}

def hisobla(vcpu, yuklama):
    uz = uzatish_narxi(yuklama)
    out = {}
    for nom, ruxsat in REJIMLAR.items():
        narx, reja = joylashtir(cpu, gb, mask_qur(ruxsat), uz, vcpu)
        out[nom] = {"narx_usd_1M": round(narx, 2),
                    "reja": [QATLAM[l] for l in reja] if reja else None,
                    "reja_id": reja}
    return out

senaryolar = {}
for L0v, nomL0 in [(L0_VCPU_US_KOLO, "L0=AQSh kolokatsiya"),
                   (L0_VCPU_UZ_KOLO, "L0=UZ kolokatsiya")]:
    for yuk in (0.20, 0.30, 1.00):
        v = {0: L0v, 1: L1_VCPU, 2: L2_VCPU}
        senaryolar["%s, kanal %d%%" % (nomL0, int(yuk*100))] = hisobla(v, yuk)

json.dump({"bosqichlar": NOMLAR, "toifa": TOIFA,
           "cpu_soat_1M": dict(zip(NOMLAR, cpu)),
           "gb_chiqish_1M": dict(zip(NOMLAR, gb)),
           "vcpu_narx": {"L0_AQSh": L0_VCPU_US_KOLO, "L0_UZ": L0_VCPU_UZ_KOLO,
                         "L1": L1_VCPU, "L2": L2_VCPU},
           "senaryolar": senaryolar},
          open("natijalar/joylashtirish.json", "w"), ensure_ascii=False, indent=1)

asosiy = senaryolar["L0=UZ kolokatsiya, kanal 30%"]
baza = asosiy["A. Cheklovsiz"]["narx_usd_1M"]
print("ASOSIY SSENARIY (L0 = UZ kolokatsiya, kanal yuklamasi 30%)\n")
print("%-38s %-14s %s"%("rejim","USD/1M so'rov","cheklovsizga nisbatan"))
print("-"*78)
for nom, r in asosiy.items():
    d = (r["narx_usd_1M"]/baza - 1) * 100
    print("%-38s %-14.2f %+.0f%%"%(nom, r["narx_usd_1M"], d))
print("\nOptimal rejalar:")
for nom, r in asosiy.items():
    print("  %s"%nom)
    print("    " + " -> ".join("%s:%s"%(n[:4], (r["reja"][k] or "")[:12]) for k,n in enumerate(NOMLAR)))
