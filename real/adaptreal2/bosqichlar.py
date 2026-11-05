# -*- coding: utf-8 -*-
"""8 bosqichli tavsiya pipeline'ini instrumentlash (2-maqola, 6.1-bo'lim).

Maqolaning hozirgi 1-jadvalidagi bosqich narxlari taxminiy edi. Bu modul
ularni 1-maqolaning REAL komponentlaridan, real oqim ustida o'lchaydi:
har bir bosqich uchun CPU-sekund va bosqichlar orasida oqib o'tadigan
BAYT hajmi.

Bosqichlar (maqoladagi tartib):
  0 ingestion         hodisani qabul qilish va normallashtirish
  1 feature           foydalanuvchi holati va tarix xususiyatlari
  2 retrieval         nomzodlar (to'liq reyting rejimida arzon skorerlar)
  3 pre-ranking       arzon skorerlar (m0, m1)
  4 ranking           asosiy skorerlar (m3, m4) va fuzya
  5 re-ranking        cross-encoder (m5)
  6 logging           qaror yozuvini saqlash
  7 analytics         paketli agregatsiya
"""
import time, json, numpy as np

NOMLAR = ["ingestion", "feature", "retrieval", "pre_ranking",
          "ranking", "re_ranking", "logging", "analytics"]


class Hisob:
    """Bosqich bo'yicha CPU va bayt hisoblagichi."""

    def __init__(self):
        self.cpu = {n: 0.0 for n in NOMLAR}
        self.bayt_ichkari = {n: 0 for n in NOMLAR}
        self.bayt_tashqari = {n: 0 for n in NOMLAR}
        self.chaqiruv = {n: 0 for n in NOMLAR}

    def qadam(self, nom, fn, kirish_bayt=0):
        t0 = time.process_time()
        natija, chiqish_bayt = fn()
        self.cpu[nom] += time.process_time() - t0
        self.bayt_ichkari[nom] += kirish_bayt
        self.bayt_tashqari[nom] += chiqish_bayt
        self.chaqiruv[nom] += 1
        return natija, chiqish_bayt

    def hisobot(self, n_sorov):
        out = {}
        for n in NOMLAR:
            c = max(1, self.chaqiruv[n])
            out[n] = {
                "chaqiruv": self.chaqiruv[n],
                "cpu_ms_sorov": round(self.cpu[n] / c * 1000.0, 5),
                "cpu_soat_1M_sorov": round(self.cpu[n] / c * 1e6 / 3600.0, 5),
                "bayt_kirish_sorov": round(self.bayt_ichkari[n] / c, 1),
                "bayt_chiqish_sorov": round(self.bayt_tashqari[n] / c, 1),
                "gb_chiqish_1M_sorov": round(self.bayt_tashqari[n] / c * 1e6 / 1024**3, 4),
            }
        return out
