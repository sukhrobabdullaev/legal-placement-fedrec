# 2-maqola kodlari (real ma'lumot bilan)

**Maqola:** "Legality-Aware Placement of Recommender Pipelines and the
Privacy-Compression-Utility Trade-off in Federated Update Streams under
Data-Localization Law"

Paket ikki qismdan iborat: `experiments/` sintetik rang qonuni tajribasini,
`real/` esa o'lchangan resurs profillari, ochiq bulut narxlari va real
federativ yangilanish spektri ustidagi tajribalarni qayta hosil qiladi.

## real/ tuzilishi

```
adaptreal2/
  bosqichlar.py     — sakkiz bosqichli resurs hisobi (Hisob)
  narx.py           — real narx modeli, joylashtir() dinamik dasturlash
run_bosqich.py      — bosqich profillari (CPU-soat va GB, 1M so'rovga)
run_joylashtirish.py— rejimlar bo'yicha optimal reja, sindirish supurmasi
run_zeta.py         — downstream NDCG mezoni (shovqinli, maqolada faqat
                      cheklov sifatida keltirilgan)
run_zeta2.py        — rekonstruksiya mezoni bo'yicha r*(eps) va spektr
run_spektr.py       — to'liq singulyar spektrni saqlash
make_fig1.py        — Figure 1 (sindirish egri chizigi + DP amallari)
make_fig3.py        — Figure 3 (o'lchangan spektr + rang qonuni)
natijalar/*.json
figures/a2_fig1.png, a2_fig3.png
```

## Ma'lumot va narxlar

- Bosqich talablari: `Maqola1_kodlari/real/` dagi ishlaydigan pipeline
  instrumentatsiyasidan: 1M so'rovga **34.30 CPU-soat**, shundan
  **91%** faqat re-ranking bosqichida.
- Qatlam narxlari 2026-08-23 dagi ochiq preyskurantlardan:
  UzCloud, AWS Frankfurt, CBU rasmiy kursi.
- Federativ yangilanish matritsasi: Amazon Reviews 2023 oqimida
  (857 505 hodisa) o'qitilgan BPR-MF, 2000 silo, 2048 eng faol item.

## Maqoladagi asosiy real raqamlar

- To'rt huquqiy rejimning **hammasida** optimal reja bir xil va to'liq
  mamlakat ichida: **0.264 USD / 1M so'rov**, komplayens narxi **0%**.
  Sabab: AWS Frankfurt UzCloud dan vCPU-soat bo'yicha **3.84 barobar qimmat**.
- Sindirish nuqtasi: komplayens narxi faqat tashqi/mahalliy narx nisbati
  **~0.47 dan past** bo'lganda paydo bo'ladi (0.50 da 0%, 0.45 da +9.1%,
  0.40 da +22.7%, 0.10 da +390.8%).
- m5 (cross-encoder) ni olib tashlash infratuzilma narxini
  0.264 dan 0.025 USD/1M ga, ya'ni **91%** ga tushiradi.
- O'lchangan spektr: zeta = **0.037**, power-law R2 = **0.904**
  (eksponensial R2 = 0.635).
- Rang qonuni: nazariya 2/(zeta+1) = **1.928**, o'lchangan qiyalik
  **2.017** (R2 = 0.937, 8 ta ichki nuqta), farq **4.6%** dan kam.
- Aniq qayta rejalashtirish: **72 amal** (N = 8, |L| = 3).

Sintetik rang qonuni raqamlari o'zgarmagan (zeta = 0.5 / 1 / 2 uchun
qiyalik 1.40 / 1.02 / 0.68, nazariya 1.33 / 1.00 / 0.67, R2 > 0.997).

## Ishga tushirish

```bash
pip install numpy matplotlib
python run_bosqich.py
python run_joylashtirish.py
python run_spektr.py
python run_zeta2.py
python make_fig1.py && python make_fig3.py
```

Barcha urug'lar kodda qayd etilgan (rng seed = 7).

## Rasmlar qayerdan chiqadi

| Maqoladagi rasm | Skript |
|---|---|
| Figure 1 (real narxlarda sindirish egri chizigi) | `real/make_fig1.py` |
| Figure 2 (sintetik rang qonuni) | `experiments/rank_law.py` |
| Figure 3 (o'lchangan spektr va rang qonuni) | `real/make_fig3.py` |

`experiments/placement_scenarios.py` endi `a2_fig1_sintetik.png` yozadi,
chunki maqoladagi Figure 1 real narxlardan olingan.
