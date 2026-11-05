# 2-maqolaning real narx qatlami

Maqolaning 6.1-bo'limidagi 9,0 / 17,3 / 13,4 shartli birliklari o'rniga
real o'lchangan resurs profili va ommaviy bulut narxlari.

## Ma'lumot manbalari

Barchasi ommaviy, 2026-08-23 holatiga. Kurs: 1 USD = 11 847,16 UZS
(MB rasmiy kursi, 21.08.2026).

| Qatlam | Asos | USD/vCPU-soat |
|---|---|---|
| L0 o'z serveri (AQSh kolokatsiya narxida) | 37signals, 840k USD/yil / 4000 vCPU | 0,02397 |
| L0 o'z serveri (UZ kolokatsiya narxida) | yuqoridagi, UzCloud rack narxi bilan | 0,00771 |
| L1 mahalliy bulut | UzCloud rasmiy: 118 UZS/vCPU-soat + 17 UZS/GB-soat | 0,01570 |
| L2 tashqi bulut | AWS m7i.large eu-central-1, 0,1207 USD/soat | 0,06035 |

Uzatish: UzCloud ichki va Tas-IX trafigi bepul; L1 dan xalqaro yo'nalish
aHOST ning majburiyatli Mbps zinasidan kanal yuklamasiga parametrlangan;
L2 dan qaytish 0,085 USD/GB (AWS va GCP o'rta hajmda shunga birlashadi).

## O'lchangan bosqich profili

1-maqolaning real komponentlaridan, real oqim ustida (400 so'rov,
100 000 hodisadan keyin):

| Bosqich | Toifa | CPU-soat/1M so'rov | GB chiqish/1M |
|---|---|---|---|
| ingestion | LOKAL | 0,0412 | 0,045 |
| feature | LOKAL | 0,0091 | 0,049 |
| retrieval | SHARTLI | 0,0004 | 98,18 |
| pre_ranking | SHARTLI | 2,5465 | 196,35 |
| ranking | SHARTLI | 0,5831 | 0,149 |
| re_ranking | SHARTLI | **31,0983** | 0,724 |
| logging | LOKAL | 0,0188 | 0,093 |
| analytics | OCHIQ | 0,0024 | 0,034 |

Jami 34,30 CPU-soat/1M so'rov, shundan **91% i re-ranking bosqichida**.

## Asosiy natija: komplayens narxi NOL

| Rejim | USD/1M so'rov | Cheklovsizga nisbatan |
|---|---|---|
| A. Cheklovsiz | 0,264 | +0% |
| B. 2021 tahriri (keng lokalizatsiya) | 0,264 | +0% |
| C. 2026 tahriri, shlyuzsiz | 0,264 | +0% |
| C+. 2026 tahriri, shlyuz bilan | 0,264 | +0% |

Sabab: cheklovsiz optimum ham **to'liq mamlakat ichida** qoladi, chunki
AWS Frankfurt UzCloud dan vCPU-soat bo'yicha **3,84 barobar qimmat**.
Ya'ni bu bozorda lokalizatsiya qonuni arxitektura tanlovini umuman
o'zgartirmaydi.

Bu natija oltita ssenariyda (L0 ning ikki narxi x kanal yuklamasining
uch qiymati) bir xil chiqadi.

## Sindirish nuqtasi

Komplayens narxi qachon paydo bo'ladi:

| L2/L1 nisbat | Erkin USD/1M | Qat'iy USD/1M | Komplayens |
|---|---|---|---|
| 0,50 | 0,264 | 0,264 | +0% |
| 0,40 | 0,215 | 0,264 | +22,7% |
| 0,30 | 0,162 | 0,264 | +63,6% |
| 0,20 | 0,108 | 0,264 | +145,4% |
| 0,10 | 0,054 | 0,264 | +390,8% |

Komplayens narxi faqat tashqi bulut mahalliydan **~60% arzon**
bo'lgandagina paydo bo'ladi. Haqiqiy nisbat esa 3,84, ya'ni teskari
tomonda.

## Ikki maqola orasidagi bog'lanish

1-maqolaning topilmasi (`m5` sifatni buzadi, uni chaqirmaslik kerak)
2-maqolaning narxiga bevosita ta'sir qiladi:

| | USD/1M so'rov | Jami CPU-soat |
|---|---|---|
| m5 bilan | 0,264 | 34,3 |
| m5 siz | 0,025 | 3,2 |
| **tejash** | **91%** | |

## Ishga tushirish

```bash
python3 run_bosqich.py          # bosqich profilini o'lchash
python3 run_joylashtirish.py    # rejimlar bo'yicha joylashtirish
```

`run_bosqich.py` 1-maqolaning `real/` paketiga bog'liq (`sys.path`).

---

## Rang qonuni REAL update matritsalarida

`run_zeta.py` (downstream NDCG varianti) va `run_zeta2.py`
(rekonstruksiya mezoni, sintetik tajriba bilan bir xil mezon).

Amazon oqimida BPR matritsa faktorizatsiyasi o'qitilib (d = 64,
40 epoch), foydalanuvchilar 2 000 siloga bo'linadi. Har bir raundda
eng faol 2 048 item uchun agregatsiyalangan gradient matritsasi
yig'iladi. Kesish silo darajasida (DP sezuvchanligi = bitta siloning
hissasi).

### O'lchangan spektr

| | |
|---|---|
| **O'lchangan zeta** | **0,037** |
| Power-law fit R^2 | 0,904 |
| Eksponensial fit R^2 | 0,635 |
| s1 / s64 nisbati | 23,3 |

Ikki xulosa. Birinchi: power-law model eksponensialdan **ancha
yaxshi** mos keladi (0,904 va 0,635), ya'ni Teorema 2 ning shakl
farazi o'rinli. Ikkinchi: real spektr maqolada faraz qilingan
`zeta` qiymatlaridan (0,5 / 1 / 2) **ancha tekis**.

### Qonun tekshiruvi

eps 4 dan 96 gacha 14 nuqtali logarifmik to'rda, 60 takror:

| eps | r* | eps | r* |
|---|---|---|---|
| 4,00 | 1 | 22,14 | 7 |
| 5,11 | 1 | 28,28 | 18 |
| 6,52 | 1 | 36,11 | 35 |
| 8,33 | 2 | 46,11 | 55 |
| 10,64 | 3 | 58,88 | 64 |
| 13,58 | 3 | 75,18 | 64 |
| 17,34 | 4 | 96,00 | 64 |

| | |
|---|---|
| Nazariy ko'rsatkich `2/(zeta+1)` | **1,928** |
| O'lchangan qiyalik | **2,017** |
| Fit R^2 | 0,937 |
| Ichki (chegaraviy bo'lmagan) nuqta | 8 |

**4,6% ichida moslik.** Rang qonuni endi sintetik matritsalarda emas,
zeta si o'lchangan real federated update matritsasida tasdiqlangan.

### Amaliy xulosa

`zeta ≈ 0,04` deyarli tekis spektrni anglatadi. Bu ikki narsani
bildiradi:

1. Past rangli siqish maqolaning `zeta = 1` misolida ko'rsatilganidan
   **ancha kam foyda** beradi, chunki kesish hadi sekin kamayadi
2. Optimal rang eps bo'yicha deyarli **kvadratik** o'sadi
   (ko'rsatkich 1,93), ya'ni maxfiylik byudjeti biroz bo'shashsa ham
   uzatiladigan rang tez ortadi

### Nima ishlamadi

Downstream NDCG varianti (`run_zeta.py`) shovqin bilan to'ldi:
BPR-MF asosi bu datasetda juda zaif (NDCG@10 = 0,0034, taqqoslash
uchun `m4` Markov modeli 0,133 beradi). Signal shovqin poloniga
juda yaqin bo'lgani uchun `r*` argmax'i beqaror chiqdi (2, 1, 8, 45,
31, 51, 9). Shuning uchun asosiy natija rekonstruksiya mezoni
bo'yicha berilgan, downstream varianti esa cheklov sifatida
hisobot qilinadi.
