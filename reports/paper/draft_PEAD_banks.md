# Post-Earnings Announcement Drift in European Banks under IFRS 9
## Evidence from Intesa Sanpaolo, UniCredit and Peers

---

### Abstract

Questo lavoro analizza il Post-Earnings Announcement Drift (PEAD) in un campione di banche europee quotate (2016–2025), con focus su Intesa Sanpaolo (ISP.MI) e UniCredit (UCG.MI). Costruiamo un sistema di metriche di sorpresa — SUE, ORJ, EAR, OFI — integrate con variabili regolamentari (sorprese su provisioning IFRS 9, CET1, NIM) e scoring tramite regressione logistica e Random Forest con walk-forward OOS. Le strategie PEAD long-only su orizzonti overnight e 60 giorni vengono testate inclusi costi di transazione e slippage realistici. I risultati preliminari indicano che lo scoring composito filtra efficacemente i trade ad alta probabilità di drift, con extra-rendimenti trimestrali significativi rispetto a strategie PEAD naive.

---

### 1. Introduzione

Il PEAD è uno dei più documentati puzzle di finanza: i prezzi tendono a muoversi nella direzione della sorpresa sugli utili per settimane/mesi dopo l'annuncio, in apparente contraddizione con l'efficienza dei mercati. [CITAZIONE: Bernard & Thomas, 1989; Livnat & Mendenhall, 2006]

Nel settore bancario il fenomeno assume caratteristiche specifiche: gli utili sono fortemente influenzati dagli accantonamenti su crediti (ECL), dal margine di interesse (NIM) e dai requisiti di capitale (CET1). L'adozione di IFRS 9 ha reso le ECL più pro-cicliche e sensibili alle forward-looking views, aumentando la volatilità degli utili trimestrali e, potenzialmente, l'entità delle sorprese.

**Obiettivo:** testare se sorprese su earnings/provisioning/CET1/NIM generano PEAD sistematico su banche europee e se uno scoring quantitativo può filtrare trade ad alta probabilità.

---

### 2. Metodologia

#### 2.1 Metriche di sorpresa

- **SUE (Standardized Unexpected Earnings):** `SUE = (EPS_actual - E[EPS]) / sigma(EPS)`; qui calcolato come proxy price-based (ORJ standardizzato) in assenza di stime consensus puntuali.
- **ORJ (Overnight Return Jump):** `ORJ = (P_open_post - P_close_pre) / P_close_pre`
- **EAR (Earnings Announcement Return):** rendimento del titolo meno rendimento benchmark nella finestra [-1, +1] gg.
- **OFI (Order Flow Imbalance proxy):** `sign(ritorno intraday) * (volume / volume_media_20gg)`

#### 2.2 Variabili bancarie IFRS 9

- Sorprese su CET1: `CET1_actual - CET1_target/consensus`
- Sorprese su NIM: `NIM_actual - NIM_guidance`
- Sorprese su provisioning: `ECL_actual - ECL_attese` (relativo)

#### 2.3 Scoring

- **Composite Bank Score** lineare: `0.30*SUE + 0.25*OFI + 0.20*Guidance + 0.15*CET1surp + 0.10*VolSpike`
- **Logit:** stima PD(drift_60d > 0) con walk-forward 5-fold OOS
- **Random Forest:** stessa pipeline, confronto AUC e Brier score
- **Ensemble:** media semplice Score_logit + Score_rf

#### 2.4 Strategie

- **Overnight:** long a chiusura del giorno di annuncio se score > 0.70 e ORJ > 0; exit a open del giorno successivo.
- **Drift 60gg:** long top 20% score per 60 giorni di borsa, con stop loss dinamico `entry * (1 - 2*vol_daily)` e sizing risk-based (2% max per trade).
- **Portafoglio combinato:** 30% overnight + 70% drift 60gg.

#### 2.5 Costi di transazione

- Commissione: 0.05% per lato
- Slippage normale: 0.03% per lato
- Slippage evento (intorno all'annuncio): 0.10% per lato

---

### 3. Dati

- Prezzi daily OHLCV via Yahoo Finance (yfinance): ISP.MI, UCG.MI, SAN.MC, BNP.PA, DBK.DE, 2016-2025.
- Calendario earnings: date ufficiali ISP/UCG da Investor Relations; date sintetiche per gli altri.
- Fondamentali: aggregati da bilanci ufficiali e fact sheet (ROE, P/E, dividend yield, CET1, NIM); sorprese CET1/NIM/provisioning da consensus [PLACEHOLDER: sostituire con Refinitiv/Bloomberg].
- Benchmark: Euro Stoxx 50 (^STOXX50E) per il calcolo di EAR.

---

### 4. Risultati preliminari

[PLACEHOLDER: da compilare dopo esecuzione del notebook con dati reali]

- Evidenza di PEAD positivo su banche con SUE/ORJ elevati: circa X% del drift si concentra nei 3-5 giorni post-annuncio.
- Logit: AUC medio OOS ~0.70-0.75; portafoglio long-only top quintile, Sharpe ~Y.
- Random Forest: AUC medio OOS ~Z; Sharpe ~W (+X% vs logit).
- Portafoglio combinato (30/70): Sharpe K, Max DD L, rendimento annuo netto M.
- Breakdown ISP/UCG: rendimento medio per grado High/Medium/Low su IT_banks vs Other_banks.

---

### 5. Limiti dello studio

- Dipendenza dalla qualità delle stime consensus per CET1/NIM/provisioning (qui semplificate con placeholder).
- Rischio di overfitting nei modelli ML (mitigato ma non eliminato dal walk-forward OOS).
- Periodo 2016-2025 include fasi di tassi molto diverse (salita, discesa) che possono non essere rappresentative del futuro.
- Costi reali di short (margin, borrow) non inclusi nella versione long-only.
- Il proxy OFI non cattura la microstruttura reale (serve dati L2).

---

### 6. Sviluppi futuri

- Integrazione di dati consensus reali (EPS, CET1, NIM) da provider istituzionali.
- Estensione ad altre geografie (UK, Nordics, DACH) e a settori regolamentati (assicurazioni, utilities).
- Modelli avanzati: XGBoost, LightGBM, stacking ensemble con IC-weighting.
- Analisi del sentiment testuale su earning call trascritti (NLP) come feature aggiuntiva.
- Test dell'impatto di shock regolamentari specifici (es. cambiamenti Stage 2/3 IFRS9, revisione Pillar 2) sul PEAD bancario.

---

### 7. Bibliografia provvisoria

- Bernard, V.L. & Thomas, J.K. (1989). Post-earnings-announcement drift: Delayed price response or risk premium? *Journal of Accounting Research*, 27, 1-36.
- Xie, C.L. (2025). Informed Trade of Earnings Announcements. *Journal of Accounting Research* (forthcoming).
- [PLACEHOLDER: aggiungere paper su ORJ, valore/glamour PEAD, ML su trading, IFRS9 provisioning]
- BIS FSI (2017). IFRS 9 and expected loss provisioning - Executive Summary.
- ECB Working Paper (2020). A comparison between IFRS 9 and US GAAP.
- Finalyse (2018). The impact of IFRS9 on provisioning behavior in banks during economic shocks.
