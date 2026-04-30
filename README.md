# PEAD su Banche Europee (Intesa, UniCredit)

Progetto quantitativo per analizzare il Post-Earnings Announcement Drift (PEAD) su un paniere di banche europee, con focus su **Intesa Sanpaolo (ISP.MI)** e **UniCredit (UCG.MI)**, nel contesto contabile IFRS 9 e dei requisiti di capitale CET1.

## Obiettivo

- Costruire metriche di sorpresa (SUE, ORJ, EAR, OFI) su annunci di earnings bancari.
- Integrare variabili regolamentari: sorprese su provisioning IFRS9, CET1 e NIM.
- Scoring tramite regressione logistica + Random Forest (walk-forward OOS).
- Backtesting di strategie long-only: overnight (0-3 gg) e drift 60 giorni.
- Metriche di performance: Sharpe, Sortino, Max Drawdown, breakdown per banca.

## Struttura

```
pead-banks-europe/
├─ README.md
├─ requirements.txt
├─ .gitignore
├─ config/
│  └─ params.yaml
├─ notebooks/
│  └─ PEAD_banks_end_to_end.ipynb
├─ src/
│  ├─ __init__.py
│  ├─ data_loading.py
│  ├─ features_pead.py
│  ├─ models_scoring.py
│  ├─ backtest_strategies.py
│  └─ reporting.py
├─ data/
│  ├─ raw/
│  │  ├─ earnings_calendar_raw.csv   # sostituire con dati reali
│  │  └─ fundamentals_raw.csv        # sostituire con dati reali
│  └─ processed/                     # generato da src/data_loading.py
└─ reports/
   ├─ figures/
   └─ paper/
      └─ draft_PEAD_banks.md
```

## Installazione

```bash
git clone https://github.com/TheGenesisAIStory/pead-banks-europe.git
cd pead-banks-europe
python -m venv .venv
source .venv/bin/activate  # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

## Esecuzione

### Notebook (percorso consigliato)

```bash
jupyter lab
# Apri: notebooks/PEAD_banks_end_to_end.ipynb
```

### Pipeline modulare

```bash
python -m src.data_loading      # Scarica e salva prezzi + fondamentali
python -m src.features_pead     # Calcola SUE, ORJ, EAR, OFI, score
python -m src.models_scoring    # Addestra logit + RF, salva modelli
python -m src.backtest_strategies  # Esegue backtest e salva risultati
python -m src.reporting         # Genera grafici e tabelle in reports/
```

## Configurazione

Tutti i parametri (tickers, date, soglie, pesi) sono in `config/params.yaml`. Modifica quel file per personalizzare l'analisi senza toccare il codice.

## Dati

I dati grezzi in `data/raw/` **non sono tracciati in Git** (vedi `.gitignore`). Sono presenti due file di esempio già compilati con placeholder realistici per ISP/UCG (2016-2025). Sostituirli con dati reali prima dell'esecuzione in produzione.

## Disclaimer

Il progetto ha finalità esclusivamente didattiche e di ricerca quantitativa. Non costituisce raccomandazione di investimento o consulenza finanziaria.

## Licenza

MIT
