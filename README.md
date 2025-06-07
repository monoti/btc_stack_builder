# BTC Stack-Builder Bot

Autonomous, micro-services Bitcoin accumulation engine targeting a **6 – 10 % annual yield in satoshis** while maintaining strict risk controls.  
Designed for long-term stack growth and built from the ground up for reliability, security and observability.

---

## ✨ Key Features
* **Prime Directive:** maximise BTC quantity vs simple HODL.
* **4-way Portfolio Split**  
  * 60 % Core HODL (cold-wallet, never traded)  
  * 25 % Basis Harvest (quarterly futures contango)  
  * 10 % Funding Capture (negative funding opportunism)  
  * 5 %  Option Premium (cash-secured put wheel)
* **Micro-services Architecture** (Python 3.13, PostgreSQL, Redis, Celery, Prometheus, Grafana).
* **Exchange Gateways** for Binance (spot, COIN-M & PERP) and Deribit (options) via `ccxt-pro`.
* **Risk & Margin Guard** checks margin every 5 min, auto-tops-up or safely unwinds.
* **State-Light Design** – full recovery from DB and exchange APIs after restart.
* **Observability-First** – Prometheus metrics, Grafana dashboards, Alertmanager → Telegram/Slack.
* **Infrastructure-as-Code** – Docker & Compose, CI/CD with GitHub Actions, automated Watchtower rollouts.
* **Comprehensive Test-suite** – unit, integration (testnets), chaos tests.

---

## 🏗️ Architecture Overview

```
Strategy Layer
 ├─ BasisHarvestEngine
 ├─ FundingCaptureEngine
 └─ OptionsWheelEngine
        │
 Scheduler/Task-Queue (APScheduler + Celery)
        │
Binance GW ── Deribit GW
        │        │
 PostgreSQL (trade & state)
        │
Risk Manager (Margin Guard)
        │
Prometheus ➜ Grafana ➜ Alertmanager
```

Containers are orchestrated with **docker-compose**; each service exports Prometheus metrics and restarts automatically.

---

## 🚀 Quick Start

### 1. Prerequisites
* Docker ≥ 24 & Docker Compose v2
* (Optional local run) Python 3.13+, `pipx` or `poetry`

### 2. Clone & build

```bash
git clone https://github.com/<your-org>/btc_stack_builder.git
cd btc_stack_builder
docker compose build
```

### 3. Configure

Copy sample config and fill secrets (never commit real keys):

```bash
cp -r config examples/my-config
export BTC_STACK_BUILDER_CONFIG_DIR=$(pwd)/my-config
# edit my-config/app.yaml / app.production.yaml
```

Secrets can also be provided by:
* Docker Secrets (`docker secret create …`)
* Hashicorp Vault (token exported as env var)

### 4. Launch stack (development / testnet)

```bash
docker compose up -d
```

Visit:

* Grafana: http://localhost:3000  (admin / changeme)
* Prometheus: http://localhost:9090
* Metrics endpoint: http://localhost:8000/metrics
* Flower (Celery): http://localhost:5555

---

## ⚙️ Configuration Guide

| File | Purpose |
|------|---------|
| `config/app.yaml` | base defaults (development / dry-run) |
| `config/app.<environment>.yaml` | overrides for `development`, `testnet`, `production` |
| `.env` | optional env overrides (prefixed `BTC_STACK_BUILDER_`) |

Important fields:

* `portfolio.cold_wallet_address` – **whitelisted hardware or multisig address**.  
* `binance.credentials / deribit.credentials` – **trade-only API keys**, IP-restricted, **no withdrawal permissions**.
* `risk.*` – adjust stop-loss, margin thresholds, max leverage.
* `dry_run` – set `true` for simulation against testnets.

---

## 🛠️ Local Usage Examples

Dry-run with verbose logging:

```bash
python -m btc_stack_builder.main \
  --environment development \
  --log-level DEBUG \
  --dry-run
```

Run only Funding Capture strategy every 10 min:

```bash
BTC_STACK_BUILDER_BASIS_HARVEST__ENABLED=false \
BTC_STACK_BUILDER_OPTION_PREMIUM__ENABLED=false \
docker compose up -d btc-stack-builder
```

---

## ☸️ Deployment Workflow

1. **Development** – run stack locally, pass all tests `pytest`.
2. **Testnet Incubation** – set `use_testnet: true` for exchanges, deploy to VPS with small fake capital.
3. **Mainnet Canary** – `dry_run: false`, real exchange keys, ≤ 5 % allocation. Observe for ≥ 30 days.
4. **Full Production** – scale capital, enable Watchtower for zero-downtime image upgrades.

CI pipeline (GitHub Actions):

* `push` → lint (`ruff`) + unit tests  
* `main` merge → build & push Docker image → tag `latest`  
* Production host pulls via Watchtower, runs DB migrations (`alembic upgrade head`).

---

## 🛡️ Security Checklist

* API keys restricted by IP & no-withdrawal.
* All outbound withdrawals go to single **whitelisted cold wallet**.
* Secrets loaded via environment or mounted secrets – **never hard-code**.
* Containers run as non-root user, minimal base images.
* TLS/SSL enforced for DB & Redis in production.

---

## 📈 Monitoring & Alerting

Grafana dashboards include:

* Margin Ratio Gauge (warnings < 450 %, critical < 400 %)
* Basis %, Funding Rate, Option Delta vs thresholds
* Open PnL, Cum. BTC Yield
* Container CPU/RAM

Alertmanager notifies on:

* Margin warnings/criticals
* Failed orders / API errors
* Successful profit withdrawals
* Container health failures

---

## 📜 Disclaimer

**This project executes real leveraged cryptocurrency trades.**  
Operating the bot exposes you to market risk, exchange risk and potential software defects.

* The authors provide **NO WARRANTY** – use at your own risk.
* Past performance ≠ future returns. The 6 – 10 % target yield is an aspirational goal, not a guarantee.
* Always review code, test extensively on testnet, and start with small capital.
* Ensure compliance with all applicable laws/regulations in your jurisdiction.

Happy stacking 🚀
