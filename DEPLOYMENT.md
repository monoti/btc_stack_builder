# BTC Stack-Builder Bot – Deployment Guide
_“Build it safe, ship it secure, watch it grow.”_

---

## 0.  Prerequisites & Reference

| Item | Minimum | Recommended |
|------|---------|-------------|
| OS   | Ubuntu 22.04 LTS (64-bit) | Same, fully-patched |
| CPU  | 2 vCPU  | 4 vCPU (Intel AES-NI / AMD-SEV preferred) |
| RAM  | 4 GiB   | 8 GiB+ |
| Disk | 40 GB SSD | 80 GB NVMe |
| Public IP | static / elastic | behind Cloud-flare → Proxied |
| Docker | ≥ 24.0 | Latest stable |
| Docker Compose | v2 plugin | Latest stable |
| Domain names | `prometheus.example.com`, `grafana.example.com`, etc. | Same + valid TLS certs (Let’s Encrypt or CA) |

> All steps assume Bash, root/sudo privileges, and a **non-root deployment user** called `btcstack`.

---

## 1.  Phase 1 – Development (Local)

1.  Install Docker Desktop / Colima + Docker Compose.
2.  Clone repository:  
    `git clone https://github.com/<you>/btc_stack_builder && cd btc_stack_builder`
3.  Copy sample config & secrets:  
   ```bash
   cp -r config config.local
   export BTC_STACK_BUILDER_CONFIG_DIR=$(pwd)/config.local
   ```
4.  Edit `config.local/app.yaml`
   * `dry_run: true`
   * `binance.use_testnet: true`
   * `deribit.use_testnet: true`
   * Add **_testnet_** API keys (trade-only, no withdrawal).
5.  Launch stack: `docker compose up -d`
6.  Verify:
   * `http://localhost:8000/metrics`  → Prometheus exposition
   * `http://localhost:3000` Grafana  (admin / changeme)

---

## 2.  Phase 2 – Testnet Incubation (Remote VPS)

### 2.1  Server Hardening

| Task | Command |
|------|---------|
| Create non-root user | `adduser btcstack && usermod -aG docker btcstack` |
| SSH hardening | Disable password login, enable U2F/Ed25519 keys |
| Firewall | `ufw allow 22,80,443,9090,3000,5555/tcp` → `ufw enable` |
| Time sync | `timedatectl set-ntp true` |
| Kernel sysctls | Disable IP source routing, enable TCP syncookies |

### 2.2  Install Runtime

```bash
curl -fsSL https://get.docker.com | sh
sudo apt-get install docker-compose-plugin
```

### 2.3  Prepare Files

```
/opt/btc_stack_builder/
├── docker-compose.yml
├── config/           # decrypted configs
├── secrets/          # Docker secrets files (chmod 600)
└── logs/ data/
```

*Generate secrets with `openssl rand -hex 32 > secrets/<name>.txt`.*

### 2.4  TLS

* Use Caddy or Traefik for automatic Let’s Encrypt.
* Mount certs into `infra/ssl/*` or terminate TLS at reverse proxy.

### 2.5  Run Stack

```bash
docker compose pull
docker compose up -d
```

### 2.6  Validation Checklist

- [ ] `docker ps` all containers healthy  
- [ ] Grafana dashboards populating  
- [ ] Alertmanager test alert delivered (Telegram/Slack)  
- [ ] Margin Guard logs every 5 min  
- [ ] **No** real funds on testnets

Run for ≥ 7 days, perform chaos tests (kill gateway container, reboot host).

---

## 3.  Phase 3 – Mainnet Canary (≤ 5 % Capital)

1. Switch `environment: production`, `dry_run: false`, keys point to _real_ exchange.
2. Whitelist server IP on exchange security panel.
3. **Ensure withdrawal permissions are _disabled_** on all API keys.
4. Load **whitelisted cold-wallet address** into secret `cold_wallet_address.txt`.
5. Replace secrets files with production values (use Ansible/Vault).
6. Bring stack up using **production override**:  
   `docker compose -f docker-compose.yml -f docker-compose.prod.yml up -d --pull always`
7. Confirm:
   * Margin Ratio gauge ≥ 800 % (initial)
   * First Basis/Funding strategy run executes **dry trade** of tiny size
   * Profit withdrawals appear in exchange history with correct destination (test small withdrawal first)
8. Observe for 30 days – if PnL, funding payments, alerts all nominal → go full.

---

## 4.  Phase 4 – Full Production

### 4.1  Scaling Capital

* Deposit remaining BTC spot into exchange **spot wallet**.  
* Use internal transfer to COIN-M wallet **only as needed**; Margin Guard tops up automatically.
* Monitor margin ratio after each deposit.

### 4.2  High Availability

| Component | Strategy |
|-----------|----------|
| DB        | Nightly `pg_dump` to offsite S3; WAL shipping optional |
| Docker    | `--restart=always` + systemd services |
| Updates   | Watchtower (monitored) or manual `docker compose pull && up -d` |
| Metrics   | Prometheus remote-write to Grafana Cloud for redundancy |

### 4.3  Key Rotation

| Item | Interval | Tool |
|------|----------|------|
| Exchange API keys | 90 days | Rotate via exchange UI + update secrets |
| Telegram Bot Token | 6 months | Regenerate and update secret |
| SSL certificates | 90 days | Auto via Let’s Encrypt cron / Caddy |

### 4.4  Backups

```bash
# PostgreSQL
pg_dump -Fc -U btcstack btc_stack_builder | \
  gpg --encrypt --recipient admin@example.com | \
  aws s3 cp - s3://btcstack-backups/$(date +%F)/db.dump.gpg
# Grafana dashboards
grafana-cli admin export-dashboards > dashboards.json
```

---

## 5.  Security Hardening Checklist ✔️

- [ ] API keys: IP-restricted, “read/write”, **no withdraw**  
- [ ] Cold-wallet address whitelisted – multisig or hardware  
- [ ] Docker containers run as unprivileged UID, `no-new-privileges`  
- [ ] `docker network` isolated, only reverse proxy exposed  
- [ ] All secrets mounted via Docker Secrets (tmpfs)  
- [ ] Prometheus & Grafana behind HTTPS + Basic Auth/OAuth  
- [ ] Fail2Ban monitoring SSH & reverse proxy logs  
- [ ] Regular CVE scans (GitHub Actions `security.yml`, Trivy)  

---

## 6.  Monitoring & Alerting

| Metric | Source | Grafana Panel |
|--------|--------|---------------|
| Margin Ratio | Margin Guard → Prometheus | Gauge, thresholds 450 % / 400 % |
| Basis % vs Threshold | Basis Engine | Trend line |
| Funding Rate | Funding Engine | SingleStat + sparkline |
| Total BTC Yield | Portfolio service | Running total |
| Container Health | cAdvisor / Node Exporter | CPU/RAM alerts |
| Withdrawal Success | Bot logs → Alertmanager | Slack/Telegram message |

Alert routing (`infra/alertmanager/*.yml`):

```
receivers:
  - name: critical
    telegram_configs:
      - api_url: https://api.telegram.org
        bot_token: <secret>
        chat_id: <secret>
        parse_mode: HTML
```

---

## 7.  Operational Run-book

| Scenario | Action |
|----------|--------|
| **Margin Ratio < 400 %** | Bot auto-tops-up or unwinds; confirm in Grafana, consider manual de-leverage. |
| Exchange API error loop | Restart gateway container; if persistent → failover to testnet mode. |
| New contract listing (quarterly) | Basis Engine rolls automatically; verify new symbol parsed correctly. |
| Host kernel update | `docker compose stop` → apply patches → reboot → `docker compose up -d`. |
| Disaster recovery | Provision fresh VPS → clone repo → restore `pg_dump` → deploy stack → replay state from exchange. |

---

## 8.  Glossary

* **COIN-M Futures** – BTC-settled quarterly contracts on Binance.
* **Funding Rate** – Payment between long & short every 8 h on perpetuals.
* **Basis** – Annualised premium of futures vs spot BTC.
* **Margin Ratio** – Wallet Balance ÷ Maintenance Margin.

---

## 9.  Support & Contribution

1. **Bug** → GitHub *Issues* using the *Bug Report* template.  
2. **Feature** → GitHub *Issues* → *Feature Request* template.  
3. Security disclosures → security@your-domain (PGP key in repo).

Happy stacking 🚀
