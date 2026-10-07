# Deploying Vroom on a Vultr VPS

One-time setup on the VPS (Docker + Compose installed, duckdns subdomain
pointing at the VPS IP):

```bash
git clone https://github.com/C-is-for-Cicero/vroom && cd vroom/deploy
cp .env.example .env   # set DOMAIN (and ODDS_API_KEY when you have one)
docker compose up -d --build
```

Caddy terminates HTTPS for `$DOMAIN` automatically. The app serves on
`app:8000` inside the compose network only.

## Scheduled refresh

Model data lives in the `f1-data` volume (`/data` in the app container).
Host cron pulls new session data and will later re-run predictions —
Friday–Sunday during race weekends, hourly:

```cron
0 * * * 5-7 docker compose -f /path/to/vroom/deploy/docker-compose.yml \
  exec -T app python -m f1pred.ingest --season 2026 --force --fastf1
```

(The predict step gets added to this line once build-order steps 2–3 exist.)

## Updating the app

```bash
cd /path/to/vroom && git pull && cd deploy && docker compose up -d --build
```
