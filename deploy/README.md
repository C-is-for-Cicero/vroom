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

## Accounts (invite-only)

The whole site requires login. Create the first (admin) user and mint
invite codes for friends from inside the app container:

```bash
docker compose exec app python -m vroom.manage create-admin <you>
docker compose exec app python -m vroom.manage invite -n 5
```

Hand each friend one code; they register at `https://$DOMAIN/register`.
Each code works once. The user database lives in the `f1-data` volume
(`/data/vroom.sqlite`) — back it up by copying the file.

## Scheduled refresh

Model data lives in the `f1-data` volume (`/data` in the app container).
Host cron pulls new session data and will later re-run predictions —
Friday–Sunday during race weekends, hourly:

```cron
0 * * * 5-7 docker compose -f /path/to/vroom/deploy/docker-compose.yml \
  exec -T app python -m f1pred.ingest --season 2026 --force --fastf1
# bookmaker odds snapshot - DORMANT: The Odds API currently has no F1
# coverage (verified 2026-10-07). Re-enable when a provider with F1 exists.
# 0 12 * * 4-6 docker compose -f /path/to/vroom/deploy/docker-compose.yml \
#   exec -T app python -m f1pred.ingest.odds --snapshot --build
```

(The predict step gets added to this line once build-order steps 2–3 exist.)

## Updating the app

```bash
cd /path/to/vroom && git pull && cd deploy && docker compose up -d --build
```
