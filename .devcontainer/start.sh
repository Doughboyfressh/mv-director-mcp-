#!/usr/bin/env bash
# Runs every time the Codespace starts: launches the MCP server and prints the connector URL.
set -u
KEY_FILE="$HOME/.mvd_access_key"
if [ ! -s "$KEY_FILE" ]; then
  python3 -c "import secrets; print(secrets.token_urlsafe(24))" > "$KEY_FILE"
  chmod 600 "$KEY_FILE"
fi
export MVD_ACCESS_KEY="$(cat "$KEY_FILE")"
export MVD_WORKDIR="${MVD_WORKDIR:-$HOME/mv-director-workspace}"
DOMAIN="${GITHUB_CODESPACES_PORT_FORWARDING_DOMAIN:-app.github.dev}"
export MVD_PUBLIC_URL="https://${CODESPACE_NAME}-8000.${DOMAIN}"

pkill -f "mv-director --http" 2>/dev/null || true
setsid nohup mv-director --http --port 8000 > "$HOME/mv-director.log" 2>&1 < /dev/null &

# Claude needs to reach the server from the internet: make the port public.
# The secret key in the URL is what keeps it private.
for i in 1 2 3 4 5; do
  sleep 3
  gh codespace ports visibility 8000:public -c "$CODESPACE_NAME" >/dev/null 2>&1 && PUBLIC_OK=1 && break
done

URL="${MVD_PUBLIC_URL}/${MVD_ACCESS_KEY}/mcp"
echo "$URL" > CONNECTOR_URL.txt
cat <<EOF

=====================================================================
 Music Video Director is running.

 Your Claude connector URL (keep it secret - it works like a password):
 $URL

 It is also saved in the file CONNECTOR_URL.txt in this Codespace.
EOF
if [ -z "${PUBLIC_OK:-}" ]; then
  echo " ACTION NEEDED: open the PORTS tab, long-press port 8000 > Port Visibility > Public."
fi
echo "====================================================================="
