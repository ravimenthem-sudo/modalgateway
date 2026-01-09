import requests
import json

SUPABASE_URL = "https://ppptzmmecvjuvbulvddh.supabase.co"
SUPABASE_KEY = "eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJpc3MiOiJzdXBhYmFzZSIsInJlZiI6InBwcHR6bW1lY3ZqdXZidWx2ZGRoIiwicm9sZSI6InNlcnZpY2Vfcm9sZSIsImlhdCI6MTc2MzY0NDU5OSwiZXhwIjoyMDc5MjIwNTk5fQ.xR215A_WvpdLKoJQt20FQYaQFSBBchTxh2Mb-fX2-s4"

tables_to_check = ["departments", "teams", "timesheets", "documents", "support_tickets"]
headers = {
    "apikey": SUPABASE_KEY,
    "Authorization": f"Bearer {SUPABASE_KEY}"
}

results = {}
for table in tables_to_check:
    url = f"{SUPABASE_URL}/rest/v1/{table}?limit=1"
    resp = requests.get(url, headers=headers)
    results[table] = "Exists" if resp.status_code == 200 else f"Missing/Error ({resp.status_code})"

print(json.dumps(results, indent=2))
