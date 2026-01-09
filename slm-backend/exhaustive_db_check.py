import requests
import re
import json

# Extract URL and Key from server.py
with open('server.py', 'r', encoding='utf-8') as f:
    content = f.read()
    url = re.search(r'SUPABASE_URL\s*=\s*[\"\'](.*?)[\"\']', content).group(1)
    key = re.search(r'SUPABASE_KEY\s*=\s*[\"\'](.*?)[\"\']', content).group(1)

headers = {
    "apikey": key,
    "Authorization": f"Bearer {key}"
}

tables = ["profiles", "tasks", "leaves", "attendance", "departments", "teams", "timesheets", "payroll"]
results = {}

for table in tables:
    res = requests.get(f"{url}/rest/v1/{table}?select=*&limit=1", headers=headers)
    if res.status_code == 200 and res.json():
        results[table] = {
            "columns": list(res.json()[0].keys()),
            "sample": res.json()[0]
        }
    else:
        results[table] = "No Data or Error"

# Also get representative IDs for roles
roles = ["executive", "manager", "teamlead", "employee"]
role_ids = {}
for role in roles:
    res = requests.get(f"{url}/rest/v1/profiles?role=eq.{role}&select=id,full_name&limit=1", headers=headers)
    if res.status_code == 200 and res.json():
        role_ids[role] = res.json()[0]
    else:
        role_ids[role] = "None Found"

print(json.dumps({"schemas": results, "role_ids": role_ids}, indent=2))
