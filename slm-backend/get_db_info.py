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

print(f"URL: {url}")

# 1. Get a valid Profile ID
p_res = requests.get(f"{url}/rest/v1/profiles?select=id&limit=1", headers=headers)
if p_res.status_code == 200 and p_res.json():
    print(f"VALID_ID: {p_res.json()[0]['id']}")
else:
    print("VALID_ID: None")

# 2. Get Attendance columns
a_res = requests.get(f"{url}/rest/v1/attendance?select=*&limit=1", headers=headers)
if a_res.status_code == 200 and a_res.json():
    print(f"ATTENDANCE_COLS: {list(a_res.json()[0].keys())}")
else:
    # Try to get from schema definition if no data
    print("ATTENDANCE_COLS: No Data")
