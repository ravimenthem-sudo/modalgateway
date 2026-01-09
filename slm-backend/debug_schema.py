import requests
import json

SUPABASE_URL = "https://ppptzmmecvjuvbulvddh.supabase.co"
SUPABASE_KEY = "eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJpc3MiOiJzdXBhYmFzZSIsInJlZiI6InBwcHR6bW1lY3ZqdXZidWx2ZGRoIiwicm9sZSI6InNlcnZpY2Vfcm9sZSIsImlhdCI6MTc2MzY0NDU5OSwiZXhwIjoyMDc5MjIwNTk5fQ.xR215A_WvpdLKoJQt20FQYaQFSBBchTxh2Mb-fX2-s4"

headers = {
    "apikey": SUPABASE_KEY,
    "Authorization": f"Bearer {SUPABASE_KEY}",
    "Range": "0-0"
}

print("Inspecting 'leaves' table...")
resp = requests.get(f"{SUPABASE_URL}/rest/v1/leaves?select=*", headers=headers)
if resp.status_code == 200:
    data = resp.json()
    if data:
        print("Columns found in 'leaves':", list(data[0].keys()))
        print("Sample data:", data[0])
    else:
        print("Table 'leaves' is empty, cannot inspect columns via select.")
else:
    print(f"Error {resp.status_code}: {resp.text}")

print("\nInspecting 'profiles' table...")
resp = requests.get(f"{SUPABASE_URL}/rest/v1/profiles?select=*", headers=headers)
if resp.status_code == 200:
    data = resp.json()
    if data:
        print("Columns found in 'profiles':", list(data[0].keys()))
    else:
        print("Table 'profiles' is empty.")
else:
    print(f"Error {resp.status_code}: {resp.text}")
