"""Fetch FG2609 East Money daily position data sample."""
from curl_cffi import requests
import pandas as pd

headers = {"Referer": "https://data.eastmoney.com/futures/sh/data.html"}
base = "https://datacenter-web.eastmoney.com/api/data/v1/get"
params = {
    "reportName": "RPT_FUTU_DAILYPOSITION",
    "columns": "ALL",
    "pageNumber": "1",
    "pageSize": "500",
    "source": "WEB",
    "client": "WEB",
    "sortColumns": "TRADE_DATE",
    "sortTypes": "-1",
    "filter": '(SECURITY_CODE="FG2609")(TRADE_DATE>="2026-06-01")',
}
r = requests.get(base, params=params, headers=headers, impersonate="chrome", timeout=20)
j = r.json()
print("api message:", j.get("message"), "success:", j.get("success"))
result = j.get("result") or {}
data = result.get("data") or []
if not data:
    # try without date filter
    params["filter"] = '(SECURITY_CODE="FG2609")'
    r = requests.get(base, params=params, headers=headers, impersonate="chrome", timeout=20)
    j = r.json()
    print("retry message:", j.get("message"))
    data = (j.get("result") or {}).get("data") or []
df = pd.DataFrame(data)
print("rows", len(df), "cols", len(df.columns))
print("columns:", df.columns.tolist())
latest = df["TRADE_DATE"].max()
d = df[df["TRADE_DATE"] == latest].copy()
print("latest", latest, "members", len(d))
cols = [
    "MEMBER_NAME_ABBR", "VOLUME", "VOLUME_CHANGE",
    "LONG_POSITION", "LP_CHANGE",
    "SHORT_POSITION", "SP_CHANGE",
    "NET_LONG_POSITION", "NLP_CHANGE",
    "SETTLE_PRICE", "LP_AVERAGE_PRICE", "SP_AVERAGE_PRICE",
]
print(d[cols].head(15).to_string())

# daily aggregates
g = df.groupby("TRADE_DATE").agg(
    vol=("VOLUME", "sum"),
    long=("LONG_POSITION", "sum"),
    short=("SHORT_POSITION", "sum"),
    lp_chg=("LP_CHANGE", "sum"),
    sp_chg=("SP_CHANGE", "sum"),
    settle=("SETTLE_PRICE", "first"),
).reset_index().sort_values("TRADE_DATE")
print("\nDaily agg (last 10):")
print(g.tail(10).to_string())

print("\nTop5 long holders on latest day:")
print(d.nlargest(5, "LONG_POSITION")[cols].to_string())

# probe other tabs report names from page bundle
for name in [
    "RPT_FUTU_POSITIONSTRUCT",
    "RPT_FUTU_POSITIONAVG",
    "RPT_FUTU_POSITIONBUILD",
    "RPT_FUTU_POSITIONPNL",
    "RPT_FUTU_POSITIONSUM",
    "RPT_FUTU_POSITIONRATIO",
]:
    p = {
        "reportName": name,
        "columns": "ALL",
        "pageNumber": "1",
        "pageSize": "3",
        "source": "WEB",
        "client": "WEB",
        "filter": '(SECURITY_CODE="FG2609")',
    }
    rr = requests.get(base, params=p, headers=headers, impersonate="chrome", timeout=10)
    jj = rr.json()
    dd = (jj.get("result") or {}).get("data")
    print(name, "->", "OK" if dd else jj.get("message", "")[:60])
