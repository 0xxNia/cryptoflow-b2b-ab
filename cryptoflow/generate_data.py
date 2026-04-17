"""CryptoFlow v2 — Fast batch generator with personas."""
import pandas as pd
import numpy as np
import duckdb, os
from datetime import datetime, timedelta

np.random.seed(42)
START_DATE = datetime(2023, 1, 1)
END_DATE   = datetime(2024, 6, 30)
N_USERS    = 50_000
OUTPUT_DIR = "data"
os.makedirs(OUTPUT_DIR, exist_ok=True)

STYLE_BASE = {
    "hodler":  {"trades_week": 1,   "avg_volume": 5000, "churn_month": 0.03, "fee_elast": 0.1},
    "swing":   {"trades_week": 8,   "avg_volume": 2000, "churn_month": 0.05, "fee_elast": 0.8},
    "scalper": {"trades_week": 30,  "avg_volume": 800,  "churn_month": 0.12, "fee_elast": 4.5},
}
STYLE_W   = {"hodler": 0.65, "swing": 0.30, "scalper": 0.05}
RISK_MULT  = {"conservative":{"t":1.0,"v":1.0,"c":1.0},"moderate":{"t":1.5,"v":1.2,"c":1.5},"degen":{"t":2.5,"v":3.0,"c":4.0}}
RISK_W    = {"conservative":0.50,"moderate":0.35,"degen":0.15}
WALLET_MULT= {"minnow":{"t":1.0,"v":0.2,"c":1.3},"dolphin":{"t":1.0,"v":1.0,"c":1.0},"whale":{"t":0.5,"v":40.0,"c":0.4}}
WALLET_W  = {"minnow":0.80,"dolphin":0.18,"whale":0.02}
CHURN_MULT = {"sticky":{"t":1.0,"c":0.3,"fe":0.2},"neutral":{"t":1.0,"c":1.0,"fe":1.0},"mercenary":{"t":1.2,"c":3.5,"fe":3.0}}
CHURN_W   = {"sticky":0.20,"neutral":0.45,"mercenary":0.35}
TIER_FEE  = {"free":0.0025,"basic":0.0018,"pro":0.0010}
COINS     = ["BTC","ETH","SOL","BNB","DOGE","XRP","MATIC","AVAX"]
COIN_W    = np.array([0.30,0.25,0.15,0.10,0.08,0.06,0.04,0.02])
CHANNELS  = ["organic","paid_google","paid_meta","referral","influencer"]
CHANNEL_W = [0.35,0.25,0.15,0.15,0.10]
COUNTRIES = ["US","UK","DE","BR","IN","NG","SG","PH","TR","MX"]
COUNTRY_W = [0.28,0.12,0.08,0.07,0.09,0.06,0.07,0.06,0.09,0.08]
AB_START  = datetime(2023,7,1)
AB_END    = datetime(2023,10,31)
AB_FEE    = 0.0015
TOTAL_DAYS= (END_DATE-START_DATE).days

# ── USERS ─────────────────────────────────────────────────────────────────────
print("Generating users...")
day_idx = np.arange(TOTAL_DAYS)
growth  = np.exp(day_idx/(TOTAL_DAYS*0.6))-1; growth/=growth.sum()
reg_day = np.random.choice(day_idx, N_USERS, p=growth)

styles  = np.random.choice(list(STYLE_W), N_USERS, p=list(STYLE_W.values()))
risks   = np.random.choice(list(RISK_MULT), N_USERS, p=list(RISK_W.values()))
wallets = np.random.choice(list(WALLET_MULT), N_USERS, p=list(WALLET_W.values()))
churns  = np.random.choice(list(CHURN_MULT), N_USERS, p=list(CHURN_W.values()))

# Compute all params as arrays
tw_arr = np.array([min(STYLE_BASE[s]["trades_week"]*RISK_MULT[r]["t"]*WALLET_MULT[w]["t"]*CHURN_MULT[c]["t"],80)
                   for s,r,w,c in zip(styles,risks,wallets,churns)])
av_arr = np.array([STYLE_BASE[s]["avg_volume"]*RISK_MULT[r]["v"]*WALLET_MULT[w]["v"]
                   for s,r,w in zip(styles,risks,wallets)])
cm_arr = np.array([min(STYLE_BASE[s]["churn_month"]*RISK_MULT[r]["c"]*WALLET_MULT[w]["c"]*CHURN_MULT[c]["c"],0.95)
                   for s,r,w,c in zip(styles,risks,wallets,churns)])
fe_arr = np.array([STYLE_BASE[s]["fee_elast"]*CHURN_MULT[c]["fe"]
                   for s,c in zip(styles,churns)])

tiers = []
for i in range(N_USERS):
    if wallets[i]=="whale": t=np.random.choice(["free","basic","pro"],p=[0.05,0.20,0.75])
    elif styles[i]=="scalper": t=np.random.choice(["free","basic","pro"],p=[0.10,0.30,0.60])
    elif risks[i]=="degen": t=np.random.choice(["free","basic","pro"],p=[0.40,0.45,0.15])
    else: t=np.random.choice(["free","basic","pro"],p=[0.65,0.25,0.10])
    tiers.append(t)
tiers = np.array(tiers)

channels = np.random.choice(CHANNELS, N_USERS, p=CHANNEL_W)
countries= np.random.choice(COUNTRIES, N_USERS, p=COUNTRY_W)

ab_group = np.full(N_USERS, None, dtype=object)
reg_dates_dt = np.array([START_DATE+timedelta(days=int(d)) for d in reg_day])
mask_ab = (reg_dates_dt >= AB_START) & (reg_dates_dt <= AB_END)
ab_group[mask_ab] = np.random.choice(["control","treatment"], mask_ab.sum(), p=[0.5,0.5])

users = pd.DataFrame({
    "user_id":       [f"u_{i:06d}" for i in range(N_USERS)],
    "registered_at": reg_dates_dt,
    "reg_day_offset":reg_day,
    "channel":channels,"country":countries,"tier":tiers,
    "trading_style":styles,"risk_profile":risks,
    "wallet_size":wallets,"churn_sensitivity":churns,
    "trades_per_week":tw_arr,"avg_volume_usd":av_arr,
    "churn_per_month":cm_arr,"fee_elasticity":fe_arr,
    "ab_group":ab_group,
})
print(f"  Users: {N_USERS:,} | A/B: {mask_ab.sum():,}")

# ── TRADES — fully vectorized ─────────────────────────────────────────────────
print("Generating trades (batch mode)...")

# For each user: expected total trades = trades_per_week * expected_active_weeks
# Expected active weeks given weekly churn p: E[weeks] = 1/p (geometric)
churn_week = 1 - (1-cm_arr)**(1/4.33)

# Treatment boosts
is_treatment = (ab_group == "treatment")
tw_effective = tw_arr.copy()
tw_effective[is_treatment] = np.minimum(
    tw_arr[is_treatment] * (1 + fe_arr[is_treatment] * 0.4), 80
)
churn_week[is_treatment] *= np.maximum(1 - fe_arr[is_treatment]*0.05, 0.6)

# Expected active weeks (capped by time available)
days_avail = np.maximum(TOTAL_DAYS - reg_day, 0)
max_weeks  = np.minimum(days_avail // 7, 78).astype(int)
exp_active_weeks = np.minimum(1/np.maximum(churn_week,0.01), max_weeks.astype(float))

# Expected total trades per user
exp_trades = tw_effective * exp_active_weeks
total_exp  = int(exp_trades.sum())
print(f"  Expected trades: ~{total_exp:,}")

# Generate trades by sampling users proportionally
trade_rows = []
BATCH = 200_000

# Generate in user batches to keep memory manageable
for batch_start in range(0, N_USERS, 5000):
    batch_end = min(batch_start+5000, N_USERS)
    batch_idx = np.arange(batch_start, batch_end)

    for ui in batch_idx:
        u_tw   = tw_effective[ui]
        u_av   = av_arr[ui]
        u_cw   = churn_week[ui]
        u_mw   = max_weeks[ui]
        u_reg  = reg_day[ui]
        u_tier = tiers[ui]
        u_ab   = ab_group[ui]
        u_st   = styles[ui]
        u_ri   = risks[ui]
        u_wa   = wallets[ui]

        if u_mw == 0: continue

        # Simulate weeks until churn using geometric distribution
        if u_cw >= 1: n_active_weeks = 0
        elif u_cw <= 0: n_active_weeks = u_mw
        else: n_active_weeks = min(np.random.geometric(u_cw), u_mw)

        if n_active_weeks == 0: continue

        # Total trades across all active weeks
        total_trades = np.random.poisson(u_tw * n_active_weeks)
        if total_trades == 0: continue
        total_trades = min(total_trades, n_active_weeks * 50)  # cap

        # Sample random timestamps within active period
        active_days = n_active_weeks * 7
        trade_offsets = np.random.randint(0, active_days*24*60, total_trades)
        trade_days_offset = u_reg + trade_offsets // (24*60)

        valid = trade_days_offset < TOTAL_DAYS
        if not valid.any(): continue
        trade_days_offset = trade_days_offset[valid]
        trade_offsets     = trade_offsets[valid]
        n = len(trade_days_offset)

        volumes = np.minimum(
            np.random.lognormal(np.log(max(u_av,1)), 0.8, n),
            2_000_000
        )
        fee_rate = AB_FEE if u_ab=="treatment" else TIER_FEE[u_tier]
        coins    = np.random.choice(COINS, n, p=COIN_W)
        sides    = np.random.choice(["buy","sell"], n)

        for j in range(n):
            ts = START_DATE + timedelta(minutes=int(trade_offsets[j]))
            if ts > END_DATE: continue
            trade_rows.append((
                f"u_{ui:06d}", ts, coins[j], sides[j],
                round(float(volumes[j]),2),
                round(float(volumes[j]*fee_rate),4),
                u_tier, u_st, u_ri, u_wa
            ))

trades = pd.DataFrame(trade_rows, columns=[
    "user_id","traded_at","coin","side","volume_usd","fee_usd",
    "tier_at_trade","trading_style","risk_profile","wallet_size"
])
trades["trade_id"] = [f"t_{i:08d}" for i in range(len(trades))]
trades = trades.sort_values("traded_at").reset_index(drop=True)
print(f"  Trades generated: {len(trades):,}")

# ── SUBSCRIPTIONS ─────────────────────────────────────────────────────────────
print("Subscriptions...")
sub_rows=[]
for i,(tier,wa,st,reg) in enumerate(zip(tiers,wallets,styles,reg_dates_dt)):
    uid=f"u_{i:06d}"
    sub_rows.append({"user_id":uid,"tier":tier,"started_at":reg,"ended_at":None,"event_type":"initial"})
    prob=0.40 if wa=="whale" else (0.30 if st=="scalper" else 0.08)
    if tier=="free" and np.random.rand()<prob:
        d=reg+timedelta(days=int(np.random.exponential(30)))
        if d<=END_DATE:
            sub_rows[-1]["ended_at"]=d
            sub_rows.append({"user_id":uid,"tier":"pro" if wa=="whale" else "basic","started_at":d,"ended_at":None,"event_type":"upgrade"})
subscriptions=pd.DataFrame(sub_rows)

ab_df=users[users["ab_group"].notna()][
    ["user_id","ab_group","registered_at","trading_style","risk_profile","wallet_size","churn_sensitivity","fee_elasticity"]
].copy().rename(columns={"registered_at":"assigned_at"})
ab_df["experiment_id"]="fee_reduction_2023q3"

# ── SAVE ──────────────────────────────────────────────────────────────────────
print("Saving...")
users.drop(columns=["reg_day_offset"],errors="ignore").to_csv(f"{OUTPUT_DIR}/users.csv",index=False)
trades.to_csv(f"{OUTPUT_DIR}/trades.csv",index=False)
subscriptions.to_csv(f"{OUTPUT_DIR}/subscriptions.csv",index=False)
ab_df.to_csv(f"{OUTPUT_DIR}/ab_assignments.csv",index=False)

con=duckdb.connect(f"{OUTPUT_DIR}/cryptoflow.duckdb")
for tbl in ["users","trades","subscriptions","ab_assignments"]:
    con.execute(f"CREATE OR REPLACE TABLE {tbl} AS SELECT * FROM read_csv('data/{tbl}.csv')")
con.close()

print("\n"+"="*55)
print("  CryptoFlow v2 — Done!")
print("="*55)
print(f"  Users:       {len(users):,}")
print(f"  Trades:      {len(trades):,}")
print(f"  Revenue:     ${trades['fee_usd'].sum():,.0f}")
print(f"  Trades/user: {len(trades)/len(users):.1f}")
print("\nStyle breakdown:")
print(users["trading_style"].value_counts())
print("\nRevenue by style x wallet:")
top=trades.groupby(["trading_style","wallet_size"])["fee_usd"].sum().sort_values(ascending=False).head(6)
for k,v in top.items(): print(f"  {k[0]:10} x {k[1]:8}: ${v:>12,.0f}")