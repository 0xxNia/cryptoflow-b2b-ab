import duckdb

con = duckdb.connect("data/cryptoflow.duckdb", read_only=True)

result = con.execute("""
    SELECT
        trading_style,
        wallet_size,
        SUM(fee_usd) AS total_revenue,
        ROUND(SUM(fee_usd) * 100.0 / SUM(SUM(fee_usd)) OVER (), 1) AS pct_of_total
    FROM trades
    GROUP BY trading_style, wallet_size
    ORDER BY total_revenue DESC
""").df()

print(result)

con.close()