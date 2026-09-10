import duckdb

def run_analytics(con: duckdb.DuckDBPyConnection):
    return con.execute('SELECT 1').fetchall()
