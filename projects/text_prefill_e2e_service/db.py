import asyncpg

async def init_db(dsn: str) -> asyncpg.Pool:
    pool = await asyncpg.create_pool(dsn)
    async with pool.acquire() as conn:
        await conn.execute('CREATE EXTENSION IF NOT EXISTS vector;')
    return pool
