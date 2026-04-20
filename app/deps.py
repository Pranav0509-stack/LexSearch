"""Shared dependencies: Redis, S3, DB session injection."""

from functools import lru_cache
from typing import AsyncIterator

import redis.asyncio as aioredis
import s3fs
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker, create_async_engine

from app.config import get_settings


@lru_cache
def get_redis() -> aioredis.Redis:
    return aioredis.from_url(get_settings().redis_url, decode_responses=True)


@lru_cache
def get_s3() -> s3fs.S3FileSystem:
    # Anonymous — we only read public judgment buckets with this handle.
    # Authenticated writes (nyayasathi-prod) go through boto3, not s3fs.
    return s3fs.S3FileSystem(anon=True)


@lru_cache
def _engine():
    return create_async_engine(get_settings().database_url, pool_size=20, max_overflow=10)


@lru_cache
def _session_maker() -> async_sessionmaker[AsyncSession]:
    return async_sessionmaker(_engine(), expire_on_commit=False)


async def get_db() -> AsyncIterator[AsyncSession]:
    async with _session_maker()() as session:
        yield session
