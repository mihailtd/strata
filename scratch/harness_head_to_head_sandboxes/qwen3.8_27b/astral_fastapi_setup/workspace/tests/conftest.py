"""Shared pytest fixtures: a fresh app + TestClient per test."""
from collections.abc import Iterator

import pytest
from fastapi.testclient import TestClient

from app.api.items import _store
from app.main import create_app


@pytest.fixture
def app():
    _store.clear()  # reset in-memory data between tests
    return create_app()


@pytest.fixture
def client(app) -> Iterator[TestClient]:
    with TestClient(app) as test_client:
        yield test_client
