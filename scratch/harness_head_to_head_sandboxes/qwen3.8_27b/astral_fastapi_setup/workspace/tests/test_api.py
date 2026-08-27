"""Smoke + behaviour tests for the API surface."""
from __future__ import annotations

import re

from fastapi import status


def test_health(client) -> None:
    resp = client.get("/health")
    assert resp.status_code == 200
    body = resp.json()
    assert body["status"] == "ok"
    assert body["app"] == "fastapi-backend"
    assert body["version"]
    assert re.match(r"Z", body["time"])  # ISO-8601 UTC


def test_openapi_lists_routes(client) -> None:
    paths = set(client.get("/openapi.json").json()["paths"])
    assert {"/health", "/items", "/items/{item_id}"} <= paths


def test_create_list_get_delete_item(client) -> None:
    created = client.post("/items", json={"name": "Coffee", "price": 4.5, "sku": "cof-1"})
    assert created.status_code == status.HTTP_201_CREATED
    item = created.json()
    assert item["name"] == "Coffee"
    item_id = item["id"]

    # listed
    listing = client.get("/items").json()
    assert any(i["id"] == item_id for i in listing)

    # got by id
    fetched = client.get(f"/items/{item_id}")
    assert fetched.status_code == 200
    assert fetched.json() == item

    # created_at is ISO-8601
    assert item["created_at"].endswith("Z")

    # deleted -> 204; then 404
    assert client.delete(f"/items/{item_id}").status_code == status.HTTP_204_NO_CONTENT
    assert client.get(f"/items/{item_id}").status_code == status.HTTP_404_NOT_FOUND


def test_validation_error_is_422(client) -> None:
    resp = client.post("/items", json={"name": "", "price": -1})
    assert resp.status_code == status.HTTP_422_UNPROCESSABLE_ENTITY
    assert resp.json()["detail"]


def test_missing_item_returns_apperror_envelope(client) -> None:
    resp = client.get("/items/00000000-0000-4000-8000-000000000000")
    assert resp.status_code == status.HTTP_404_NOT_FOUND
    body = resp.json()
    assert body["error"] == "AppError"
    assert "not found" in body["message"]
