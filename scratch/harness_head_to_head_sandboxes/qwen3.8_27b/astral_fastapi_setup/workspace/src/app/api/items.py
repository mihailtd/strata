"""In-memory item collection (placeholder for a real datastore)."""
from datetime import UTC, datetime
from uuid import UUID, uuid4

from fastapi import APIRouter, Request
from fastapi import status

from app.errors import AppError
from app.schemas import Item, ItemCreate

router = APIRouter(prefix="/items", tags=["items"])

# TODO: replace with a real datastore (SQLAlchemy/SQLModel, Redis, ...).
_store: dict[UUID, Item] = {}


@router.get("", response_model=list[Item])
async def list_items(request: Request) -> list[Item]:
    """List all items."""
    return sorted(_store.values(), key=lambda i: i.created_at)


@router.post("", response_model=Item, status_code=status.HTTP_201_CREATED)
async def create_item(payload: ItemCreate, request: Request) -> Item:
    """Create a new item."""
    item = Item(**payload.model_dump(), id=uuid4(), created_at=datetime.now(UTC))
    _store[item.id] = item
    return item


@router.get("/{item_id}", response_model=Item)
async def get_item(item_id: UUID, request: Request) -> Item:
    """Fetch one item by id."""
    item = _store.get(item_id)
    if item is None:
        raise AppError(f"item {item_id} not found", status_code=status.HTTP_404_NOT_FOUND)
    return item


@router.delete("/{item_id}", status_code=status.HTTP_204_NO_CONTENT)
async def delete_item(item_id: UUID, request: Request) -> None:
    """Delete an item by id (204 No Content on success)."""
    if _store.pop(item_id, None) is None:
        raise AppError(f"item {item_id} not found", status_code=status.HTTP_404_NOT_FOUND)
