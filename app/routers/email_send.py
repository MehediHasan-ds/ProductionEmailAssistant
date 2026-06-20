"""Send endpoint: sends an email via SMTP."""
from __future__ import annotations

from fastapi import APIRouter

from app.schemas.api import SendRequest, SendResponse
from app.services.email_send_service import send_email

router = APIRouter()


@router.post("/send", response_model=SendResponse)
async def send(request: SendRequest) -> SendResponse:
    return await send_email(request)
