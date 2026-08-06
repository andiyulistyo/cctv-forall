"""Authentication routes."""
from __future__ import annotations

from fastapi import APIRouter, HTTPException, status

from ..auth import authenticate, create_access_token
from ..schemas import LoginRequest, TokenResponse

router = APIRouter(prefix="/auth", tags=["auth"])


@router.post("/login", response_model=TokenResponse)
def login(payload: LoginRequest) -> TokenResponse:
    user = authenticate(payload.username, payload.password)
    if user is None:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED, detail="Invalid credentials"
        )
    token = create_access_token(user.username)
    return TokenResponse(access_token=token)
