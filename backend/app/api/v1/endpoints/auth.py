"""JWT Authentication endpoints – login, refresh, register, logout."""
from datetime import timedelta
from fastapi import APIRouter, Depends, HTTPException, status
from fastapi.security import OAuth2PasswordRequestForm
from sqlalchemy.ext.asyncio import AsyncSession
from app.db.session import get_db
from app.core.config import settings
from app.core.security import create_access_token, verify_password, get_password_hash
from app.schemas.auth import Token, UserCreate, UserResponse

router = APIRouter()

@router.post("/login", response_model=Token)
async def login(form_data: OAuth2PasswordRequestForm = Depends(), db: AsyncSession = Depends(get_db)):
    from app.services.user_service import UserService
    user = await UserService.authenticate(db, form_data.username, form_data.password)
    if not user:
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED,
                            detail="Incorrect credentials", headers={"WWW-Authenticate": "Bearer"})
    token = create_access_token({"sub": str(user.id), "org": str(user.organization_id)},
                                 expires_delta=timedelta(minutes=settings.ACCESS_TOKEN_EXPIRE_MINUTES))
    return {"access_token": token, "token_type": "bearer",
            "expires_in": settings.ACCESS_TOKEN_EXPIRE_MINUTES * 60}

@router.post("/register", response_model=UserResponse, status_code=status.HTTP_201_CREATED)
async def register(user_data: UserCreate, db: AsyncSession = Depends(get_db)):
    from app.services.user_service import UserService
    return await UserService.create(db, user_data)

@router.post("/refresh", response_model=Token)
async def refresh_token(refresh_token: str, db: AsyncSession = Depends(get_db)):
    from app.core.security import verify_token
    payload = verify_token(refresh_token)
    token = create_access_token({"sub": payload["sub"]})
    return {"access_token": token, "token_type": "bearer"}
