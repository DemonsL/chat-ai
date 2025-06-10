from fastapi import APIRouter, Depends, HTTPException, status
from fastapi.security import OAuth2PasswordRequestForm

from app.api.dependencies import get_auth_service
from app.schemas.token import Token
from app.schemas.user import User, UserCreate
from app.services.auth_service import AuthService

router = APIRouter()


@router.post("/login", response_model=Token)
async def login(
    form_data: OAuth2PasswordRequestForm = Depends(),
    auth_service: AuthService = Depends(get_auth_service),
):
    """
    使用OAuth2 密码流获取JWT访问令牌

    - **username**: 用户名或邮箱
    - **password**: 密码
    """
    result = await auth_service.login(
        username=form_data.username, password=form_data.password
    )
    if not result:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="用户名或密码不正确",
            headers={"WWW-Authenticate": "Bearer"},
        )
    return result


@router.post("/register", response_model=User, status_code=status.HTTP_201_CREATED)
async def register(
    user_in: UserCreate,
    auth_service: AuthService = Depends(get_auth_service),
):
    """
    注册新用户

    - **username**: 用户名
    - **email**: 邮箱
    - **password**: 密码
    - **full_name**: 姓名（可选）
    """
    try:
        user = await auth_service.register(user_in)
        return user
    except ValueError as e:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=str(e),
        )
