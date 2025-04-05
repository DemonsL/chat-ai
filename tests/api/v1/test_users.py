from fastapi import status
from fastapi.testclient import TestClient
from sqlalchemy.orm import Session

from app.core.config import settings
from app.db.models.user import User # Import user model if needed for asserts
from app.schemas.user import UserRead

# Note: Fixtures like `client`, `db_session`, `test_user`, `auth_headers`
# are automatically available due to conftest.py

def test_read_users_me_unauthenticated(client: TestClient):
    """Test getting current user without authentication."""
    response = client.get(f"{settings.API_V1_STR}/users/me")
    assert response.status_code == status.HTTP_401_UNAUTHORIZED

def test_read_users_me_authenticated(
    client: TestClient,
    test_user: User, # Fixture provides the created user
    auth_headers: dict[str, str] # Fixture provides auth headers
):
    """Test getting current user when authenticated."""
    response = client.get(f"{settings.API_V1_STR}/users/me", headers=auth_headers)
    assert response.status_code == status.HTTP_200_OK
    current_user = UserRead(**response.json())
    assert current_user.email == test_user.email
    assert current_user.full_name == test_user.full_name
    assert current_user.id == test_user.id

# Add more tests for other user endpoints (create, update, etc.)
# Example: Test creating a user (usually needs superuser or specific permissions)
# def test_create_user(client: TestClient, superuser_auth_headers: dict[str, str]):
#     user_data = {"email": "newuser@example.com", "password": "newpassword", "full_name": "New User"}
#     response = client.post(f"{settings.API_V1_STR}/users/", headers=superuser_auth_headers, json=user_data)
#     assert response.status_code == status.HTTP_201_CREATED # Or 200 depending on your impl
#     # Add assertions about the response data and database state