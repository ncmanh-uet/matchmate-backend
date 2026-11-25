import os
from collections.abc import Iterator
from datetime import datetime, timedelta, timezone
from pathlib import Path
from uuid import uuid4

import pytest
from alembic import command
from alembic.config import Config
from fastapi.testclient import TestClient
from sqlalchemy import Engine
from sqlalchemy.orm import Session, sessionmaker

from app import composition
from app.db.models import ParticipationModel, SessionModel, UserModel, VenueModel
from app.db.session import create_db_engine, create_session_factory
from app.features.auth import dependencies as auth_dependencies
from app.features.auth.tokens import JwtTokenIssuer
from app.main import app

TEST_SECRET = "matchmate-tests-only-secret-32-bytes"


@pytest.fixture
def client(monkeypatch, db_session_factory):
    """Mọi feature dùng cùng DB tạm; không gọi vào DB local khi test API."""
    monkeypatch.setenv("JWT_SECRET_KEY", TEST_SECRET)
    for module in (
        auth_dependencies,
        composition,
    ):
        monkeypatch.setattr(module, "_session_factory", db_session_factory)
    saved = app.dependency_overrides.copy()
    try:
        with TestClient(app, raise_server_exceptions=False) as test_client:
            yield test_client
    finally:
        app.dependency_overrides.clear()
        app.dependency_overrides.update(saved)


@pytest.fixture
def auth_headers():
    """Tạo JWT cho user đã có trong fixture, vẫn đi qua middleware thật."""

    def headers(user_id):
        token = JwtTokenIssuer(TEST_SECRET, ttl_seconds=3600).issue(user_id)
        return {"Authorization": f"Bearer {token.access_token}"}

    return headers


@pytest.fixture
def postgres(monkeypatch: pytest.MonkeyPatch) -> Iterator[Engine]:
    url = os.getenv("MATCHMATE_TEST_POSTGRES_URL")
    if not url:
        pytest.skip("Set MATCHMATE_TEST_POSTGRES_URL to a disposable PostgreSQL server")
    admin = create_db_engine(url)
    database = f"matchmate_test_{uuid4().hex}"
    engine = None
    created = False
    try:
        if admin.dialect.name != "postgresql":
            pytest.fail("MATCHMATE_TEST_POSTGRES_URL must point to PostgreSQL")
        with admin.connect().execution_options(isolation_level="AUTOCOMMIT") as connection:
            connection.exec_driver_sql(f"CREATE DATABASE {database}")
            created = True
        target_url = admin.url.set(database=database).render_as_string(hide_password=False)
        monkeypatch.setenv("DATABASE_URL", target_url)
        command.upgrade(Config("alembic.ini"), "head")
        engine = create_db_engine(target_url)
        yield engine
    finally:
        if engine is not None:
            engine.dispose()
        try:
            if created:
                with admin.connect().execution_options(isolation_level="AUTOCOMMIT") as connection:
                    connection.exec_driver_sql(f"DROP DATABASE {database}")
        finally:
            admin.dispose()


@pytest.fixture
def db_engine(tmp_path: Path, request: pytest.FixtureRequest) -> Iterator[Engine]:
    """Mặc định dùng một SQLite file riêng. Test có thể chọn PostgreSQL."""
    if getattr(request, "param", "sqlite") == "postgres":
        yield request.getfixturevalue("postgres")
        return

    engine = create_db_engine(f"sqlite:///{tmp_path / 'test.db'}")

    try:
        # Import models đã đăng ký đủ năm bảng vào cùng metadata.
        VenueModel.metadata.create_all(engine)
        yield engine
    finally:
        engine.dispose()


@pytest.fixture
def db_session_factory(db_engine: Engine) -> sessionmaker[Session]:
    """Cấp các DB Session cùng trỏ vào DB tạm của test."""
    return create_session_factory(db_engine)


@pytest.fixture
def db_session(
    db_session_factory: sessionmaker[Session],
) -> Iterator[Session]:
    """DB Session để chuẩn bị dữ liệu và kiểm kết quả."""
    with db_session_factory() as session:
        try:
            yield session
        finally:
            session.rollback()


@pytest.fixture
def game_data(db_session_factory: sessionmaker[Session]) -> dict[str, int]:
    """Hai kèo nối tiếp, mỗi kèo đã có host ACCEPTED; player chưa có lượt."""
    now = datetime(2030, 11, 1, 10, tzinfo=timezone.utc)
    with db_session_factory() as session:
        users = [
            UserModel(
                email=f"{name}@example.com",
                password_hash="test-only",
                display_name=name,
                skill_level=9,
                gender=gender,
                created_at=now,
                updated_at=now,
            )
            for name, gender in [("host", "MALE"), ("player", "FEMALE"), ("other", "MALE")]
        ]
        venue = VenueModel(name="Sân test", address="Địa chỉ test", district_code="CAU_GIAY")
        session.add_all([*users, venue])
        session.flush()
        games = [
            SessionModel(
                host_user_id=users[0].id,
                venue_id=venue.id,
                title=f"Kèo {index}",
                notes="",
                starts_at=now + timedelta(hours=2 + 2 * index),
                ends_at=now + timedelta(hours=4 + 2 * index),
                play_type="DOUBLES",
                fee_male_vnd=60000,
                fee_female_vnd=40000,
                min_level=8,
                max_level=10,
                gender_policy="ANY",
                status="OPEN",
                created_at=now,
            )
            for index in range(2)
        ]
        session.add_all(games)
        session.flush()
        for game in games:
            session.add(
                ParticipationModel(
                    user_id=users[0].id,
                    session_id=game.id,
                    status="ACCEPTED",
                    gender_at_join="MALE",
                    quoted_fee_vnd=60000,
                    created_at=now,
                    decided_at=now,
                )
            )
        session.commit()
        return dict(
            host=users[0].id,
            player=users[1].id,
            other=users[2].id,
            venue=venue.id,
            first=games[0].id,
            second=games[1].id,
        )
