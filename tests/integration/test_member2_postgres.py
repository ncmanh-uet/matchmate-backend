import sqlite3
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timedelta, timezone
from threading import Event

import pytest
from alembic import command
from alembic.config import Config
from sqlalchemy import event, func, select, update
from sqlalchemy.exc import IntegrityError

from app.db.models import Base, MatchingRequestModel, ParticipationModel, SessionModel, VenueModel
from app.db.session import create_session_factory
from app.db.unit_of_work import SqlAlchemyUnitOfWork
from app.domain.errors import DatabaseBusyError, DatabaseTimeoutError
from scripts.import_sqlite_to_postgres import import_sqlite
from scripts.prepare_member2_phase1 import prepare


@pytest.fixture
def source(tmp_path):
    path = tmp_path / "fixture"
    prepare(path)
    return path / "migration-source.db"


@pytest.fixture
def db_session_factory(postgres):
    return create_session_factory(postgres)


def test_migration_import_sequences_and_constraints(postgres, source):
    assert import_sqlite(source, postgres) == {
        "venues": 1,
        "users": 3,
        "sessions": 1,
        "participations": 4,
        "matching_requests": 3,
    }
    command.check(Config("alembic.ini"))
    with pytest.raises(ValueError, match="existing rows"):
        import_sqlite(source, postgres)

    with postgres.connect() as connection:
        for table in Base.metadata.sorted_tables:
            highest_id = connection.scalar(select(func.max(table.c.id)))
            seq = connection.exec_driver_sql(
                "SELECT pg_get_serial_sequence(%s, 'id')", (table.name,)
            ).scalar_one()
            assert (
                connection.exec_driver_sql("SELECT nextval(%s)", (seq,)).scalar_one() > highest_id
            )

    for model, active_status, index in (
        (ParticipationModel, "ACCEPTED", "uq_participations_active"),
        (MatchingRequestModel, "ACTIVE", "uq_matching_requests_active_user"),
    ):
        table = model.__table__
        with pytest.raises(IntegrityError) as duplicate, postgres.begin() as connection:
            active = (
                connection.execute(select(table).where(table.c.status == active_status).limit(1))
                .mappings()
                .one()
            )
            connection.execute(
                table.insert(), {key: val for key, val in active.items() if key != "id"}
            )
        assert duplicate.value.orig.diag.constraint_name == index


def test_import_rolls_back_all_tables_on_failure(postgres, source):
    with sqlite3.connect(source) as connection:
        connection.execute(
            "UPDATE matching_requests SET id = ? WHERE id = (SELECT max(id) FROM "
            "matching_requests)",
            (2**63 - 1,),
        )
    with pytest.raises(ValueError, match="No automatic IDs"):
        import_sqlite(source, postgres)
    with postgres.connect() as connection:
        for table in Base.metadata.sorted_tables:
            assert connection.scalar(select(func.count()).select_from(table)) == 0


@pytest.mark.parametrize("lock_method", ["lock_session", "lock_user"])
def test_row_lock_uses_independent_connections_and_rolls_back(
    postgres, source, db_session_factory, lock_method
):
    import_sqlite(source, postgres)
    model = Base.metadata.tables["sessions" if lock_method == "lock_session" else "users"]
    first = SqlAlchemyUnitOfWork(db_session_factory, DatabaseBusyError)
    second = SqlAlchemyUnitOfWork(db_session_factory, DatabaseBusyError)
    with first.write():
        row_id = first.session.scalar(select(func.min(model.c.id)))
        first_pid = first.session.connection().exec_driver_sql("SELECT pg_backend_pid()").scalar()
        assert getattr(first, lock_method)(row_id) is True
        assert getattr(first, lock_method)(2**63 - 1) is False
        with pytest.raises(DatabaseBusyError), second.write():
            connection = second.session.connection()
            assert connection.exec_driver_sql("SELECT pg_backend_pid()").scalar() != first_pid
            connection.exec_driver_sql("SET LOCAL lock_timeout = '150ms'")
            second.session.add(VenueModel(name="Rollback", address="Test", district_code="TEST"))
            second.session.flush()
            getattr(second, lock_method)(row_id)

    with second.write():
        assert getattr(second, lock_method)(row_id) is True
        assert second.session.scalar(select(func.count()).select_from(VenueModel)) == 1


def test_postgres_snapshot_and_statement_timeout(db_session_factory):
    reader = SqlAlchemyUnitOfWork(db_session_factory, DatabaseBusyError)
    writer = SqlAlchemyUnitOfWork(db_session_factory, DatabaseBusyError)
    count = select(func.count()).select_from(VenueModel)
    with reader.read():
        assert reader.session.scalar(count) == 0
        assert (
            reader.session.connection().exec_driver_sql("SHOW transaction_read_only").scalar()
            == "on"
        )
        with writer.write():
            writer.session.add(VenueModel(name="New", address="Test", district_code="TEST"))
            writer.commit()
        assert reader.session.scalar(count) == 0
    with reader.read():
        assert reader.session.scalar(count) == 1
    with pytest.raises(DatabaseTimeoutError), writer.write():
        connection = writer.session.connection()
        connection.exec_driver_sql("SET LOCAL statement_timeout = '100ms'")
        connection.exec_driver_sql("SELECT pg_sleep(1)")


def test_session_detail_keeps_one_snapshot_during_cancellation(
    postgres, client, game_data, auth_headers
):
    host = auth_headers(game_data["host"])
    player = auth_headers(game_data["player"])
    session_id = game_data["first"]
    url = f"/sessions/{session_id}"
    pending = client.post("/participations", headers=player, json={"session_id": session_id})
    assert pending.status_code == 201, pending.text
    approved = client.post(
        f"/participations/{pending.json()['id']}/decision",
        headers=host,
        json={"decision": "APPROVE"},
    )
    assert approved.status_code == 200, approved.text
    before = client.get(url, headers=host)
    assert before.status_code == 200, before.text
    assert before.json()["status"] == "OPEN"
    assert before.json()["accepted_count"] == 2

    reading_participations = Event()
    resume_detail = Event()

    def pause_participation_query(_conn, _cursor, statement, _params, _context, _many):
        if "FROM participations JOIN users" in statement and not reading_participations.is_set():
            reading_participations.set()
            assert resume_detail.wait(timeout=10), "Cancellation did not finish"

    event.listen(postgres, "before_cursor_execute", pause_participation_query)
    try:
        with ThreadPoolExecutor(max_workers=1) as pool:
            in_flight = pool.submit(client.get, url, headers=host)
            try:
                assert reading_participations.wait(timeout=5), (
                    "Detail never reached the roster query"
                )
                cancelled = client.delete(url, headers=host)
                assert cancelled.status_code == 200, cancelled.text
            finally:
                resume_detail.set()
            during = in_flight.result(timeout=5)
    finally:
        event.remove(postgres, "before_cursor_execute", pause_participation_query)

    assert during.status_code == 200, during.text
    assert during.json() == before.json()
    after = client.get(url, headers=host)
    assert after.status_code == 200, after.text
    snapshot = after.json()
    assert snapshot == cancelled.json()
    assert snapshot["status"] == "CANCELLED"
    assert snapshot["accepted_count"] == 0
    assert [part["status"] for part in snapshot["management"]["participations"]] == [
        "CANCELLED",
        "CANCELLED",
    ]


def test_approval_reloads_participation_after_session_lock_wait(
    postgres, client, game_data, auth_headers
):
    player = auth_headers(game_data["player"])
    host = auth_headers(game_data["host"])
    session_id = game_data["first"]
    pending = client.post("/participations", headers=player, json={"session_id": session_id})
    assert pending.status_code == 201, pending.text
    participation_id = pending.json()["id"]
    approval_waiting = Event()

    def observe_session_lock(_conn, _cursor, statement, _params, _context, _many):
        if "FROM sessions" in statement and "FOR UPDATE" in statement:
            approval_waiting.set()

    with ThreadPoolExecutor(max_workers=1) as pool:
        with postgres.begin() as blocker:
            blocker.execute(
                select(SessionModel.id).where(SessionModel.id == session_id).with_for_update()
            )
            event.listen(postgres, "before_cursor_execute", observe_session_lock)
            try:
                approval = pool.submit(
                    client.post,
                    f"/participations/{participation_id}/decision",
                    headers=host,
                    json={"decision": "APPROVE"},
                )
                assert approval_waiting.wait(timeout=3)
                blocker.execute(
                    update(ParticipationModel)
                    .where(ParticipationModel.id == participation_id)
                    .values(status="CANCELLED", cancelled_at=datetime.now(timezone.utc))
                )
            finally:
                event.remove(postgres, "before_cursor_execute", observe_session_lock)

        result = approval.result(timeout=5)

    assert result.status_code == 409, result.text
    assert result.json()["error"]["code"] == "INVALID_STATE"
    with postgres.connect() as connection:
        status = connection.scalar(
            select(ParticipationModel.status).where(ParticipationModel.id == participation_id)
        )
    assert status == "CANCELLED"


def test_api_lifecycle_and_matching_on_migrated_postgres(postgres, source, client):
    import_sqlite(source, postgres)
    response = client.get("/venues")
    assert response.status_code == 200
    assert response.json()["total"] == 1
    with postgres.connect() as connection:
        venue_id = connection.scalar(select(VenueModel.id))

    def register_and_login(name, gender):
        credentials = {"email": f"{name}@example.com", "password": "test-only-password"}
        payload = credentials | {"display_name": name, "skill_level": "TB", "gender": gender}
        assert client.post("/auth/register", json=payload).status_code == 201
        login = client.post("/auth/login", json=credentials)
        assert login.status_code == 200, login.text
        return {"Authorization": "Bearer " + login.json()["access_token"]}

    host = register_and_login("postgres-host", "MALE")
    player = register_and_login("postgres-player", "FEMALE")
    start = datetime.now(timezone.utc) + timedelta(days=1)
    slot = {
        "venue_id": venue_id,
        "starts_at": start.isoformat(),
        "ends_at": (start + timedelta(hours=2)).isoformat(),
        "play_type": "SINGLES",
    }
    game = client.post(
        "/sessions",
        headers=host,
        json=slot
        | {
            "title": "Kèo kiểm thử PostgreSQL",
            "fee_male_vnd": 60000,
            "fee_female_vnd": 40000,
            "min_level": "TB_MINUS",
            "max_level": "TB_PLUS",
        },
    )
    assert game.status_code == 201, game.text
    game_id = game.json()["id"]
    matching = client.post(
        "/matching/requests",
        headers=player,
        json=slot
        | {
            "proposed_fee_male_vnd": 60000,
            "proposed_fee_female_vnd": 40000,
            "budget_max_vnd": 60000,
        },
    )
    assert matching.status_code == 201, matching.text
    matching_id = matching.json()["request"]["id"]
    pending = client.post("/participations", headers=player, json={"session_id": game_id})
    assert pending.status_code == 201, pending.text
    approved = client.post(
        f"/participations/{pending.json()['id']}/decision",
        headers=host,
        json={"decision": "APPROVE"},
    )
    assert approved.status_code == 200, approved.text
    detail = client.get(f"/sessions/{game_id}", headers=player)
    assert detail.status_code == 200, detail.text
    assert detail.json()["accepted_count"] == 2
    with postgres.connect() as connection:
        closed = connection.execute(
            select(MatchingRequestModel.status, MatchingRequestModel.closed_reason).where(
                MatchingRequestModel.id == matching_id
            )
        ).one()
    assert closed == ("CLOSED", "CONFIRMED")
