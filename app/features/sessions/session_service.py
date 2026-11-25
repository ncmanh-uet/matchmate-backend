from typing import Any

from app.domain.enums import ParticipationStatus
from app.domain.ports import Clock, MatchingRequestWriter, UnitOfWork
from app.domain.time_rules import is_valid_time_window
from app.features.auth.contracts import PlayerProfileReader
from app.features.catalog.contracts import VenueStatusReader
from app.features.sessions.domain.policies import (
    require_host_eligibility,
    session_cancellation_time,
)
from app.features.sessions.session_contracts import (
    CreateSessionInput,
    SessionDetailReader,
    SessionError,
    SessionRepository,
)
from app.features.sessions.session_detail_service import build_session_detail


class SessionService:
    def __init__(
        self,
        uow: UnitOfWork,
        sessions: SessionRepository,
        details: SessionDetailReader,
        users: PlayerProfileReader,
        clock: Clock,
        venues: VenueStatusReader,
        matching_writer: MatchingRequestWriter,
    ) -> None:
        self.uow = uow
        self.sessions = sessions
        self.details = details
        self.users = users
        self.clock = clock
        self.venues = venues
        self.matching_writer = matching_writer

    def create(self, data: CreateSessionInput, host_id: int) -> dict[str, Any]:
        with self.uow.write():
            if not self.uow.lock_user(host_id):
                raise SessionError("UNAUTHORIZED", "Tài khoản không tồn tại")
            now = self.clock.now()

            try:
                valid_time = is_valid_time_window(data.starts_at, data.ends_at, now)
            except ValueError:
                valid_time = False

            if not valid_time:
                raise SessionError(
                    "INVALID_TIME_WINDOW",
                    "Kèo phải ở tương lai và dài từ 60 đến 240 phút",
                )

            active = self.venues.is_active(data.venue_id)
            if active is None:
                raise SessionError("VENUE_NOT_FOUND", "Không tìm thấy sân")
            if not active:
                raise SessionError("VENUE_INACTIVE", "Sân đang ngừng hoạt động")

            host = self.users.get_by_id(host_id)
            if host is None:
                raise SessionError("UNAUTHORIZED", "Tài khoản không tồn tại")
            require_host_eligibility(host, data)

            if self.sessions.has_accepted_overlap(host_id, data.starts_at, data.ends_at):
                raise SessionError("TIME_CONFLICT", "Host đã có lịch chồng giờ")

            session_id = self.sessions.create_with_host(data, host, now)
            self.matching_writer.close_overlapping(host_id, data.starts_at, data.ends_at, now)
            detail = build_session_detail(
                self.details.get_detail_data(session_id), host_id, self.clock
            )
            self.uow.commit()
            return detail

    def cancel(self, session_id: int, actor_id: int) -> dict[str, Any]:
        with self.uow.write():
            if not self.uow.lock_session(session_id):
                raise SessionError("SESSION_NOT_FOUND", "Không tìm thấy kèo")
            snapshot = self.details.get_detail_data(session_id)
            if snapshot is None:
                raise SessionError("SESSION_NOT_FOUND", "Không tìm thấy kèo")

            if snapshot["host_user_id"] == actor_id:
                active_statuses = (ParticipationStatus.PENDING, ParticipationStatus.ACCEPTED)
                affected_users = {
                    part["user_id"]
                    for part in snapshot["participations"]
                    if part["status"] in active_statuses
                }
                for user_id in sorted(affected_users):
                    self.uow.lock_user(user_id)

            data = snapshot["session"]
            cancelled_at = session_cancellation_time(
                host_id=snapshot["host_user_id"],
                actor_id=actor_id,
                status=data["status"],
                starts_at=data["starts_at"],
                clock=self.clock,
            )
            if cancelled_at is not None:
                self.sessions.cancel_active(session_id, cancelled_at)

            detail = build_session_detail(
                self.details.get_detail_data(session_id), actor_id, self.clock
            )
            self.uow.commit()
            return detail
