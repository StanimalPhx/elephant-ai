"""DatabaseInstance: per-database object graph."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from elephant.brain.question_manager import QuestionManager
    from elephant.config import ScheduleConfig
    from elephant.data.store import DataStore
    from elephant.flows.anytime_log import AnytimeLogFlow
    from elephant.flows.evening_checkin import EveningCheckinFlow
    from elephant.flows.integrity_check import IntegrityCheckFlow
    from elephant.flows.monthly_report import MonthlyReportFlow
    from elephant.flows.morning_digest import MorningDigestFlow
    from elephant.flows.weekly_recap import WeeklyRecapFlow
    from elephant.flows.year_in_review import YearInReviewFlow
    from elephant.git_ops import GitRepo
    from elephant.messaging.base import MessagingClient


@dataclass
class DatabaseInstance:
    """Bundles all per-database objects for isolation."""

    name: str
    auth_secret: str
    store: DataStore
    git: GitRepo
    messaging: MessagingClient
    anytime: AnytimeLogFlow
    morning: MorningDigestFlow
    evening: EveningCheckinFlow
    question_mgr: QuestionManager
    monthly_report: MonthlyReportFlow
    weekly_recap: WeeklyRecapFlow
    year_in_review: YearInReviewFlow
    integrity_check: IntegrityCheckFlow
    schedule: ScheduleConfig
