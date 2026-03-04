"""Tests for metrics tracking: models, store methods, flow integration."""

from __future__ import annotations

from datetime import date
from unittest.mock import patch

from elephant.data.models import DailyMetrics, MetricsFile
from elephant.data.store import DataStore


class TestMetricsStore:
    def test_read_empty_metrics(self, data_dir):
        store = DataStore(data_dir)
        store.initialize()
        metrics = store.read_metrics()
        assert metrics.days == []

    def test_write_and_read_metrics(self, data_dir):
        store = DataStore(data_dir)
        store.initialize()
        metrics = MetricsFile(
            days=[DailyMetrics(date=date(2026, 3, 1), memories_created=5, digests_sent=1)]
        )
        store.write_metrics(metrics)
        loaded = store.read_metrics()
        assert len(loaded.days) == 1
        assert loaded.days[0].memories_created == 5
        assert loaded.days[0].digests_sent == 1

    def test_increment_metric_creates_today_entry(self, data_dir):
        store = DataStore(data_dir)
        store.initialize()
        today = date(2026, 3, 4)
        with patch("elephant.data.store._date") as mock_date:
            mock_date.today.return_value = today
            mock_date.side_effect = lambda *a, **kw: date(*a, **kw)
            store.increment_metric("memories_created")
        metrics = store.read_metrics()
        assert len(metrics.days) == 1
        assert metrics.days[0].date == today
        assert metrics.days[0].memories_created == 1

    def test_increment_metric_accumulates(self, data_dir):
        store = DataStore(data_dir)
        store.initialize()
        today = date(2026, 3, 4)
        with patch("elephant.data.store._date") as mock_date:
            mock_date.today.return_value = today
            mock_date.side_effect = lambda *a, **kw: date(*a, **kw)
            store.increment_metric("memories_created")
            store.increment_metric("memories_created", 2)
            store.increment_metric("digests_sent")
        metrics = store.read_metrics()
        assert len(metrics.days) == 1
        assert metrics.days[0].memories_created == 3
        assert metrics.days[0].digests_sent == 1

    def test_increment_metric_separate_days(self, data_dir):
        store = DataStore(data_dir)
        store.initialize()
        # Manually write one day
        metrics = MetricsFile(
            days=[DailyMetrics(date=date(2026, 3, 3), memories_created=2)]
        )
        store.write_metrics(metrics)
        # Increment for today (different day)
        today = date(2026, 3, 4)
        with patch("elephant.data.store._date") as mock_date:
            mock_date.today.return_value = today
            mock_date.side_effect = lambda *a, **kw: date(*a, **kw)
            store.increment_metric("digests_sent")
        metrics = store.read_metrics()
        assert len(metrics.days) == 2
        assert metrics.days[0].date == date(2026, 3, 3)
        assert metrics.days[1].date == date(2026, 3, 4)
        assert metrics.days[1].digests_sent == 1
