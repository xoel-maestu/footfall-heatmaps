"""Use case: R7, drop slots older than the retention."""

RETENTION_DAYS = 90
DAY_S = 86_400


def purge_slots(store, now, retention_days=RETENTION_DAYS):
    """Slots that started more than retention_days before now; how many rows went."""
    return store.purge(int(now - retention_days * DAY_S))
