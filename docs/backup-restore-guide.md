# Backup & Disaster Recovery Guide — ConsciousAI Journal V2

This guide defines the backup, validation, and disaster recovery procedures for ConsciousAI Journal V2.

## 1. Backup Strategy Overview

| Asset | Mechanism | Frequency | Retention |
| :--- | :--- | :--- | :--- |
| **PostgreSQL Database** | Logical snapshot (`pg_dump`) | Every 6 hours | 30 days |
| **Continuous WAL** | Point-in-Time Recovery (PITR) | Continuous | 7 days |
| **Redis Cache / State** | AOF + RDB snapshots (`bgsave`) | Hourly | 48 hours |
| **Vector Index Data** | File snapshot / S3 sync | Daily | 30 days |

## 2. PostgreSQL Backup Procedures

### Automated Logical Backup (`pg_dump`)
```bash
# Create compressed logical database backup
pg_dump -Fc -h "$DB_HOST" -U "$DB_USER" -d "$DB_NAME" -f "backup_$(date +%Y%m%d_%H%M%S).dump"
```

### Point-in-Time Recovery (PITR)
Enable WAL archiving in `postgresql.conf`:
```ini
wal_level = replica
archive_mode = on
archive_command = 'test ! -f /mnt/wal_archive/%f && cp %p /mnt/wal_archive/%f'
```

## 3. Database Restoration Procedure

1. **Provision replacement instance or isolate target database**.
2. **Restore logical dump**:
   ```bash
   pg_restore -v -h "$RESTORE_HOST" -U "$RESTORE_USER" -d "$RESTORE_DB" --clean --no-owner "backup_file.dump"
   ```
3. **Run schema check and migration alignment**:
   ```bash
   alembic current
   alembic upgrade head
   ```
4. **Validate application readiness**:
   ```bash
   curl -f http://localhost:8000/ready
   ```

## 4. Disaster Recovery Drills

Schedule quarterly restore verification drills in a non-production environment:
1. Restore the most recent backup into an isolated staging database.
2. Execute automated integration and isolation tests against the restored data:
   ```bash
   uv run pytest tests/integration/test_auth_isolation_regression.py
   ```
3. Verify zero data corruption and complete user partition isolation.
