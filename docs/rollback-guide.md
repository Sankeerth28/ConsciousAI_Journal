# Deployment Rollback Guide — ConsciousAI Journal V2

This guide establishes the rollback protocols and procedures in the event of an unstable deployment.

## 1. Rollback Trigger Criteria

Initiate immediate rollback if any of the following occur within 15 minutes post-deployment:
1. `/ready` probe returns HTTP 503 on new replicas.
2. 5xx error rate spikes above 1% of total traffic.
3. Authentication or permission regressions are detected.
4. Latency p95 degrades by more than 100% relative to baseline.
5. Critical uncaught exceptions appear repeatedly in application logs.

## 2. Application Container Rollback

### Docker Compose (Staging)
```bash
# Pull and switch to previous known-stable image tag
IMAGE_TAG=previous-stable docker compose -f docker-compose.staging.yml up -d web
```

### Kubernetes / ECS / Cloud Run
```bash
# Rollback deployment to previous revision
kubectl rollout undo deployment/consciousai-web
```

## 3. Database Migration Downgrade

> [!CAUTION]
> Downgrading database migrations can cause data loss if irreversible schema modifications were applied. Verify whether the previous application version can safely operate against the current schema (expand-and-contract pattern) before executing a downgrade.

If a database schema rollback is required:
1. Check current migration revision:
   ```bash
   alembic current
   ```
2. Revert one revision backward:
   ```bash
   alembic downgrade -1
   ```
3. Or downgrade to a specific safe revision:
   ```bash
   alembic downgrade <target_revision_id>
   ```

## 4. Post-Rollback Validation

1. **Verify service readiness**:
   ```bash
   curl -i http://localhost:8000/ready
   ```
2. **Execute regression test suite**:
   ```bash
   uv run pytest tests/integration/test_health.py tests/integration/test_auth_isolation_regression.py
   ```
3. **Notify stakeholders** and commence root-cause analysis (RCA).
