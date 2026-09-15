"""Unit tests validating deployment artifacts, Dockerfile security, and CI configurations."""

from __future__ import annotations

from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parent.parent.parent


class TestDockerfileSecurity:
    """Verify Dockerfile security standards and non-root execution."""

    def test_dockerfile_exists(self):
        dockerfile = ROOT_DIR / "Dockerfile"
        assert dockerfile.is_file()

    def test_dockerfile_runs_as_non_root_uid_10001(self):
        content = (ROOT_DIR / "Dockerfile").read_text(encoding="utf-8")
        # Must create user 10001
        assert "10001" in content
        assert "appuser" in content
        # Must switch to non-root USER
        assert "USER 10001:10001" in content or "USER appuser" in content
        # Must not have USER root as the final user
        assert content.strip().splitlines()[-1] != "USER root"

    def test_dockerfile_defines_healthcheck(self):
        content = (ROOT_DIR / "Dockerfile").read_text(encoding="utf-8")
        assert "HEALTHCHECK" in content
        assert "/health" in content

    def test_dockerfile_has_no_hardcoded_secrets(self):
        content = (ROOT_DIR / "Dockerfile").read_text(encoding="utf-8")
        assert "JWT_SECRET_KEY=" not in content
        assert "POSTGRES_PASSWORD=" not in content
        assert "SECRET" not in content.upper() or "SECRET_KEY" not in content


class TestDockerComposeStaging:
    """Verify docker-compose.staging.yml architecture and security."""

    def test_compose_file_exists(self):
        compose_file = ROOT_DIR / "docker-compose.staging.yml"
        assert compose_file.is_file()

    def test_compose_defines_required_services(self):
        content = (ROOT_DIR / "docker-compose.staging.yml").read_text(encoding="utf-8")
        assert "web:" in content
        assert "db:" in content
        assert "redis:" in content

    def test_compose_defines_healthchecks(self):
        content = (ROOT_DIR / "docker-compose.staging.yml").read_text(encoding="utf-8")
        assert "healthcheck:" in content
        assert "/ready" in content

    def test_compose_uses_variable_interpolation_for_secrets(self):
        content = (ROOT_DIR / "docker-compose.staging.yml").read_text(encoding="utf-8")
        # Secrets should be passed via env variables, not hardcoded passwords
        assert "${POSTGRES_PASSWORD" in content
        assert "${JWT_SECRET_KEY" in content


class TestDockerIgnore:
    """Verify sensitive files are excluded from Docker build context."""

    def test_dockerignore_excludes_sensitive_files(self):
        dockerignore = ROOT_DIR / ".dockerignore"
        assert dockerignore.is_file()
        content = dockerignore.read_text(encoding="utf-8")
        assert ".env" in content
        assert "*.pem" in content or "secrets" in content
        assert ".git" in content
