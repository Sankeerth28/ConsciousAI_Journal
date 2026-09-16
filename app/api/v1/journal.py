"""Journal REST API endpoints for ConsciousAI Journal V2."""

from __future__ import annotations

import contextlib
import csv
import io
import json
from datetime import datetime
from typing import TYPE_CHECKING, Any

from fastapi import APIRouter, Depends, HTTPException, Query, Response, status
from sqlmodel import Session

from app.api.deps import (
    get_current_user_id,
    get_db,
    get_feedback_repo,
    get_journal_repo,
    get_pipeline,
)
from app.models.feedback import Feedback
from app.models.journal import JournalEntry
from app.repositories.feedback import FeedbackRepository
from app.repositories.journal import JournalRepository
from app.schemas.feedback import FeedbackCreate, FeedbackRead
from app.schemas.journal import (
    JournalEntryCreate,
    JournalEntryCreateResponse,
    JournalEntryRead,
    JournalEntryUpdate,
    JournalExportFormat,
    JournalListResponse,
)

if TYPE_CHECKING:
    from app.ai.pipeline import JournalReflectionPipeline

router = APIRouter(tags=["journals"])


@router.post(
    "",
    response_model=JournalEntryCreateResponse,
    status_code=status.HTTP_201_CREATED,
    summary="Create a new journal entry with AI reflection and safety checks",
)
def create_journal_entry(
    payload: JournalEntryCreate,
    response: Response,
    user_id: str = Depends(get_current_user_id),
    repo: JournalRepository = Depends(get_journal_repo),
    pipeline: JournalReflectionPipeline = Depends(get_pipeline),
    session: Session = Depends(get_db),
) -> JournalEntryCreateResponse:
    """Create a new journal entry adhering to strict safety-first order.

    Safety Boundary:
    - If input contains crisis ideation or self-harm, pipeline halts immediately.
    - Zero persistence occurs; entry is NOT saved to the database.
    - Safe crisis envelope and regional helplines are returned immediately.
    - If safe, entry is persisted along with AI reflection and classifications.
    """
    # 1. Execute reflection pipeline with database context and user isolation
    pipeline_result = pipeline.process_journal_entry(
        text=payload.text,
        persona=payload.persona,
        region=payload.region,
        session=session,
        user_id=user_id,
    )

    # 2. Safety Interception: Refuse persistence if input is unsafe
    if not pipeline_result.input_safety.is_safe:
        # Do not persist anything to database
        response.status_code = status.HTTP_200_OK
        return JournalEntryCreateResponse(
            entry=None,
            reflection=pipeline_result.reflection,
            input_safety=pipeline_result.input_safety,
            output_safety=pipeline_result.output_safety,
        )

    # 3. Input is safe: persist journal entry to database
    entry = JournalEntry(
        user_id=user_id,
        text=payload.text,
        mood_score=payload.mood_score,
        top_emotion=pipeline_result.emotion.top_emotion if pipeline_result.emotion else None,
        top_value=pipeline_result.value.top_value if pipeline_result.value else None,
        detected_emotions=(
            [e.label for e in pipeline_result.emotion.emotions if e.score >= 0.15]
            or (
                [pipeline_result.emotion.top_emotion] if pipeline_result.emotion.top_emotion else []
            )
            if pipeline_result.emotion and pipeline_result.emotion.emotions
            else []
        ),
        detected_values=(
            [v.label for v in pipeline_result.value.values if v.score >= 0.15]
            or ([pipeline_result.value.top_value] if pipeline_result.value.top_value else [])
            if pipeline_result.value and pipeline_result.value.values
            else []
        ),
        tags=payload.tags,
        is_private=payload.is_private,
        ai_response=pipeline_result.reflection.response,
    )

    try:
        created_entry = repo.create(entry)
    except Exception as exc:
        session.rollback()
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to persist journal entry.",
        ) from exc

    return JournalEntryCreateResponse(
        entry=JournalEntryRead.model_validate(created_entry),
        reflection=pipeline_result.reflection,
        input_safety=pipeline_result.input_safety,
        output_safety=pipeline_result.output_safety,
    )


@router.get(
    "",
    response_model=JournalListResponse,
    summary="List user's journal entries with filtering and deterministic pagination",
)
def list_journal_entries(
    skip: int = Query(default=0, ge=0, description="Offset number of records"),
    limit: int = Query(default=50, ge=1, le=100, description="Page size limit"),
    emotion: str | None = Query(default=None, description="Filter by top emotion"),
    value_theme: str | None = Query(default=None, description="Filter by top value theme"),
    tag: str | None = Query(default=None, description="Filter by tag"),
    is_private: bool | None = Query(default=None, description="Filter by privacy visibility"),
    start_date: datetime | None = Query(default=None, description="Inclusive start datetime"),
    end_date: datetime | None = Query(default=None, description="Inclusive end datetime"),
    search: str | None = Query(default=None, description="Substring search query"),
    user_id: str = Depends(get_current_user_id),
    repo: JournalRepository = Depends(get_journal_repo),
) -> JournalListResponse:
    """Retrieve paginated journal entries for the current user."""
    items = repo.list(
        skip=skip,
        limit=limit,
        emotion=emotion,
        value_theme=value_theme,
        tag=tag,
        is_private=is_private,
        include_deleted=False,
        start_date=start_date,
        end_date=end_date,
        search=search,
        user_id=user_id,
    )
    total = repo.count(
        emotion=emotion,
        value_theme=value_theme,
        tag=tag,
        is_private=is_private,
        include_deleted=False,
        start_date=start_date,
        end_date=end_date,
        search=search,
        user_id=user_id,
    )

    has_more = (skip + len(items)) < total

    return JournalListResponse(
        items=[JournalEntryRead.model_validate(item) for item in items],
        total=total,
        skip=skip,
        limit=limit,
        has_more=has_more,
    )


@router.get(
    "/export",
    summary="Export user's journal entries in CSV or JSON format",
)
def export_journal_entries(
    format: JournalExportFormat = Query(
        default=JournalExportFormat.JSON,
        description="Target export format (csv or json).",
    ),
    start_date: datetime | None = Query(default=None, description="Inclusive start datetime"),
    end_date: datetime | None = Query(default=None, description="Inclusive end datetime"),
    tag: str | None = Query(default=None, description="Filter by tag"),
    user_id: str = Depends(get_current_user_id),
    repo: JournalRepository = Depends(get_journal_repo),
) -> Response:
    """Export the authenticated user's non-deleted journal entries."""
    entries = repo.list(
        skip=0,
        limit=100_000,
        tag=tag,
        include_deleted=False,
        start_date=start_date,
        end_date=end_date,
        user_id=user_id,
    )

    if format == JournalExportFormat.CSV:
        output = io.StringIO()
        writer = csv.writer(output)
        writer.writerow(
            [
                "id",
                "created_at",
                "updated_at",
                "text",
                "mood_score",
                "top_emotion",
                "top_value",
                "tags",
                "ai_response",
                "is_private",
            ]
        )
        for e in entries:
            writer.writerow(
                [
                    e.id,
                    e.created_at.isoformat() if e.created_at else "",
                    e.updated_at.isoformat() if e.updated_at else "",
                    e.text,
                    e.mood_score if e.mood_score is not None else "",
                    e.top_emotion or "",
                    e.top_value or "",
                    ",".join(e.tags) if e.tags else "",
                    e.ai_response or "",
                    e.is_private,
                ]
            )
        csv_data = output.getvalue()
        return Response(
            content=csv_data,
            media_type="text/csv",
            headers={"Content-Disposition": 'attachment; filename="journals.csv"'},
        )

    # JSON export
    serialized = [JournalEntryRead.model_validate(e).model_dump(mode="json") for e in entries]
    json_data = json.dumps(serialized, indent=2)
    return Response(
        content=json_data,
        media_type="application/json",
        headers={"Content-Disposition": 'attachment; filename="journals.json"'},
    )


@router.get(
    "/{entry_id}",
    response_model=JournalEntryRead,
    summary="Get a single journal entry by ID",
)
def get_journal_entry(
    entry_id: int,
    user_id: str = Depends(get_current_user_id),
    repo: JournalRepository = Depends(get_journal_repo),
) -> JournalEntryRead:
    """Fetch a single journal entry owned by the current user."""
    entry = repo.get_by_id(entry_id, include_deleted=False, user_id=user_id)
    if not entry:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Journal entry not found",
        )
    return JournalEntryRead.model_validate(entry)


@router.patch(
    "/{entry_id}",
    response_model=JournalEntryRead,
    summary="Update mutable fields of a journal entry",
)
def update_journal_entry(
    entry_id: int,
    payload: JournalEntryUpdate,
    user_id: str = Depends(get_current_user_id),
    repo: JournalRepository = Depends(get_journal_repo),
) -> JournalEntryRead:
    """Update mutable attributes of an existing entry owned by current user."""
    entry = repo.get_by_id(entry_id, include_deleted=False, user_id=user_id)
    if not entry:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Journal entry not found",
        )

    updates = payload.model_dump(exclude_unset=True)
    if not updates:
        return JournalEntryRead.model_validate(entry)

    try:
        updated = repo.update(entry_id, user_id=user_id, **updates)
        if not updated:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="Journal entry not found",
            )
        return JournalEntryRead.model_validate(updated)
    except ValueError as exc:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail=str(exc),
        ) from exc


@router.delete(
    "/{entry_id}",
    summary="Delete a journal entry (soft delete by default, permanent optional)",
)
def delete_journal_entry(
    entry_id: int,
    permanent: bool = Query(default=False, description="Whether to permanently remove record"),
    user_id: str = Depends(get_current_user_id),
    repo: JournalRepository = Depends(get_journal_repo),
) -> dict[str, Any]:
    """Delete a journal entry owned by the current user."""
    entry = repo.get_by_id(entry_id, include_deleted=True, user_id=user_id)
    if not entry or (not permanent and entry.is_deleted):
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Journal entry not found",
        )

    if permanent:
        success = repo.hard_delete(entry_id, user_id=user_id)
    else:
        success = repo.soft_delete(entry_id, user_id=user_id)

    if not success:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Journal entry not found",
        )

    return {
        "message": "Journal entry permanently deleted"
        if permanent
        else "Journal entry soft-deleted",
        "id": entry_id,
        "permanent": permanent,
    }


@router.post(
    "/{entry_id}/restore",
    response_model=JournalEntryRead,
    summary="Restore a soft-deleted journal entry",
)
def restore_journal_entry(
    entry_id: int,
    user_id: str = Depends(get_current_user_id),
    repo: JournalRepository = Depends(get_journal_repo),
) -> JournalEntryRead:
    """Restore a soft-deleted journal entry owned by the current user."""
    entry = repo.get_by_id(entry_id, include_deleted=True, user_id=user_id)
    if not entry or not entry.is_deleted:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Soft-deleted journal entry not found",
        )

    restored = repo.restore(entry_id, user_id=user_id)
    if not restored:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Journal entry could not be restored",
        )
    return JournalEntryRead.model_validate(restored)


@router.post(
    "/{entry_id}/feedback",
    response_model=FeedbackRead,
    status_code=status.HTTP_201_CREATED,
    summary="Submit user feedback on an AI reflection",
)
def submit_feedback(
    entry_id: int,
    payload: FeedbackCreate,
    user_id: str = Depends(get_current_user_id),
    journal_repo: JournalRepository = Depends(get_journal_repo),
    feedback_repo: FeedbackRepository = Depends(get_feedback_repo),
) -> FeedbackRead:
    """Submit user feedback for an existing, non-deleted entry owned by the user."""
    entry = journal_repo.get_by_id(entry_id, include_deleted=False, user_id=user_id)
    if not entry:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Journal entry not found",
        )

    feedback = Feedback(
        journal_entry_id=entry_id,
        feedback_type=payload.feedback_type,
        comment=payload.comment,
    )
    created = feedback_repo.create(feedback)

    # Legacy compatibility update
    with contextlib.suppress(Exception):
        journal_repo.update(entry_id, user_id=user_id, feedback=payload.feedback_type)

    return FeedbackRead.model_validate(created)


@router.get(
    "/{entry_id}/feedback",
    response_model=list[FeedbackRead],
    summary="List feedback history for a journal entry",
)
def list_entry_feedback(
    entry_id: int,
    user_id: str = Depends(get_current_user_id),
    journal_repo: JournalRepository = Depends(get_journal_repo),
    feedback_repo: FeedbackRepository = Depends(get_feedback_repo),
) -> list[FeedbackRead]:
    """Retrieve all feedback records for an entry owned by the current user."""
    entry = journal_repo.get_by_id(entry_id, include_deleted=False, user_id=user_id)
    if not entry:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Journal entry not found",
        )

    feedbacks = feedback_repo.list_by_entry(entry_id)
    return [FeedbackRead.model_validate(f) for f in feedbacks]
