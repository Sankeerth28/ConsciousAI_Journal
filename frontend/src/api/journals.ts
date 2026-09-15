import { apiClient } from './client';
import type {
  FeedbackCreate,
  FeedbackRead,
  JournalEntryCreate,
  JournalEntryCreateResponse,
  JournalEntryRead,
  JournalEntryUpdate,
  JournalExportFormat,
  JournalListResponse,
} from './types';

export interface ListJournalParams {
  skip?: number;
  limit?: number;
  tag?: string;
  persona?: string;
  is_private?: boolean;
  include_deleted?: boolean;
}

/**
 * Fetch paginated list of journal entries.
 * Endpoint: GET /api/v1/journals
 */
export async function listJournals(params?: ListJournalParams): Promise<JournalListResponse> {
  const response = await apiClient.get<JournalListResponse>('/api/v1/journals', {
    params,
  });
  return response.data;
}

/**
 * Create a new journal entry and receive AI reflection & safety analysis.
 * Endpoint: POST /api/v1/journals
 */
export async function createJournal(
  payload: JournalEntryCreate
): Promise<JournalEntryCreateResponse> {
  const response = await apiClient.post<JournalEntryCreateResponse>(
    '/api/v1/journals',
    payload
  );
  return response.data;
}

/**
 * Fetch a single journal entry by ID.
 * Endpoint: GET /api/v1/journals/{entry_id}
 */
export async function getJournal(entryId: number): Promise<JournalEntryRead> {
  const response = await apiClient.get<JournalEntryRead>(`/api/v1/journals/${entryId}`);
  return response.data;
}

/**
 * Update mutable fields of an existing journal entry (tags, mood_score, is_private).
 * Endpoint: PATCH /api/v1/journals/{entry_id}
 */
export async function updateJournal(
  entryId: number,
  payload: JournalEntryUpdate
): Promise<JournalEntryRead> {
  const response = await apiClient.patch<JournalEntryRead>(
    `/api/v1/journals/${entryId}`,
    payload
  );
  return response.data;
}

/**
 * Soft delete a journal entry.
 * Endpoint: DELETE /api/v1/journals/{entry_id}
 */
export async function deleteJournal(entryId: number): Promise<void> {
  await apiClient.delete(`/api/v1/journals/${entryId}`);
}

/**
 * Restore a soft-deleted journal entry.
 * Endpoint: POST /api/v1/journals/{entry_id}/restore
 */
export async function restoreJournal(entryId: number): Promise<JournalEntryRead> {
  const response = await apiClient.post<JournalEntryRead>(
    `/api/v1/journals/${entryId}/restore`
  );
  return response.data;
}

/**
 * Export journal entries in CSV or JSON format.
 * Endpoint: GET /api/v1/journals/export
 */
export async function exportJournals(format: JournalExportFormat): Promise<Blob> {
  const response = await apiClient.get('/api/v1/journals/export', {
    params: { format },
    responseType: 'blob',
  });
  return response.data;
}

/**
 * Submit feedback for a journal entry's AI response.
 * Endpoint: POST /api/v1/journals/{entry_id}/feedback
 */
export async function submitJournalFeedback(
  entryId: number,
  payload: FeedbackCreate
): Promise<FeedbackRead> {
  const response = await apiClient.post<FeedbackRead>(
    `/api/v1/journals/${entryId}/feedback`,
    payload
  );
  return response.data;
}
