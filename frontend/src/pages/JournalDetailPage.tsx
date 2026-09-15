import React, { useEffect, useState } from 'react';
import { useParams, useNavigate } from 'react-router-dom';
import {
  ArrowLeft,
  Calendar,
  Lock,
  Edit2,
  Trash2,
  RotateCcw,
  Tag,
  AlertTriangle,
} from 'lucide-react';
import {
  getJournal,
  updateJournal,
  deleteJournal,
  restoreJournal,
} from '../api/journals';
import type { JournalEntryRead } from '../api/types';
import { formatApiError } from '../api/client';
import { Button } from '../components/ui/Button';
import { Card } from '../components/ui/Card';
import { Modal } from '../components/ui/Modal';
import { Skeleton } from '../components/ui/Skeleton';
import { MoodSelector } from '../components/journal/MoodSelector';
import { EmotionBadge, ValueBadge } from '../components/journal/EmotionBadge';
import { ReflectionCard } from '../components/journal/ReflectionCard';

export const JournalDetailPage: React.FC = () => {
  const { id } = useParams<{ id: string }>();
  const navigate = useNavigate();

  const [entry, setEntry] = useState<JournalEntryRead | null>(null);
  const [isLoading, setIsLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);

  // Edit modal state
  const [isEditModalOpen, setIsEditModalOpen] = useState(false);
  const [editMood, setEditMood] = useState<number | null>(7.0);
  const [editTags, setEditTags] = useState<string>('');
  const [editIsPrivate, setEditIsPrivate] = useState(false);
  const [isUpdating, setIsUpdating] = useState(false);

  // Delete modal state
  const [isDeleteModalOpen, setIsDeleteModalOpen] = useState(false);
  const [isDeleting, setIsDeleting] = useState(false);
  const [isRestoring, setIsRestoring] = useState(false);

  useEffect(() => {
    let isMounted = true;
    const fetchEntry = async () => {
      if (!id) return;
      try {
        setIsLoading(true);
        const data = await getJournal(parseInt(id, 10));
        if (isMounted) {
          setEntry(data);
          setEditMood(data.mood_score ?? 7.0);
          setEditTags(data.tags?.join(', ') || '');
          setEditIsPrivate(data.is_private);
        }
      } catch (err) {
        if (isMounted) {
          setError(formatApiError(err));
        }
      } finally {
        if (isMounted) {
          setIsLoading(false);
        }
      }
    };

    fetchEntry();
    return () => {
      isMounted = false;
    };
  }, [id]);

  const handleUpdate = async (e: React.FormEvent) => {
    e.preventDefault();
    if (!entry) return;

    try {
      setIsUpdating(true);
      const parsedTags = editTags
        .split(',')
        .map((t) => t.trim())
        .filter(Boolean);

      const updated = await updateJournal(entry.id, {
        mood_score: editMood,
        tags: parsedTags,
        is_private: editIsPrivate,
      });

      setEntry(updated);
      setIsEditModalOpen(false);
    } catch (err) {
      alert(formatApiError(err));
    } finally {
      setIsUpdating(false);
    }
  };

  const handleDelete = async () => {
    if (!entry) return;
    try {
      setIsDeleting(true);
      await deleteJournal(entry.id);
      setIsDeleteModalOpen(false);
      navigate('/journal/history');
    } catch (err) {
      alert(formatApiError(err));
    } finally {
      setIsDeleting(false);
    }
  };

  const handleRestore = async () => {
    if (!entry) return;
    try {
      setIsRestoring(true);
      const restored = await restoreJournal(entry.id);
      setEntry(restored);
    } catch (err) {
      alert(formatApiError(err));
    } finally {
      setIsRestoring(false);
    }
  };

  if (isLoading) {
    return (
      <div className="max-w-3xl mx-auto space-y-6 animate-fadeIn py-6">
        <Skeleton className="h-8 w-40" />
        <Skeleton className="h-64 w-full" />
        <Skeleton className="h-40 w-full" />
      </div>
    );
  }

  if (error || !entry) {
    return (
      <div className="max-w-md mx-auto text-center py-16 space-y-4">
        <div className="mx-auto flex h-12 w-12 items-center justify-center rounded-2xl bg-rose-500/10 text-rose-400">
          <AlertTriangle className="h-6 w-6" />
        </div>
        <h2 className="text-xl font-bold text-white">Entry Not Found</h2>
        <p className="text-sm text-slate-400">
          {error || 'This journal entry does not exist or has been removed.'}
        </p>
        <Button onClick={() => navigate('/journal/history')} variant="outline">
          Back to History
        </Button>
      </div>
    );
  }

  const createdDate = new Date(entry.created_at).toLocaleDateString('en-US', {
    month: 'long',
    day: 'numeric',
    year: 'numeric',
    hour: '2-digit',
    minute: '2-digit',
  });

  return (
    <div className="max-w-3xl mx-auto space-y-6 animate-fadeIn">
      {/* Header and Actions */}
      <div className="flex flex-wrap items-center justify-between gap-4 pb-4 border-b border-slate-800">
        <button
          onClick={() => navigate('/journal/history')}
          className="flex items-center gap-1.5 text-xs font-semibold text-slate-400 hover:text-white transition-colors"
        >
          <ArrowLeft className="h-4 w-4" /> Back to History
        </button>

        <div className="flex items-center gap-2">
          <Button
            size="sm"
            variant="outline"
            onClick={handleRestore}
            isLoading={isRestoring}
            icon={<RotateCcw className="h-3.5 w-3.5" />}
          >
            Restore
          </Button>

          <Button
            size="sm"
            variant="outline"
            onClick={() => setIsEditModalOpen(true)}
            icon={<Edit2 className="h-3.5 w-3.5" />}
          >
            Edit Attributes
          </Button>

          <Button
            size="sm"
            variant="danger"
            onClick={() => setIsDeleteModalOpen(true)}
            icon={<Trash2 className="h-3.5 w-3.5" />}
          >
            Delete
          </Button>
        </div>
      </div>

      {/* Main Journal Content Card */}
      <Card className="p-6 sm:p-8 space-y-6">
        {/* Metadata row */}
        <div className="flex flex-wrap items-center justify-between gap-3 text-xs border-b border-slate-800 pb-4">
          <div className="flex items-center gap-2 text-slate-400">
            <Calendar className="h-4 w-4 text-cyan-400" />
            <span className="font-medium text-slate-300">{createdDate}</span>
            {entry.is_private && (
              <span className="flex items-center gap-1 text-amber-400 font-semibold ml-2">
                <Lock className="h-3.5 w-3.5" /> Private
              </span>
            )}
          </div>

          {entry.mood_score !== null && entry.mood_score !== undefined && (
            <span className="px-3 py-1 rounded-full border border-cyan-500/30 bg-cyan-500/10 text-cyan-400 font-semibold">
              Mood Score: {entry.mood_score.toFixed(1)} / 10
            </span>
          )}
        </div>

        {/* Journal Text */}
        <div className="text-base sm:text-lg text-slate-100 leading-relaxed whitespace-pre-wrap font-sans">
          {entry.text}
        </div>

        {/* Emotions, Core Values, Tags */}
        <div className="pt-4 border-t border-slate-800 flex flex-wrap gap-4 text-xs">
          {entry.detected_emotions?.length > 0 && (
            <div className="flex items-center gap-1.5 flex-wrap">
              <span className="text-slate-400 font-medium">Emotions:</span>
              {entry.detected_emotions.map((emo, idx) => (
                <EmotionBadge key={idx} emotion={emo} />
              ))}
            </div>
          )}

          {entry.detected_values?.length > 0 && (
            <div className="flex items-center gap-1.5 flex-wrap">
              <span className="text-slate-400 font-medium">Values:</span>
              {entry.detected_values.map((val, idx) => (
                <ValueBadge key={idx} value={val} />
              ))}
            </div>
          )}

          {entry.tags?.length > 0 && (
            <div className="flex items-center gap-1.5 flex-wrap">
              <span className="text-slate-400 font-medium">Tags:</span>
              {entry.tags.map((tag, idx) => (
                <span
                  key={idx}
                  className="flex items-center gap-1 px-2.5 py-0.5 rounded-md bg-slate-800 text-slate-300"
                >
                  <Tag className="h-3 w-3 text-slate-500" />
                  {tag}
                </span>
              ))}
            </div>
          )}
        </div>
      </Card>

      {/* AI Reflection section if available */}
      {entry.ai_response && (
        <ReflectionCard
          reflection={{
            response: entry.ai_response,
            persona: 'Supportive',
            model_name: 'FastAPI Reflection Engine',
            safety_flag: false,
            fallback_used: false,
          }}
          entryId={entry.id}
          detectedEmotions={entry.detected_emotions}
          detectedValues={entry.detected_values}
        />
      )}

      {/* Edit Attributes Modal */}
      <Modal
        isOpen={isEditModalOpen}
        onClose={() => setIsEditModalOpen(false)}
        title="Update Entry Attributes"
        description="Update mood score, tags, and privacy settings."
      >
        <form onSubmit={handleUpdate} className="space-y-4">
          <MoodSelector
            value={editMood}
            onChange={(val) => setEditMood(val)}
            disabled={isUpdating}
          />

          <div>
            <label className="block text-xs font-semibold uppercase tracking-wider text-slate-300 mb-1.5">
              Tags (comma-separated)
            </label>
            <input
              type="text"
              value={editTags}
              onChange={(e) => setEditTags(e.target.value)}
              className="w-full rounded-xl bg-slate-800 border border-slate-700 p-2.5 text-sm text-slate-100 focus:outline-none focus:ring-2 focus:ring-cyan-500"
              placeholder="growth, focus, mindfulness"
            />
          </div>

          <div className="flex items-center justify-between p-3 rounded-xl bg-slate-800/60 border border-slate-700">
            <span className="text-xs text-slate-200">Mark as Private</span>
            <input
              type="checkbox"
              checked={editIsPrivate}
              onChange={(e) => setEditIsPrivate(e.target.checked)}
              className="h-4 w-4 rounded bg-slate-800 border-slate-700 text-cyan-500 focus:ring-cyan-400"
            />
          </div>

          <div className="flex justify-end gap-2 pt-3 border-t border-slate-800">
            <Button
              type="button"
              variant="ghost"
              onClick={() => setIsEditModalOpen(false)}
            >
              Cancel
            </Button>
            <Button type="submit" variant="primary" isLoading={isUpdating}>
              Save Changes
            </Button>
          </div>
        </form>
      </Modal>

      {/* Delete Confirmation Modal */}
      <Modal
        isOpen={isDeleteModalOpen}
        onClose={() => setIsDeleteModalOpen(false)}
        title="Delete Journal Entry"
        description="This will soft-delete your entry from active lists."
      >
        <div className="space-y-4">
          <p className="text-sm text-slate-300 leading-relaxed">
            Are you sure you want to remove this journal entry? You can restore it later if needed.
          </p>
          <div className="flex justify-end gap-2 pt-3 border-t border-slate-800">
            <Button
              variant="ghost"
              onClick={() => setIsDeleteModalOpen(false)}
              disabled={isDeleting}
            >
              Keep Entry
            </Button>
            <Button
              variant="danger"
              onClick={handleDelete}
              isLoading={isDeleting}
              icon={<Trash2 className="h-4 w-4" />}
            >
              Confirm Delete
            </Button>
          </div>
        </div>
      </Modal>
    </div>
  );
};
