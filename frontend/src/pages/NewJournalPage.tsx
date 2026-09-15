import React, { useState } from 'react';
import { NavLink, useNavigate } from 'react-router-dom';
import {
  Send,
  Lock,
  Tag,
  ArrowLeft,
  RotateCcw,
  Eye,
  AlertCircle,
} from 'lucide-react';
import { createJournal } from '../api/journals';
import type {
  JournalEntryCreateResponse,
  PersonaEnum,
  RegionEnum,
} from '../api/types';
import { formatApiError } from '../api/client';
import { Button } from '../components/ui/Button';
import { Card } from '../components/ui/Card';
import { Textarea } from '../components/ui/Textarea';
import { Select } from '../components/ui/Select';
import { MoodSelector } from '../components/journal/MoodSelector';
import { PersonaSelector } from '../components/journal/PersonaSelector';
import { ReflectionCard } from '../components/journal/ReflectionCard';
import { SafetyNotice } from '../components/journal/SafetyNotice';

export const NewJournalPage: React.FC = () => {
  const navigate = useNavigate();

  // Form inputs
  const [text, setText] = useState('');
  const [moodScore, setMoodScore] = useState<number | null>(7.0);
  const [tagInput, setTagInput] = useState('');
  const [tags, setTags] = useState<string[]>([]);
  const [persona, setPersona] = useState<PersonaEnum>('Supportive');
  const [region, setRegion] = useState<RegionEnum>('GLOBAL');
  const [isPrivate, setIsPrivate] = useState(false);

  // Submission state
  const [isSubmitting, setIsSubmitting] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [responseResult, setResponseResult] = useState<JournalEntryCreateResponse | null>(null);

  // Tag addition helper
  const handleAddTag = (e: React.KeyboardEvent) => {
    if (e.key === 'Enter' || e.key === ',') {
      e.preventDefault();
      const trimmed = tagInput.trim().replace(/^,|,$/g, '');
      if (trimmed && !tags.includes(trimmed)) {
        setTags([...tags, trimmed]);
        setTagInput('');
      }
    }
  };

  const removeTag = (toRemove: string) => {
    setTags(tags.filter((t) => t !== toRemove));
  };

  const handleSubmit = async (e: React.FormEvent) => {
    e.preventDefault();
    setError(null);

    if (!text.trim()) {
      setError('Please write your reflection before saving.');
      return;
    }

    try {
      setIsSubmitting(true);
      const res = await createJournal({
        text: text.trim(),
        mood_score: moodScore,
        tags: tags.length > 0 ? tags : undefined,
        persona,
        region,
        is_private: isPrivate,
      });
      setResponseResult(res);
    } catch (err) {
      setError(formatApiError(err));
    } finally {
      setIsSubmitting(false);
    }
  };

  const resetForm = () => {
    setText('');
    setMoodScore(7.0);
    setTags([]);
    setTagInput('');
    setPersona('Supportive');
    setResponseResult(null);
    setError(null);
  };

  // If entry created successfully, render companion reflection view
  if (responseResult) {
    const isSafe = responseResult.input_safety?.is_safe ?? true;

    return (
      <div className="max-w-3xl mx-auto space-y-6 animate-fadeIn py-4">
        <div className="flex items-center justify-between">
          <button
            onClick={() => navigate('/dashboard')}
            className="inline-flex items-center gap-2 text-xs font-semibold text-slate-400 hover:text-slate-200 transition-colors"
          >
            <ArrowLeft className="h-4 w-4" /> Back to Dashboard
          </button>
          <span className="text-xs font-semibold px-2.5 py-1 rounded-full bg-emerald-500/10 text-emerald-400 border border-emerald-500/20">
            Saved to Staging Database
          </span>
        </div>

        {/* Safety notice if flagged */}
        {!isSafe && <SafetyNotice safety={responseResult.input_safety} />}

        {/* Reflection Card */}
        {responseResult.reflection && (
          <ReflectionCard
            reflection={responseResult.reflection}
            entryId={responseResult.entry?.id}
            detectedEmotions={responseResult.entry?.detected_emotions}
            detectedValues={responseResult.entry?.detected_values}
          />
        )}

        {/* Saved entry excerpt */}
        {responseResult.entry && (
          <Card className="p-6">
            <h4 className="text-xs font-bold uppercase tracking-wider text-slate-400 mb-2">
              Your Entry
            </h4>
            <p className="text-sm text-slate-200 whitespace-pre-wrap leading-relaxed">
              {responseResult.entry.text}
            </p>
          </Card>
        )}

        {/* Post-submission Action Bar */}
        <div className="flex flex-wrap items-center justify-end gap-3 pt-4 border-t border-slate-800">
          <Button
            variant="outline"
            onClick={resetForm}
            icon={<RotateCcw className="h-4 w-4" />}
          >
            Write Another Entry
          </Button>

          {responseResult.entry && (
            <NavLink to={`/journal/${responseResult.entry.id}`}>
              <Button variant="primary" icon={<Eye className="h-4 w-4" />}>
                View Entry Details
              </Button>
            </NavLink>
          )}

          <Button variant="ghost" onClick={() => navigate('/dashboard')}>
            Return to Dashboard
          </Button>
        </div>
      </div>
    );
  }

  return (
    <div className="max-w-3xl mx-auto space-y-6 animate-fadeIn">
      {/* Page Title & Back */}
      <div className="flex items-center justify-between pb-2 border-b border-slate-800/80">
        <div>
          <h1 className="text-2xl font-bold text-white tracking-tight">
            New Reflection
          </h1>
          <p className="text-xs text-slate-400 mt-0.5">
            Express freely. ConsciousAI will analyze themes and emotions.
          </p>
        </div>

        <button
          type="button"
          onClick={() => navigate('/dashboard')}
          className="text-xs font-semibold text-slate-400 hover:text-slate-200 transition-colors"
        >
          Cancel
        </button>
      </div>

      {error && (
        <div className="p-3.5 rounded-xl bg-rose-500/10 border border-rose-500/20 text-xs text-rose-300 flex items-start gap-2.5 animate-fadeIn">
          <AlertCircle className="h-4 w-4 shrink-0 mt-0.5 text-rose-400" />
          <span>{error}</span>
        </div>
      )}

      <form onSubmit={handleSubmit} className="space-y-6">
        {/* Main Writing Area */}
        <Textarea
          placeholder="What is on your mind today? Write candidly about your thoughts, struggles, or moments of gratitude..."
          rows={10}
          maxLength={10000}
          showCount
          currentLength={text.length}
          value={text}
          onChange={(e) => setText(e.target.value)}
          className="text-base leading-relaxed p-4 min-h-[220px]"
          required
        />

        {/* Mood Selector Slider */}
        <MoodSelector
          value={moodScore}
          onChange={(val) => setMoodScore(val)}
          disabled={isSubmitting}
        />

        {/* Persona Selector Radio Cards */}
        <PersonaSelector
          value={persona}
          onChange={(p) => setPersona(p)}
          disabled={isSubmitting}
        />

        {/* Tags, Region & Privacy Settings */}
        <div className="grid grid-cols-1 sm:grid-cols-2 gap-4">
          {/* Tags Input */}
          <div className="space-y-1.5">
            <label className="block text-xs font-semibold uppercase tracking-wider text-slate-300">
              Tags (Press Enter or comma)
            </label>
            <div className="relative">
              <input
                type="text"
                placeholder="e.g. clarity, career, anxiety..."
                value={tagInput}
                onChange={(e) => setTagInput(e.target.value)}
                onKeyDown={handleAddTag}
                disabled={isSubmitting}
                className="w-full rounded-xl bg-slate-900/60 border border-slate-800 text-sm text-slate-100 placeholder:text-slate-500 px-3.5 py-2.5 focus:outline-none focus:ring-2 focus:ring-cyan-500/30 focus:border-cyan-500"
              />
            </div>

            {tags.length > 0 && (
              <div className="flex flex-wrap gap-1.5 pt-1.5">
                {tags.map((t) => (
                  <span
                    key={t}
                    className="inline-flex items-center gap-1 text-xs px-2.5 py-0.5 rounded-md bg-slate-800 text-slate-200 border border-slate-700"
                  >
                    <Tag className="h-2.5 w-2.5 text-cyan-400" />
                    {t}
                    <button
                      type="button"
                      onClick={() => removeTag(t)}
                      className="text-slate-400 hover:text-rose-400 ml-1 font-bold"
                    >
                      ×
                    </button>
                  </span>
                ))}
              </div>
            )}
          </div>

          {/* Region selector */}
          <div>
            <Select
              label="Crisis Helpline Region"
              value={region}
              onChange={(e) => setRegion(e.target.value as RegionEnum)}
              options={[
                { label: 'Global / International', value: 'GLOBAL' },
                { label: 'United States (US)', value: 'US' },
                { label: 'Canada (CA)', value: 'CA' },
                { label: 'United Kingdom (GB)', value: 'GB' },
                { label: 'Australia (AU)', value: 'AU' },
                { label: 'India (IN)', value: 'IN' },
              ]}
            />
          </div>
        </div>

        {/* Private Toggle */}
        <div className="flex items-center justify-between p-4 rounded-xl bg-slate-900/40 border border-slate-800">
          <div className="flex items-center gap-3">
            <Lock className={`h-4 w-4 ${isPrivate ? 'text-amber-400' : 'text-slate-500'}`} />
            <div>
              <span className="block text-xs font-semibold text-slate-200">
                Private Journal Entry
              </span>
              <span className="block text-[11px] text-slate-400">
                Mark as personal to filter or hide from public dashboards.
              </span>
            </div>
          </div>
          <input
            type="checkbox"
            checked={isPrivate}
            onChange={(e) => setIsPrivate(e.target.checked)}
            className="h-4 w-4 rounded bg-slate-800 border-slate-700 text-cyan-500 focus:ring-cyan-400 cursor-pointer"
          />
        </div>

        {/* Submit Actions */}
        <div className="flex items-center justify-end gap-3 pt-4 border-t border-slate-800">
          <Button
            type="button"
            variant="ghost"
            onClick={() => navigate('/dashboard')}
            disabled={isSubmitting}
          >
            Discard
          </Button>

          <Button
            type="submit"
            variant="primary"
            size="lg"
            isLoading={isSubmitting}
            icon={<Send className="h-4 w-4" />}
          >
            Reflect & Save Entry
          </Button>
        </div>
      </form>
    </div>
  );
};
