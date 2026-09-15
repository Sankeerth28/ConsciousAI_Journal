import React, { useEffect, useState, useMemo } from 'react';
import { NavLink } from 'react-router-dom';
import {
  LineChart as LineChartIcon,
  Smile,
  Compass,
  Tag,
  Sparkles,
  Info,
  PenSquare,
} from 'lucide-react';
import { listJournals } from '../api/journals';
import type { JournalEntryRead } from '../api/types';
import { Card } from '../components/ui/Card';
import { Button } from '../components/ui/Button';
import { Skeleton } from '../components/ui/Skeleton';
import { EmptyState } from '../components/common/EmptyState';
import { MoodChart } from '../components/analytics/MoodChart';
import { EmotionDistributionChart } from '../components/analytics/EmotionDistributionChart';
import { ValueDistributionChart } from '../components/analytics/ValueDistributionChart';

export const InsightsPage: React.FC = () => {
  const [entries, setEntries] = useState<JournalEntryRead[]>([]);
  const [isLoading, setIsLoading] = useState(true);

  useEffect(() => {
    let isMounted = true;
    const fetchAllData = async () => {
      try {
        setIsLoading(true);
        const res = await listJournals({ limit: 100 });
        if (isMounted) {
          setEntries(res.items || []);
        }
      } catch {
        // Handle gracefully
      } finally {
        if (isMounted) {
          setIsLoading(false);
        }
      }
    };

    fetchAllData();
    return () => {
      isMounted = false;
    };
  }, []);

  // Compute tag frequencies
  const tagCounts = useMemo(() => {
    const counts: Record<string, number> = {};
    entries.forEach((e) => {
      e.tags?.forEach((t) => {
        counts[t] = (counts[t] || 0) + 1;
      });
    });
    return Object.entries(counts).sort((a, b) => b[1] - a[1]).slice(0, 8);
  }, [entries]);

  // Generate reflective observations based on real data
  const reflectiveObservations = useMemo(() => {
    const notes: string[] = [];
    if (entries.length === 0) return notes;

    const moodEntries = entries.filter((e) => e.mood_score !== null && e.mood_score !== undefined);
    if (moodEntries.length >= 2) {
      const avg = moodEntries.reduce((acc, curr) => acc + (curr.mood_score || 0), 0) / moodEntries.length;
      if (avg >= 7.5) {
        notes.push('Your recent entries show a predominantly positive, grounded emotional equilibrium.');
      } else if (avg <= 4.5) {
        notes.push('Your recent reflections suggest you may be processing challenging experiences or experiencing lower energy.');
      } else {
        notes.push('A pattern in your journal data indicates balanced, steady day-to-day fluctuations.');
      }
    }

    const emotions = entries.map((e) => e.top_emotion).filter(Boolean);
    if (emotions.length > 0) {
      const top = emotions[0];
      notes.push(`You frequently acknowledge feelings relating to “${top}”. You may want to reflect on what environments foster this state.`);
    }

    const values = entries.map((e) => e.top_value).filter(Boolean);
    if (values.length > 0) {
      const topVal = values[0];
      notes.push(`The core value of “${topVal}” consistently shapes how you interpret daily events and choices.`);
    }

    return notes;
  }, [entries]);

  if (isLoading) {
    return (
      <div className="space-y-6 animate-fadeIn py-4">
        <Skeleton className="h-8 w-48" />
        <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
          <Skeleton className="h-80 w-full" />
          <Skeleton className="h-80 w-full" />
        </div>
      </div>
    );
  }

  if (entries.length === 0) {
    return (
      <div className="py-12">
        <EmptyState
          icon={<LineChartIcon className="h-8 w-8" />}
          title="No Analytics Available Yet"
          description="Insights are computed dynamically from your reflection history. Write your first journal entry to unlock patterns."
          actionLabel="Write Journal Entry"
          onAction={() => window.location.assign('/journal/new')}
        />
      </div>
    );
  }

  return (
    <div className="space-y-8 animate-fadeIn">
      {/* Title */}
      <div className="pb-4 border-b border-slate-800/80 flex flex-col sm:flex-row sm:items-center justify-between gap-4">
        <div>
          <h1 className="text-2xl font-bold text-white tracking-tight">
            Insights & Emotional Trends
          </h1>
          <p className="text-xs text-slate-400 mt-0.5">
            Holistic analysis computed strictly from your personal reflection data
          </p>
        </div>

        <NavLink to="/journal/new">
          <Button size="sm" variant="primary" icon={<PenSquare className="h-3.5 w-3.5" />}>
            New Entry
          </Button>
        </NavLink>
      </div>

      {/* Reflective Observations Banner */}
      {reflectiveObservations.length > 0 && (
        <Card className="p-6 bg-gradient-to-r from-slate-900/90 via-[#0e1628] to-slate-900/90 border border-cyan-500/20">
          <div className="flex items-center gap-2.5 text-xs font-semibold uppercase tracking-wider text-cyan-400 mb-3">
            <Sparkles className="h-4 w-4 text-cyan-400" />
            <span>Mindful Reflections from Your History</span>
          </div>
          <div className="space-y-2 text-sm text-slate-200 leading-relaxed">
            {reflectiveObservations.map((obs, idx) => (
              <p key={idx} className="flex items-start gap-2">
                <span className="text-cyan-400 font-bold">•</span>
                <span>{obs}</span>
              </p>
            ))}
          </div>
          <div className="mt-4 pt-3 border-t border-slate-800/60 flex items-center gap-1.5 text-[11px] text-slate-400">
            <Info className="h-3.5 w-3.5" />
            <span>Non-clinical emotional awareness observations based solely on self-reported entries.</span>
          </div>
        </Card>
      )}

      {/* Charts Grid */}
      <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
        {/* Mood Trajectory */}
        <Card className="p-6">
          <div className="flex items-center justify-between mb-4">
            <div>
              <h3 className="text-sm font-bold text-white flex items-center gap-2">
                <LineChartIcon className="h-4 w-4 text-cyan-400" />
                Mood Trajectory Over Time
              </h3>
              <p className="text-xs text-slate-400 mt-0.5">
                1.0 (challenging) to 10.0 (optimal)
              </p>
            </div>
          </div>
          <MoodChart entries={entries} />
        </Card>

        {/* Emotion Distribution */}
        <Card className="p-6">
          <div className="flex items-center justify-between mb-4">
            <div>
              <h3 className="text-sm font-bold text-white flex items-center gap-2">
                <Smile className="h-4 w-4 text-rose-400" />
                Detected Emotional Spectrum
              </h3>
              <p className="text-xs text-slate-400 mt-0.5">
                Frequency of emotional states identified by ConsciousAI
              </p>
            </div>
          </div>
          <EmotionDistributionChart entries={entries} />
        </Card>

        {/* Value Distribution */}
        <Card className="p-6">
          <div className="flex items-center justify-between mb-4">
            <div>
              <h3 className="text-sm font-bold text-white flex items-center gap-2">
                <Compass className="h-4 w-4 text-violet-400" />
                Underlying Core Values
              </h3>
              <p className="text-xs text-slate-400 mt-0.5">
                Principles and motivations reflected in your writing
              </p>
            </div>
          </div>
          <ValueDistributionChart entries={entries} />
        </Card>

        {/* Top Tags & Habits */}
        <Card className="p-6">
          <div className="flex items-center justify-between mb-4">
            <div>
              <h3 className="text-sm font-bold text-white flex items-center gap-2">
                <Tag className="h-4 w-4 text-emerald-400" />
                Recurring Themes & Tags
              </h3>
              <p className="text-xs text-slate-400 mt-0.5">
                Most frequent thematic topics tagged in your space
              </p>
            </div>
          </div>

          {tagCounts.length === 0 ? (
            <div className="py-12 text-center text-xs text-slate-500 italic">
              No tags used yet. Add tags to your reflections to track specific life themes.
            </div>
          ) : (
            <div className="flex flex-wrap gap-2 pt-2">
              {tagCounts.map(([tag, count]) => (
                <div
                  key={tag}
                  className="flex items-center gap-2 px-3 py-1.5 rounded-xl bg-slate-800/80 border border-slate-700/80 text-xs font-medium text-slate-200"
                >
                  <Tag className="h-3 w-3 text-cyan-400" />
                  <span>{tag}</span>
                  <span className="text-[10px] px-1.5 py-0.2 rounded-full bg-slate-900 text-slate-400">
                    {count}
                  </span>
                </div>
              ))}
            </div>
          )}
        </Card>
      </div>
    </div>
  );
};
