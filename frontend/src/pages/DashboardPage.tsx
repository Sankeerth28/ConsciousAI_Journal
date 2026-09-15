import React, { useEffect, useState, useMemo } from 'react';
import { NavLink } from 'react-router-dom';
import {
  PenSquare,
  Sparkles,
  BookOpen,
  Calendar,
  Smile,
  Compass,
  ArrowRight,
  TrendingUp,
} from 'lucide-react';
import { useAuth } from '../context/AuthContext';
import { listJournals } from '../api/journals';
import type { JournalEntryRead } from '../api/types';
import { Button } from '../components/ui/Button';
import { Card } from '../components/ui/Card';
import { Skeleton } from '../components/ui/Skeleton';
import { EmptyState } from '../components/common/EmptyState';
import { JournalCard } from '../components/journal/JournalCard';
import { MoodChart } from '../components/analytics/MoodChart';

export const DashboardPage: React.FC = () => {
  const { user } = useAuth();
  const [entries, setEntries] = useState<JournalEntryRead[]>([]);
  const [isLoading, setIsLoading] = useState(true);

  useEffect(() => {
    let isMounted = true;
    const fetchDashboardData = async () => {
      try {
        setIsLoading(true);
        const data = await listJournals({ limit: 20 });
        if (isMounted) {
          setEntries(data.items || []);
        }
      } catch {
        // Handled gracefully via empty state
      } finally {
        if (isMounted) {
          setIsLoading(false);
        }
      }
    };

    fetchDashboardData();
    return () => {
      isMounted = false;
    };
  }, []);

  // Compute summary stats from real entries
  const stats = useMemo(() => {
    const totalEntries = entries.length;
    const moodEntries = entries.filter((e) => e.mood_score !== null && e.mood_score !== undefined);
    const avgMood =
      moodEntries.length > 0
        ? moodEntries.reduce((acc, curr) => acc + (curr.mood_score || 0), 0) / moodEntries.length
        : null;

    // Most common emotion
    const emotionCounts: Record<string, number> = {};
    entries.forEach((e) => {
      if (e.top_emotion) emotionCounts[e.top_emotion] = (emotionCounts[e.top_emotion] || 0) + 1;
    });
    const topEmotion = Object.entries(emotionCounts).sort((a, b) => b[1] - a[1])[0]?.[0] || null;

    // Most common value
    const valueCounts: Record<string, number> = {};
    entries.forEach((e) => {
      if (e.top_value) valueCounts[e.top_value] = (valueCounts[e.top_value] || 0) + 1;
    });
    const topValue = Object.entries(valueCounts).sort((a, b) => b[1] - a[1])[0]?.[0] || null;

    return { totalEntries, avgMood, topEmotion, topValue };
  }, [entries]);

  const currentDate = new Date().toLocaleDateString('en-US', {
    weekday: 'long',
    month: 'long',
    day: 'numeric',
    year: 'numeric',
  });

  const greetingName = user?.email ? user.email.split('@')[0] : 'friend';

  return (
    <div className="space-y-8 animate-fadeIn">
      {/* Welcome Banner */}
      <div className="relative rounded-3xl bg-gradient-to-r from-slate-900/90 via-[#0e1628] to-slate-900/90 border border-slate-800/90 p-6 sm:p-8 shadow-2xl overflow-hidden">
        <div className="absolute top-0 right-0 w-80 h-80 bg-cyan-500/10 rounded-full blur-3xl pointer-events-none" />

        <div className="relative z-10 flex flex-col md:flex-row md:items-center justify-between gap-6">
          <div className="space-y-2">
            <div className="flex items-center gap-2 text-xs font-semibold uppercase tracking-wider text-cyan-400">
              <Calendar className="h-3.5 w-3.5" />
              <span>{currentDate}</span>
            </div>
            <h1 className="text-2xl sm:text-3xl font-extrabold text-white">
              Welcome, <span className="capitalize">{greetingName}</span>.
            </h1>
            <p className="text-sm text-slate-400 max-w-lg">
              How are you feeling right now? Taking a few quiet moments to write can help
              uncover valuable emotional patterns.
            </p>
          </div>

          <div className="shrink-0">
            <NavLink to="/journal/new">
              <Button
                size="lg"
                variant="primary"
                className="w-full sm:w-auto text-sm sm:text-base font-semibold shadow-lg shadow-cyan-500/20"
                icon={<PenSquare className="h-4 w-4" />}
              >
                Write a Journal Entry
              </Button>
            </NavLink>
          </div>
        </div>
      </div>

      {/* Metrics Row */}
      <div className="grid grid-cols-2 lg:grid-cols-4 gap-4 sm:gap-6">
        <Card className="p-5">
          <div className="flex items-center justify-between">
            <span className="text-xs font-semibold text-slate-400 uppercase tracking-wider">
              Total Entries
            </span>
            <div className="p-2 rounded-xl bg-slate-800/80 text-cyan-400">
              <BookOpen className="h-4 w-4" />
            </div>
          </div>
          <div className="mt-3">
            {isLoading ? (
              <Skeleton className="h-7 w-14" />
            ) : (
              <span className="text-2xl font-bold text-white tracking-tight">
                {stats.totalEntries}
              </span>
            )}
          </div>
        </Card>

        <Card className="p-5">
          <div className="flex items-center justify-between">
            <span className="text-xs font-semibold text-slate-400 uppercase tracking-wider">
              Avg Mood Score
            </span>
            <div className="p-2 rounded-xl bg-slate-800/80 text-emerald-400">
              <TrendingUp className="h-4 w-4" />
            </div>
          </div>
          <div className="mt-3">
            {isLoading ? (
              <Skeleton className="h-7 w-16" />
            ) : stats.avgMood !== null ? (
              <div className="flex items-baseline gap-1">
                <span className="text-2xl font-bold text-white tracking-tight">
                  {stats.avgMood.toFixed(1)}
                </span>
                <span className="text-xs text-slate-400">/ 10</span>
              </div>
            ) : (
              <span className="text-xs text-slate-500 italic">No mood scores</span>
            )}
          </div>
        </Card>

        <Card className="p-5">
          <div className="flex items-center justify-between">
            <span className="text-xs font-semibold text-slate-400 uppercase tracking-wider">
              Dominant Emotion
            </span>
            <div className="p-2 rounded-xl bg-slate-800/80 text-rose-400">
              <Smile className="h-4 w-4" />
            </div>
          </div>
          <div className="mt-3">
            {isLoading ? (
              <Skeleton className="h-7 w-20" />
            ) : stats.topEmotion ? (
              <span className="text-lg sm:text-xl font-bold text-white capitalize truncate block">
                {stats.topEmotion}
              </span>
            ) : (
              <span className="text-xs text-slate-500 italic">None detected</span>
            )}
          </div>
        </Card>

        <Card className="p-5">
          <div className="flex items-center justify-between">
            <span className="text-xs font-semibold text-slate-400 uppercase tracking-wider">
              Primary Value
            </span>
            <div className="p-2 rounded-xl bg-slate-800/80 text-violet-400">
              <Compass className="h-4 w-4" />
            </div>
          </div>
          <div className="mt-3">
            {isLoading ? (
              <Skeleton className="h-7 w-20" />
            ) : stats.topValue ? (
              <span className="text-lg sm:text-xl font-bold text-white capitalize truncate block">
                {stats.topValue}
              </span>
            ) : (
              <span className="text-xs text-slate-500 italic">None detected</span>
            )}
          </div>
        </Card>
      </div>

      {/* Mood Trend Chart Preview */}
      <Card className="p-6 sm:p-7">
        <div className="flex items-center justify-between mb-4">
          <div>
            <h2 className="text-base font-bold text-white">Mood Trajectory</h2>
            <p className="text-xs text-slate-400 mt-0.5">
              Subjective mood trends across your recent entries
            </p>
          </div>
          <NavLink
            to="/insights"
            className="text-xs font-semibold text-cyan-400 hover:text-cyan-300 flex items-center gap-1 transition-colors"
          >
            All Insights <ArrowRight className="h-3.5 w-3.5" />
          </NavLink>
        </div>

        {isLoading ? (
          <Skeleton className="h-64 w-full" />
        ) : (
          <MoodChart entries={entries} />
        )}
      </Card>

      {/* Recent Journal Entries List */}
      <div className="space-y-4">
        <div className="flex items-center justify-between">
          <div>
            <h2 className="text-lg font-bold text-white">Recent Reflections</h2>
            <p className="text-xs text-slate-400">Your latest thoughts and companion responses</p>
          </div>
          {entries.length > 0 && (
            <NavLink
              to="/journal/history"
              className="text-xs font-semibold text-cyan-400 hover:text-cyan-300 flex items-center gap-1 transition-colors"
            >
              View All Entries <ArrowRight className="h-3.5 w-3.5" />
            </NavLink>
          )}
        </div>

        {isLoading ? (
          <div className="space-y-4">
            <Skeleton className="h-32 w-full" />
            <Skeleton className="h-32 w-full" />
            <Skeleton className="h-32 w-full" />
          </div>
        ) : entries.length === 0 ? (
          <EmptyState
            icon={<Sparkles className="h-7 w-7" />}
            title="Your sanctuary is empty"
            description="Start by writing your first reflection. ConsciousAI will provide grounding perspective and emotional clarity."
            actionLabel="Write First Entry"
            onAction={() => window.location.assign('/journal/new')}
          />
        ) : (
          <div className="grid grid-cols-1 md:grid-cols-2 gap-4 sm:gap-6">
            {entries.slice(0, 4).map((entry) => (
              <JournalCard key={entry.id} entry={entry} />
            ))}
          </div>
        )}
      </div>
    </div>
  );
};
