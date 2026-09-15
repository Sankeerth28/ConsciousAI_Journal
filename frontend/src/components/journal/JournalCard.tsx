import React from 'react';
import { NavLink } from 'react-router-dom';
import { Lock, ArrowUpRight, Calendar, Tag } from 'lucide-react';
import type { JournalEntryRead } from '../../api/types';
import { Card } from '../ui/Card';
import { EmotionBadge, ValueBadge } from './EmotionBadge';

export interface JournalCardProps {
  entry: JournalEntryRead;
}

export const JournalCard: React.FC<JournalCardProps> = ({ entry }) => {
  const formattedDate = new Date(entry.created_at).toLocaleDateString('en-US', {
    month: 'short',
    day: 'numeric',
    year: 'numeric',
    hour: '2-digit',
    minute: '2-digit',
  });

  const getMoodColor = (mood: number | null | undefined) => {
    if (mood === null || mood === undefined) return 'bg-slate-800 text-slate-400 border-slate-700';
    if (mood >= 8.0) return 'bg-emerald-500/10 text-emerald-400 border-emerald-500/30';
    if (mood >= 6.0) return 'bg-cyan-500/10 text-cyan-400 border-cyan-500/30';
    if (mood >= 4.0) return 'bg-amber-500/10 text-amber-400 border-amber-500/30';
    return 'bg-rose-500/10 text-rose-400 border-rose-500/30';
  };

  return (
    <NavLink to={`/journal/${entry.id}`} className="block group">
      <Card hover className="p-5 sm:p-6 transition-all duration-200">
        {/* Top bar: Date, privacy lock, mood score */}
        <div className="flex items-center justify-between gap-3 text-xs mb-3">
          <div className="flex items-center gap-2 text-slate-400">
            <Calendar className="h-3.5 w-3.5 text-slate-500" />
            <span>{formattedDate}</span>
            {entry.is_private && (
              <span className="flex items-center gap-1 text-amber-400/90 font-medium ml-1">
                <Lock className="h-3 w-3" /> Private
              </span>
            )}
          </div>

          {entry.mood_score !== null && entry.mood_score !== undefined && (
            <span
              className={`px-2.5 py-0.5 rounded-full border text-xs font-semibold ${getMoodColor(
                entry.mood_score
              )}`}
            >
              Mood {entry.mood_score.toFixed(1)}
            </span>
          )}
        </div>

        {/* Text snippet */}
        <p className="text-sm sm:text-base text-slate-200 line-clamp-3 leading-relaxed group-hover:text-cyan-200 transition-colors">
          {entry.text}
        </p>

        {/* Emotions, Values & Tags */}
        <div className="mt-4 pt-4 border-t border-slate-800/60 flex flex-wrap items-center justify-between gap-3 text-xs">
          <div className="flex flex-wrap items-center gap-2">
            {entry.top_emotion && <EmotionBadge emotion={entry.top_emotion} />}
            {entry.top_value && <ValueBadge value={entry.top_value} />}
            {entry.tags?.slice(0, 3).map((tag, idx) => (
              <span
                key={idx}
                className="flex items-center gap-1 text-[11px] px-2 py-0.5 rounded-md bg-slate-800/80 text-slate-300"
              >
                <Tag className="h-2.5 w-2.5 text-slate-500" />
                {tag}
              </span>
            ))}
          </div>

          <span className="text-cyan-400 group-hover:translate-x-0.5 group-hover:-translate-y-0.5 transition-transform flex items-center gap-0.5 font-medium">
            Read <ArrowUpRight className="h-3.5 w-3.5" />
          </span>
        </div>
      </Card>
    </NavLink>
  );
};
