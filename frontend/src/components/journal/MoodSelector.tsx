import React from 'react';
import { Smile, Frown, Meh, Sparkles } from 'lucide-react';

export interface MoodSelectorProps {
  value: number | null;
  onChange: (val: number | null) => void;
  disabled?: boolean;
}

export const MoodSelector: React.FC<MoodSelectorProps> = ({
  value,
  onChange,
  disabled = false,
}) => {
  const currentMood = value ?? 7.0;

  const getMoodDescriptor = (score: number) => {
    if (score >= 8.5) return { label: 'Joyful & Peaceful', color: 'text-emerald-400', bg: 'bg-emerald-500/10 border-emerald-500/30', icon: Sparkles };
    if (score >= 7.0) return { label: 'Good & Energized', color: 'text-cyan-400', bg: 'bg-cyan-500/10 border-cyan-500/30', icon: Smile };
    if (score >= 5.0) return { label: 'Balanced & Steady', color: 'text-blue-400', bg: 'bg-blue-500/10 border-blue-500/30', icon: Meh };
    if (score >= 3.5) return { label: 'Low Energy & Thoughtful', color: 'text-amber-400', bg: 'bg-amber-500/10 border-amber-500/30', icon: Meh };
    return { label: 'Struggling & Overwhelmed', color: 'text-rose-400', bg: 'bg-rose-500/10 border-rose-500/30', icon: Frown };
  };

  const moodInfo = getMoodDescriptor(currentMood);
  const MoodIcon = moodInfo.icon;

  return (
    <div className="space-y-3 rounded-2xl bg-slate-900/40 border border-slate-800/80 p-4 sm:p-5">
      <div className="flex items-center justify-between">
        <label className="text-xs font-semibold uppercase tracking-wider text-slate-300">
          Subjective Mood Score (1.0 - 10.0)
        </label>
        {value !== null ? (
          <div className={`flex items-center gap-2 px-3 py-1 rounded-full border text-xs font-semibold ${moodInfo.bg} ${moodInfo.color}`}>
            <MoodIcon className="h-3.5 w-3.5" />
            <span>{currentMood.toFixed(1)}</span>
            <span className="opacity-75 font-normal">• {moodInfo.label}</span>
          </div>
        ) : (
          <span className="text-xs text-slate-500 italic">Not set</span>
        )}
      </div>

      <div className="flex items-center gap-4 pt-2">
        <span className="text-xs font-bold text-rose-400">1.0</span>
        <input
          type="range"
          min="1.0"
          max="10.0"
          step="0.5"
          value={currentMood}
          disabled={disabled}
          onChange={(e) => onChange(parseFloat(e.target.value))}
          className="w-full h-2 rounded-lg bg-slate-800 appearance-none cursor-pointer accent-cyan-400 disabled:opacity-50"
        />
        <span className="text-xs font-bold text-emerald-400">10.0</span>
      </div>

      <div className="flex justify-between items-center text-[11px] text-slate-500 pt-1">
        <span>1: Challenging</span>
        <span>5: Neutral</span>
        <span>10: Optimal</span>
      </div>
    </div>
  );
};
