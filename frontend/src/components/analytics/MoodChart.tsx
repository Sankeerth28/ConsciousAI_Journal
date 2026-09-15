import React from 'react';
import {
  AreaChart,
  Area,
  XAxis,
  YAxis,
  Tooltip,
  ResponsiveContainer,
  CartesianGrid,
} from 'recharts';
import type { JournalEntryRead } from '../../api/types';
import { EmptyState } from '../common/EmptyState';
import { LineChart as LineChartIcon } from 'lucide-react';

export interface MoodChartProps {
  entries: JournalEntryRead[];
}

export const MoodChart: React.FC<MoodChartProps> = ({ entries }) => {
  const validEntries = entries
    .filter((e) => e.mood_score !== null && e.mood_score !== undefined)
    .sort(
      (a, b) => new Date(a.created_at).getTime() - new Date(b.created_at).getTime()
    );

  if (validEntries.length < 2) {
    return (
      <EmptyState
        icon={<LineChartIcon className="h-6 w-6" />}
        title="Not Enough Mood Data"
        description="Write at least two journal entries with a mood score to view your emotional timeline."
        className="py-10"
      />
    );
  }

  const chartData = validEntries.map((e) => ({
    date: new Date(e.created_at).toLocaleDateString('en-US', {
      month: 'short',
      day: 'numeric',
    }),
    mood: e.mood_score,
    topEmotion: e.top_emotion || 'Neutral',
  }));

  return (
    <div className="w-full h-64 sm:h-72">
      <ResponsiveContainer width="100%" height="100%">
        <AreaChart data={chartData} margin={{ top: 10, right: 10, left: -20, bottom: 0 }}>
          <defs>
            <linearGradient id="moodGradient" x1="0" y1="0" x2="0" y2="1">
              <stop offset="5%" stopColor="#06b6d4" stopOpacity={0.4} />
              <stop offset="95%" stopColor="#06b6d4" stopOpacity={0.0} />
            </linearGradient>
          </defs>
          <CartesianGrid strokeDasharray="3 3" stroke="rgba(255, 255, 255, 0.07)" vertical={false} />
          <XAxis
            dataKey="date"
            stroke="#64748b"
            tick={{ fill: '#94a3b8', fontSize: 11 }}
            axisLine={false}
            tickLine={false}
          />
          <YAxis
            domain={[1, 10]}
            ticks={[2, 4, 6, 8, 10]}
            stroke="#64748b"
            tick={{ fill: '#94a3b8', fontSize: 11 }}
            axisLine={false}
            tickLine={false}
          />
          <Tooltip
            content={({ active, payload }) => {
              if (active && payload && payload.length) {
                const data = payload[0].payload;
                return (
                  <div className="rounded-xl bg-slate-900 border border-slate-700 p-3 shadow-xl text-xs">
                    <p className="font-semibold text-slate-200">{data.date}</p>
                    <p className="text-cyan-400 font-bold mt-1">
                      Mood Score: {data.mood?.toFixed(1)} / 10
                    </p>
                    <p className="text-slate-400 text-[11px] mt-0.5">
                      Emotion: {data.topEmotion}
                    </p>
                  </div>
                );
              }
              return null;
            }}
          />
          <Area
            type="monotone"
            dataKey="mood"
            stroke="#06b6d4"
            strokeWidth={3}
            fillOpacity={1}
            fill="url(#moodGradient)"
          />
        </AreaChart>
      </ResponsiveContainer>
    </div>
  );
};
