import React from 'react';
import {
  BarChart,
  Bar,
  XAxis,
  YAxis,
  Tooltip,
  ResponsiveContainer,
  Cell,
  CartesianGrid,
} from 'recharts';
import type { JournalEntryRead } from '../../api/types';
import { EmptyState } from '../common/EmptyState';
import { Smile } from 'lucide-react';

export interface EmotionDistributionChartProps {
  entries: JournalEntryRead[];
}

const BAR_COLORS = ['#06b6d4', '#6366f1', '#10b981', '#f59e0b', '#ec4899', '#8b5cf6'];

export const EmotionDistributionChart: React.FC<EmotionDistributionChartProps> = ({
  entries,
}) => {
  const counts: Record<string, number> = {};
  entries.forEach((e) => {
    if (e.top_emotion) {
      counts[e.top_emotion] = (counts[e.top_emotion] || 0) + 1;
    }
    e.detected_emotions?.forEach((emo) => {
      if (emo !== e.top_emotion) {
        counts[emo] = (counts[emo] || 0) + 1;
      }
    });
  });

  const chartData = Object.entries(counts)
    .map(([emotion, count]) => ({ emotion, count }))
    .sort((a, b) => b.count - a.count)
    .slice(0, 6);

  if (chartData.length === 0) {
    return (
      <EmptyState
        icon={<Smile className="h-6 w-6" />}
        title="No Emotion Data Yet"
        description="Write reflections to let ConsciousAI detect underlying emotional patterns."
        className="py-10"
      />
    );
  }

  return (
    <div className="w-full h-64 sm:h-72">
      <ResponsiveContainer width="100%" height="100%">
        <BarChart data={chartData} margin={{ top: 10, right: 10, left: -20, bottom: 0 }}>
          <CartesianGrid strokeDasharray="3 3" stroke="rgba(255, 255, 255, 0.07)" vertical={false} />
          <XAxis
            dataKey="emotion"
            stroke="#64748b"
            tick={{ fill: '#94a3b8', fontSize: 11 }}
            axisLine={false}
            tickLine={false}
          />
          <YAxis
            allowDecimals={false}
            stroke="#64748b"
            tick={{ fill: '#94a3b8', fontSize: 11 }}
            axisLine={false}
            tickLine={false}
          />
          <Tooltip
            content={({ active, payload }) => {
              if (active && payload && payload.length) {
                const item = payload[0].payload;
                return (
                  <div className="rounded-xl bg-slate-900 border border-slate-700 p-3 shadow-xl text-xs">
                    <p className="font-semibold text-slate-200 capitalize">{item.emotion}</p>
                    <p className="text-cyan-400 font-bold mt-1">
                      Detected {item.count} {item.count === 1 ? 'time' : 'times'}
                    </p>
                  </div>
                );
              }
              return null;
            }}
          />
          <Bar dataKey="count" radius={[6, 6, 0, 0]}>
            {chartData.map((_, index) => (
              <Cell key={`cell-${index}`} fill={BAR_COLORS[index % BAR_COLORS.length]} />
            ))}
          </Bar>
        </BarChart>
      </ResponsiveContainer>
    </div>
  );
};
