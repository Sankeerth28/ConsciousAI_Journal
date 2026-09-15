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
import { Compass } from 'lucide-react';

export interface ValueDistributionChartProps {
  entries: JournalEntryRead[];
}

const VALUE_COLORS = ['#8b5cf6', '#6366f1', '#06b6d4', '#10b981', '#f59e0b'];

export const ValueDistributionChart: React.FC<ValueDistributionChartProps> = ({ entries }) => {
  const counts: Record<string, number> = {};
  entries.forEach((e) => {
    if (e.top_value) {
      counts[e.top_value] = (counts[e.top_value] || 0) + 1;
    }
    e.detected_values?.forEach((val) => {
      if (val !== e.top_value) {
        counts[val] = (counts[val] || 0) + 1;
      }
    });
  });

  const chartData = Object.entries(counts)
    .map(([value, count]) => ({ value, count }))
    .sort((a, b) => b.count - a.count)
    .slice(0, 5);

  if (chartData.length === 0) {
    return (
      <EmptyState
        icon={<Compass className="h-6 w-6" />}
        title="No Values Detected Yet"
        description="Your core principles, motivations, and values will appear here as you journal."
        className="py-10"
      />
    );
  }

  return (
    <div className="w-full h-64 sm:h-72">
      <ResponsiveContainer width="100%" height="100%">
        <BarChart
          layout="vertical"
          data={chartData}
          margin={{ top: 10, right: 20, left: 30, bottom: 0 }}
        >
          <CartesianGrid strokeDasharray="3 3" stroke="rgba(255, 255, 255, 0.07)" horizontal={false} />
          <XAxis
            type="number"
            allowDecimals={false}
            stroke="#64748b"
            tick={{ fill: '#94a3b8', fontSize: 11 }}
            axisLine={false}
            tickLine={false}
          />
          <YAxis
            dataKey="value"
            type="category"
            stroke="#64748b"
            tick={{ fill: '#94a3b8', fontSize: 11 }}
            axisLine={false}
            tickLine={false}
            width={90}
          />
          <Tooltip
            content={({ active, payload }) => {
              if (active && payload && payload.length) {
                const item = payload[0].payload;
                return (
                  <div className="rounded-xl bg-slate-900 border border-slate-700 p-3 shadow-xl text-xs">
                    <p className="font-semibold text-slate-200 capitalize">{item.value}</p>
                    <p className="text-violet-400 font-bold mt-1">
                      Identified {item.count} {item.count === 1 ? 'time' : 'times'}
                    </p>
                  </div>
                );
              }
              return null;
            }}
          />
          <Bar dataKey="count" radius={[0, 6, 6, 0]}>
            {chartData.map((_, index) => (
              <Cell key={`cell-${index}`} fill={VALUE_COLORS[index % VALUE_COLORS.length]} />
            ))}
          </Bar>
        </BarChart>
      </ResponsiveContainer>
    </div>
  );
};
