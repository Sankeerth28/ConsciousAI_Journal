import React from 'react';
import type { HTMLAttributes } from 'react';

export interface CardProps extends HTMLAttributes<HTMLDivElement> {
  hover?: boolean;
  glow?: 'cyan' | 'indigo' | 'emerald' | 'rose' | 'none';
}

export const Card: React.FC<CardProps> = ({
  children,
  hover = false,
  glow = 'none',
  className = '',
  ...props
}) => {
  const glowStyles = {
    none: '',
    cyan: 'hover:shadow-glow-cyan',
    indigo: 'hover:shadow-glow-indigo',
    emerald: 'hover:shadow-glow-emerald',
    rose: 'hover:shadow-glow-rose',
  };

  return (
    <div
      className={`rounded-2xl bg-slate-900/70 dark:bg-slate-900/70 light:bg-white/90 backdrop-blur-xl border border-slate-800/80 dark:border-slate-800/80 light:border-slate-200 shadow-xl transition-all duration-300 ${
        hover
          ? 'hover:border-slate-700 dark:hover:border-slate-700 light:hover:border-slate-300 hover:-translate-y-0.5'
          : ''
      } ${glowStyles[glow]} ${className}`}
      {...props}
    >
      {children}
    </div>
  );
};
