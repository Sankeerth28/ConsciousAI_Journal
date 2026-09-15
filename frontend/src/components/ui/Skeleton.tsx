import React from 'react';

export interface SkeletonProps {
  className?: string;
}

export const Skeleton: React.FC<SkeletonProps> = ({ className = '' }) => {
  return (
    <div
      className={`animate-pulse rounded-xl bg-slate-800/60 dark:bg-slate-800/60 light:bg-slate-200 ${className}`}
    />
  );
};
