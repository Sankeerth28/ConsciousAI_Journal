import React from 'react';
import { BookOpen } from 'lucide-react';
import { Button } from '../ui/Button';

export interface EmptyStateProps {
  icon?: React.ReactNode;
  title: string;
  description: string;
  actionLabel?: string;
  onAction?: () => void;
  className?: string;
}

export const EmptyState: React.FC<EmptyStateProps> = ({
  icon,
  title,
  description,
  actionLabel,
  onAction,
  className = '',
}) => {
  return (
    <div
      className={`flex flex-col items-center justify-center p-8 sm:p-12 text-center rounded-2xl border border-dashed border-slate-800 bg-slate-900/30 ${className}`}
    >
      <div className="flex h-14 w-14 items-center justify-center rounded-2xl bg-cyan-500/10 text-cyan-400 border border-cyan-500/20 mb-4">
        {icon || <BookOpen className="h-7 w-7" />}
      </div>
      <h3 className="text-base sm:text-lg font-semibold text-slate-100 dark:text-slate-100 light:text-slate-900">
        {title}
      </h3>
      <p className="mt-1.5 max-w-sm text-sm text-slate-400">
        {description}
      </p>
      {actionLabel && onAction && (
        <div className="mt-6">
          <Button onClick={onAction} size="sm" variant="primary">
            {actionLabel}
          </Button>
        </div>
      )}
    </div>
  );
};
