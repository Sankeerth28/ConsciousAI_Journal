import { forwardRef } from 'react';
import type { TextareaHTMLAttributes } from 'react';

export interface TextareaProps extends TextareaHTMLAttributes<HTMLTextAreaElement> {
  label?: string;
  error?: string;
  helperText?: string;
  showCount?: boolean;
  maxLength?: number;
  currentLength?: number;
}

export const Textarea = forwardRef<HTMLTextAreaElement, TextareaProps>(
  (
    {
      label,
      error,
      helperText,
      showCount = false,
      maxLength,
      currentLength = 0,
      className = '',
      id,
      ...props
    },
    ref
  ) => {
    const inputId = id || (label ? label.toLowerCase().replace(/\s+/g, '-') : undefined);

    return (
      <div className="w-full space-y-1.5">
        <div className="flex justify-between items-center">
          {label && (
            <label
              htmlFor={inputId}
              className="block text-xs font-semibold uppercase tracking-wider text-slate-300 dark:text-slate-300 light:text-slate-700"
            >
              {label}
            </label>
          )}

          {showCount && maxLength && (
            <span
              className={`text-xs ${
                currentLength >= maxLength ? 'text-rose-400 font-semibold' : 'text-slate-400'
              }`}
            >
              {currentLength} / {maxLength}
            </span>
          )}
        </div>

        <textarea
          ref={ref}
          id={inputId}
          maxLength={maxLength}
          className={`w-full rounded-xl bg-slate-900/60 dark:bg-slate-900/60 light:bg-white border text-sm text-slate-100 dark:text-slate-100 light:text-slate-900 placeholder:text-slate-500 p-3.5 transition-all duration-200 focus:outline-none focus:ring-2 focus:ring-cyan-500/30 focus:border-cyan-500 disabled:opacity-50 disabled:cursor-not-allowed resize-y ${
            error
              ? 'border-rose-500 focus:border-rose-500 focus:ring-rose-500/20'
              : 'border-slate-800 dark:border-slate-800 light:border-slate-300'
          } ${className}`}
          {...props}
        />

        {error && (
          <p className="text-xs text-rose-400 font-medium tracking-tight">
            {error}
          </p>
        )}

        {helperText && !error && (
          <p className="text-xs text-slate-400">{helperText}</p>
        )}
      </div>
    );
  }
);

Textarea.displayName = 'Textarea';
