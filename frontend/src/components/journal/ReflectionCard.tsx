import React, { useState } from 'react';
import { Sparkles, ThumbsUp, Check, Bot } from 'lucide-react';
import type { ReflectionResult } from '../../api/types';
import { Badge } from '../ui/Badge';
import { submitJournalFeedback } from '../../api/journals';

export interface ReflectionCardProps {
  reflection: ReflectionResult;
  entryId?: number;
  detectedEmotions?: string[];
  detectedValues?: string[];
}

export const ReflectionCard: React.FC<ReflectionCardProps> = ({
  reflection,
  entryId,
  detectedEmotions = [],
  detectedValues = [],
}) => {
  const [feedbackSent, setFeedbackSent] = useState<string | null>(null);
  const [isSubmitting, setIsSubmitting] = useState(false);

  const handleFeedback = async (type: string) => {
    if (!entryId || feedbackSent) return;
    try {
      setIsSubmitting(true);
      await submitJournalFeedback(entryId, { feedback_type: type });
      setFeedbackSent(type);
    } catch {
      // Gracefully ignore feedback errors
    } finally {
      setIsSubmitting(false);
    }
  };

  return (
    <div className="relative rounded-2xl bg-gradient-to-b from-slate-900/90 via-[#0c1222]/80 to-slate-900/90 border border-cyan-500/20 shadow-2xl p-6 sm:p-7 backdrop-blur-xl overflow-hidden">
      {/* Decorative top ambient glow */}
      <div className="absolute top-0 right-1/4 w-72 h-24 bg-cyan-500/10 rounded-full blur-3xl pointer-events-none" />

      {/* Header with Companion Persona */}
      <div className="flex flex-wrap items-center justify-between gap-3 pb-4 border-b border-slate-800/80">
        <div className="flex items-center gap-3">
          <div className="flex h-9 w-9 items-center justify-center rounded-xl bg-gradient-to-tr from-cyan-500 to-indigo-600 text-white shadow-glow-cyan">
            <Sparkles className="h-4 w-4" />
          </div>
          <div>
            <h3 className="text-sm font-bold text-slate-100 flex items-center gap-2">
              ConsciousAI Reflection
              <Badge variant="cyan" size="sm">
                {reflection.persona}
              </Badge>
            </h3>
            <span className="text-[11px] text-slate-400 flex items-center gap-1">
              <Bot className="h-3 w-3 text-cyan-400" />
              {reflection.model_name}
            </span>
          </div>
        </div>
      </div>

      {/* Reflection Content */}
      <div className="py-5 text-sm sm:text-base text-slate-200 leading-relaxed whitespace-pre-wrap font-sans">
        {reflection.response}
      </div>

      {/* Emotions & Values Detected */}
      {(detectedEmotions.length > 0 || detectedValues.length > 0) && (
        <div className="pt-4 border-t border-slate-800/60 flex flex-wrap gap-4">
          {detectedEmotions.length > 0 && (
            <div className="flex items-center gap-1.5 flex-wrap">
              <span className="text-xs text-slate-400 font-medium">Emotions:</span>
              {detectedEmotions.map((emo, idx) => (
                <span
                  key={idx}
                  className="text-xs px-2.5 py-0.5 rounded-full bg-cyan-500/10 text-cyan-300 border border-cyan-500/20"
                >
                  {emo}
                </span>
              ))}
            </div>
          )}

          {detectedValues.length > 0 && (
            <div className="flex items-center gap-1.5 flex-wrap">
              <span className="text-xs text-slate-400 font-medium">Core Values:</span>
              {detectedValues.map((val, idx) => (
                <span
                  key={idx}
                  className="text-xs px-2.5 py-0.5 rounded-full bg-indigo-500/10 text-indigo-300 border border-indigo-500/20"
                >
                  {val}
                </span>
              ))}
            </div>
          )}
        </div>
      )}

      {/* Feedback interaction */}
      {entryId && (
        <div className="mt-5 pt-4 border-t border-slate-800/60 flex items-center justify-between text-xs text-slate-400">
          <span>Was this reflection helpful?</span>
          {feedbackSent ? (
            <span className="text-emerald-400 font-medium flex items-center gap-1">
              <Check className="h-3.5 w-3.5" /> Thank you for your feedback!
            </span>
          ) : (
            <div className="flex items-center gap-2">
              <button
                type="button"
                disabled={isSubmitting}
                onClick={() => handleFeedback('Insightful')}
                className="px-2.5 py-1 rounded-lg bg-slate-800 hover:bg-slate-700 text-slate-300 hover:text-white transition-colors flex items-center gap-1"
              >
                <ThumbsUp className="h-3 w-3" /> Insightful
              </button>
              <button
                type="button"
                disabled={isSubmitting}
                onClick={() => handleFeedback('Helpful')}
                className="px-2.5 py-1 rounded-lg bg-slate-800 hover:bg-slate-700 text-slate-300 hover:text-white transition-colors"
              >
                Helpful
              </button>
            </div>
          )}
        </div>
      )}
    </div>
  );
};
