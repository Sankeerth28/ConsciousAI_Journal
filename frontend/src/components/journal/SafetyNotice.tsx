import React from 'react';
import { Heart, LifeBuoy, PhoneCall } from 'lucide-react';
import type { SafetyCheckResult } from '../../api/types';

export interface SafetyNoticeProps {
  safety: SafetyCheckResult;
}

export const SafetyNotice: React.FC<SafetyNoticeProps> = ({ safety }) => {
  if (safety.is_safe) return null;

  return (
    <div className="rounded-2xl bg-gradient-to-r from-rose-950/40 via-purple-950/30 to-slate-900/50 border border-rose-500/30 p-5 sm:p-6 shadow-xl space-y-4 animate-fadeIn">
      <div className="flex items-center gap-3">
        <div className="flex h-10 w-10 items-center justify-center rounded-xl bg-rose-500/20 text-rose-300 shrink-0">
          <LifeBuoy className="h-5 w-5" />
        </div>
        <div>
          <h3 className="text-sm sm:text-base font-semibold text-rose-200">
            Support & Compassionate Resources
          </h3>
          <p className="text-xs text-rose-300/80">
            You are not alone. Caring people are ready to listen whenever you need support.
          </p>
        </div>
      </div>

      {safety.crisis_response && (
        <div className="p-4 rounded-xl bg-black/40 border border-rose-500/20 text-xs sm:text-sm text-slate-200 leading-relaxed">
          {safety.crisis_response}
        </div>
      )}

      <div className="grid grid-cols-1 sm:grid-cols-2 gap-3 pt-2 text-xs text-slate-300">
        <div className="flex items-center gap-2 p-3 rounded-lg bg-slate-900/60 border border-slate-800">
          <PhoneCall className="h-4 w-4 text-cyan-400 shrink-0" />
          <span><strong>US & Canada:</strong> Call or text <strong>988</strong></span>
        </div>
        <div className="flex items-center gap-2 p-3 rounded-lg bg-slate-900/60 border border-slate-800">
          <Heart className="h-4 w-4 text-rose-400 shrink-0" />
          <span><strong>UK:</strong> 111 / 999 • <strong>EU:</strong> 112</span>
        </div>
      </div>
    </div>
  );
};
