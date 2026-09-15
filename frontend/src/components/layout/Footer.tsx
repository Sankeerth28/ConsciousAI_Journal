import React from 'react';
import { ShieldCheck, Heart } from 'lucide-react';

export const Footer: React.FC = () => {
  return (
    <footer className="border-t border-slate-800/60 py-6 px-4 sm:px-6 lg:px-8 mt-auto text-center text-xs text-slate-400">
      <div className="max-w-7xl mx-auto flex flex-col sm:flex-row items-center justify-between gap-4">
        <div className="flex items-center gap-1.5 text-slate-400">
          <span>ConsciousAI Journal</span>
          <span>•</span>
          <span className="flex items-center gap-1 text-cyan-400">
            <ShieldCheck className="h-3.5 w-3.5" /> Privacy-First Architecture
          </span>
        </div>

        <p className="flex items-center justify-center gap-1 text-slate-400">
          Crafted for mindful reflection <Heart className="h-3 w-3 text-rose-500 fill-rose-500" />
        </p>

        <p className="text-slate-400 text-[11px]">
          Non-clinical wellness companion. All models run securely.
        </p>
      </div>
    </footer>
  );
};
