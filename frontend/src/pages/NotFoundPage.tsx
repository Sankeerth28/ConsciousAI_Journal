import React from 'react';
import { NavLink } from 'react-router-dom';
import { Sparkles, Home, ArrowLeft } from 'lucide-react';
import { Button } from '../components/ui/Button';

export const NotFoundPage: React.FC = () => {
  return (
    <div className="min-h-screen bg-[#090d16] text-slate-100 flex flex-col items-center justify-center p-6 text-center">
      <div className="flex h-14 w-14 items-center justify-center rounded-2xl bg-cyan-500/10 text-cyan-400 border border-cyan-500/20 mb-6">
        <Sparkles className="h-7 w-7" />
      </div>

      <h1 className="text-6xl font-extrabold tracking-tight text-white mb-2 font-mono">
        404
      </h1>
      <h2 className="text-xl font-bold text-slate-200 mb-2">
        Page Not Found
      </h2>
      <p className="text-sm text-slate-400 max-w-sm mb-8 leading-relaxed">
        The reflection page or path you requested does not exist in your sanctuary.
      </p>

      <div className="flex items-center gap-3">
        <NavLink to="/dashboard">
          <Button variant="primary" icon={<Home className="h-4 w-4" />}>
            Go to Dashboard
          </Button>
        </NavLink>
        <NavLink to="/">
          <Button variant="outline" icon={<ArrowLeft className="h-4 w-4" />}>
            Home
          </Button>
        </NavLink>
      </div>
    </div>
  );
};
