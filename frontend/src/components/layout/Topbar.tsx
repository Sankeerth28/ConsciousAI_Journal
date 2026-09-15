import React, { useEffect, useState } from 'react';
import { NavLink } from 'react-router-dom';
import { Sun, Moon, PenSquare, Activity } from 'lucide-react';
import { useTheme } from '../../context/ThemeContext';
import { checkReadiness } from '../../api/health';
import { Button } from '../ui/Button';

export const Topbar: React.FC = () => {
  const { theme, toggleTheme } = useTheme();
  const [backendStatus, setBackendStatus] = useState<'connected' | 'error' | 'checking'>('checking');

  useEffect(() => {
    let isMounted = true;
    const probe = async () => {
      try {
        const res = await checkReadiness();
        if (isMounted) {
          setBackendStatus(res.status === 'ready' ? 'connected' : 'error');
        }
      } catch {
        if (isMounted) {
          setBackendStatus('error');
        }
      }
    };

    probe();
    const interval = setInterval(probe, 30000);
    return () => {
      isMounted = false;
      clearInterval(interval);
    };
  }, []);

  return (
    <header className="sticky top-0 z-30 flex h-16 items-center justify-between px-4 sm:px-6 lg:px-8 border-b border-slate-800/80 bg-[#090d16]/80 backdrop-blur-xl">
      {/* Left: Brand mobile fallback & status indicator */}
      <div className="flex items-center gap-3">
        <div className="flex items-center gap-2 px-2.5 py-1 rounded-full bg-slate-800/50 border border-slate-700/60 text-xs">
          <Activity
            className={`h-3.5 w-3.5 ${
              backendStatus === 'connected'
                ? 'text-emerald-400 animate-pulse'
                : backendStatus === 'error'
                ? 'text-rose-400'
                : 'text-amber-400 animate-spin'
            }`}
          />
          <span className="text-slate-300 font-medium text-[11px]">
            {backendStatus === 'connected'
              ? 'FastAPI Live'
              : backendStatus === 'error'
              ? 'Backend Offline'
              : 'Connecting...'}
          </span>
        </div>
      </div>

      {/* Right actions: Theme toggle + New Journal CTA */}
      <div className="flex items-center gap-2.5 sm:gap-3">
        <button
          onClick={toggleTheme}
          aria-label="Toggle dark/light theme"
          className="rounded-xl p-2 text-slate-400 hover:text-slate-100 hover:bg-slate-800/60 border border-transparent hover:border-slate-700/60 transition-all"
        >
          {theme === 'dark' ? <Sun className="h-4 w-4" /> : <Moon className="h-4 w-4" />}
        </button>

        <NavLink to="/journal/new">
          <Button size="sm" variant="primary" icon={<PenSquare className="h-3.5 w-3.5" />}>
            <span className="hidden sm:inline">New Entry</span>
          </Button>
        </NavLink>
      </div>
    </header>
  );
};
