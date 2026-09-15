import React, { useEffect, useState } from 'react';
import { useNavigate } from 'react-router-dom';
import {
  User,
  Shield,
  Palette,
  Activity,
  LogOut,
  Sun,
  Moon,
  Database,
  Cpu,
  CheckCircle2,
  AlertCircle,
} from 'lucide-react';
import { useAuth } from '../context/AuthContext';
import { useTheme } from '../context/ThemeContext';
import { checkHealth, checkReadiness } from '../api/health';
import type { ReadyResponse } from '../api/types';
import { Card } from '../components/ui/Card';
import { Button } from '../components/ui/Button';

export const SettingsPage: React.FC = () => {
  const { user, logout } = useAuth();
  const { theme, setTheme } = useTheme();
  const navigate = useNavigate();

  const [healthStatus, setHealthStatus] = useState<'online' | 'offline'>('online');
  const [readiness, setReadiness] = useState<ReadyResponse | null>(null);

  useEffect(() => {
    let isMounted = true;
    const fetchStatus = async () => {
      try {
        await checkHealth();
        const r = await checkReadiness();
        if (isMounted) {
          setHealthStatus('online');
          setReadiness(r);
        }
      } catch {
        if (isMounted) {
          setHealthStatus('offline');
        }
      }
    };

    fetchStatus();
    return () => {
      isMounted = false;
    };
  }, []);

  const handleLogout = () => {
    logout();
    navigate('/login');
  };

  const accountCreated = user?.created_at
    ? new Date(user.created_at).toLocaleDateString('en-US', {
        month: 'long',
        day: 'numeric',
        year: 'numeric',
      })
    : 'Unknown';

  return (
    <div className="max-w-3xl mx-auto space-y-8 animate-fadeIn">
      {/* Title */}
      <div className="pb-4 border-b border-slate-800/80">
        <h1 className="text-2xl font-bold text-white tracking-tight">
          Account & Preferences
        </h1>
        <p className="text-xs text-slate-400 mt-0.5">
          Manage your profile, theme settings, and inspect system telemetry
        </p>
      </div>

      {/* Profile Section */}
      <Card className="p-6 space-y-4">
        <div className="flex items-center gap-3 pb-3 border-b border-slate-800">
          <div className="p-2 rounded-xl bg-cyan-500/10 text-cyan-400">
            <User className="h-5 w-5" />
          </div>
          <div>
            <h2 className="text-sm font-bold text-white">Profile Information</h2>
            <p className="text-xs text-slate-400">Authenticated account credentials</p>
          </div>
        </div>

        <div className="grid grid-cols-1 sm:grid-cols-2 gap-4 text-xs">
          <div className="p-3.5 rounded-xl bg-slate-900/60 border border-slate-800">
            <span className="text-slate-500 block uppercase tracking-wider font-semibold text-[10px]">
              Email Address
            </span>
            <span className="text-sm font-medium text-slate-200 mt-0.5 block truncate">
              {user?.email || 'N/A'}
            </span>
          </div>

          <div className="p-3.5 rounded-xl bg-slate-900/60 border border-slate-800">
            <span className="text-slate-500 block uppercase tracking-wider font-semibold text-[10px]">
              Member Since
            </span>
            <span className="text-sm font-medium text-slate-200 mt-0.5 block">
              {accountCreated}
            </span>
          </div>

          <div className="p-3.5 rounded-xl bg-slate-900/60 border border-slate-800 sm:col-span-2">
            <span className="text-slate-500 block uppercase tracking-wider font-semibold text-[10px]">
              User Identifier (UUID)
            </span>
            <span className="text-xs font-mono text-slate-400 mt-0.5 block select-all">
              {user?.id || 'N/A'}
            </span>
          </div>
        </div>
      </Card>

      {/* Theme Preferences */}
      <Card className="p-6 space-y-4">
        <div className="flex items-center gap-3 pb-3 border-b border-slate-800">
          <div className="p-2 rounded-xl bg-indigo-500/10 text-indigo-400">
            <Palette className="h-5 w-5" />
          </div>
          <div>
            <h2 className="text-sm font-bold text-white">Appearance & Theme</h2>
            <p className="text-xs text-slate-400">Select your preferred interface aesthetic</p>
          </div>
        </div>

        <div className="grid grid-cols-2 gap-3">
          <button
            type="button"
            onClick={() => setTheme('dark')}
            className={`flex items-center justify-center gap-2 p-3.5 rounded-xl border text-xs font-semibold transition-all ${
              theme === 'dark'
                ? 'bg-cyan-500/10 border-cyan-500/50 text-cyan-300 shadow-glow-cyan'
                : 'bg-slate-900/40 border-slate-800 text-slate-400 hover:text-white'
            }`}
          >
            <Moon className="h-4 w-4" />
            <span>Dark Theme (Recommended)</span>
          </button>

          <button
            type="button"
            onClick={() => setTheme('light')}
            className={`flex items-center justify-center gap-2 p-3.5 rounded-xl border text-xs font-semibold transition-all ${
              theme === 'light'
                ? 'bg-cyan-500/10 border-cyan-500/50 text-cyan-300 shadow-glow-cyan'
                : 'bg-slate-900/40 border-slate-800 text-slate-400 hover:text-white'
            }`}
          >
            <Sun className="h-4 w-4" />
            <span>Light Theme</span>
          </button>
        </div>
      </Card>

      {/* Backend Infrastructure Telemetry */}
      <Card className="p-6 space-y-4">
        <div className="flex items-center gap-3 pb-3 border-b border-slate-800">
          <div className="p-2 rounded-xl bg-emerald-500/10 text-emerald-400">
            <Activity className="h-5 w-5" />
          </div>
          <div>
            <h2 className="text-sm font-bold text-white">Backend System Telemetry</h2>
            <p className="text-xs text-slate-400">Real-time status probes from FastAPI</p>
          </div>
        </div>

        <div className="grid grid-cols-1 sm:grid-cols-3 gap-3 text-xs">
          <div className="p-3.5 rounded-xl bg-slate-900/60 border border-slate-800 flex items-center justify-between">
            <div className="flex items-center gap-2">
              <Cpu className="h-4 w-4 text-cyan-400" />
              <span>FastAPI API</span>
            </div>
            {healthStatus === 'online' ? (
              <span className="text-emerald-400 font-semibold flex items-center gap-1">
                <CheckCircle2 className="h-3.5 w-3.5" /> Healthy
              </span>
            ) : (
              <span className="text-rose-400 font-semibold flex items-center gap-1">
                <AlertCircle className="h-3.5 w-3.5" /> Offline
              </span>
            )}
          </div>

          <div className="p-3.5 rounded-xl bg-slate-900/60 border border-slate-800 flex items-center justify-between">
            <div className="flex items-center gap-2">
              <Database className="h-4 w-4 text-indigo-400" />
              <span>PostgreSQL 16</span>
            </div>
            {readiness?.database === 'connected' ? (
              <span className="text-emerald-400 font-semibold flex items-center gap-1">
                <CheckCircle2 className="h-3.5 w-3.5" /> Connected
              </span>
            ) : (
              <span className="text-amber-400 font-semibold">Probing...</span>
            )}
          </div>

          <div className="p-3.5 rounded-xl bg-slate-900/60 border border-slate-800 flex items-center justify-between">
            <div className="flex items-center gap-2">
              <Activity className="h-4 w-4 text-rose-400" />
              <span>Redis 7</span>
            </div>
            {readiness?.redis === 'connected' ? (
              <span className="text-emerald-400 font-semibold flex items-center gap-1">
                <CheckCircle2 className="h-3.5 w-3.5" /> Connected
              </span>
            ) : (
              <span className="text-amber-400 font-semibold">Probing...</span>
            )}
          </div>
        </div>
      </Card>

      {/* Security and Logout */}
      <Card className="p-6 flex flex-col sm:flex-row items-center justify-between gap-4 border-rose-500/20 bg-gradient-to-r from-slate-900/80 to-rose-950/20">
        <div className="flex items-center gap-3">
          <div className="p-2 rounded-xl bg-rose-500/10 text-rose-400">
            <Shield className="h-5 w-5" />
          </div>
          <div>
            <h2 className="text-sm font-bold text-white">Session Security</h2>
            <p className="text-xs text-slate-400">
              Clear all authentication tokens and end this active session
            </p>
          </div>
        </div>

        <Button
          variant="danger"
          size="sm"
          onClick={handleLogout}
          icon={<LogOut className="h-4 w-4" />}
        >
          Sign Out of ConsciousAI
        </Button>
      </Card>
    </div>
  );
};
