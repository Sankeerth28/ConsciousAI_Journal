import React from 'react';
import { NavLink } from 'react-router-dom';
import {
  Sparkles,
  ArrowRight,
  ShieldCheck,
  Brain,
  Compass,
  Lock,
  Feather,
} from 'lucide-react';
import { Button } from '../components/ui/Button';
import { Card } from '../components/ui/Card';
import { useAuth } from '../context/AuthContext';

export const LandingPage: React.FC = () => {
  const { isAuthenticated } = useAuth();

  const features = [
    {
      icon: Sparkles,
      title: 'AI-Powered Reflections',
      description:
        'Engage with adaptive companion personas that provide insightful, compassionate reframing for every thought.',
      color: 'text-cyan-400',
      border: 'border-cyan-500/20',
    },
    {
      icon: Brain,
      title: 'Emotion Awareness',
      description:
        'Automatic detection of underlying emotional nuance helps you name, understand, and navigate what you feel.',
      color: 'text-rose-400',
      border: 'border-rose-500/20',
    },
    {
      icon: Compass,
      title: 'Personal Growth Insights',
      description:
        'Identify your subconscious core values and track mood trends over time with zero manual tagging required.',
      color: 'text-violet-400',
      border: 'border-violet-500/20',
    },
    {
      icon: ShieldCheck,
      title: 'Privacy-First Architecture',
      description:
        'Your journals belong to you. Strict cryptographic authorization and private visibility controls safeguard your sanctuary.',
      color: 'text-emerald-400',
      border: 'border-emerald-500/20',
    },
  ];

  return (
    <div className="min-h-screen bg-[#090d16] text-slate-100 flex flex-col selection:bg-cyan-500/20 selection:text-cyan-300">
      {/* Navigation Bar */}
      <header className="sticky top-0 z-40 w-full border-b border-slate-800/80 bg-[#090d16]/80 backdrop-blur-xl">
        <div className="max-w-7xl mx-auto flex h-16 items-center justify-between px-4 sm:px-6 lg:px-8">
          <NavLink to="/" className="flex items-center gap-3">
            <div className="flex h-9 w-9 items-center justify-center rounded-xl bg-gradient-to-tr from-cyan-500 to-indigo-600 shadow-glow-cyan">
              <Sparkles className="h-4 w-4 text-white" />
            </div>
            <span className="text-base font-bold tracking-tight text-slate-100">
              ConsciousAI <span className="text-cyan-400">Journal</span>
            </span>
          </NavLink>

          <div className="flex items-center gap-3">
            {isAuthenticated ? (
              <NavLink to="/dashboard">
                <Button size="sm" variant="primary" icon={<ArrowRight className="h-4 w-4" />}>
                  Go to Dashboard
                </Button>
              </NavLink>
            ) : (
              <>
                <NavLink to="/login">
                  <Button size="sm" variant="ghost">
                    Sign In
                  </Button>
                </NavLink>
                <NavLink to="/register">
                  <Button size="sm" variant="primary">
                    Get Started
                  </Button>
                </NavLink>
              </>
            )}
          </div>
        </div>
      </header>

      {/* Hero Section with Ambient Glow */}
      <section className="relative pt-20 pb-24 sm:pt-28 sm:pb-32 px-4 sm:px-6 lg:px-8 overflow-hidden">
        <div className="absolute top-1/4 left-1/2 -translate-x-1/2 -translate-y-1/2 w-[600px] h-[350px] bg-gradient-to-r from-cyan-500/15 via-indigo-600/15 to-purple-500/15 blur-[120px] rounded-full pointer-events-none" />

        <div className="max-w-4xl mx-auto text-center relative z-10">
          <div className="inline-flex items-center gap-2 px-3 py-1.5 rounded-full border border-cyan-500/30 bg-cyan-500/10 text-cyan-300 text-xs font-semibold mb-8 animate-fadeIn">
            <Sparkles className="h-3.5 w-3.5 text-cyan-400" />
            <span>AI-Guided Emotional Intelligence & Reflection</span>
          </div>

          <h1 className="text-4xl sm:text-6xl lg:text-7xl font-extrabold tracking-tight text-white leading-[1.1]">
            Understand your thoughts.{' '}
            <span className="bg-gradient-to-r from-cyan-400 via-indigo-400 to-violet-400 bg-clip-text text-transparent">
              Reflect with intention.
            </span>
          </h1>

          <p className="mt-6 text-base sm:text-xl text-slate-400 max-w-2xl mx-auto leading-relaxed">
            A sanctuary for your inner monologue. Capture daily experiences, unlock
            emotional clarity with adaptive companion personas, and discover what matters most.
          </p>

          <div className="mt-10 flex flex-col sm:flex-row items-center justify-center gap-4">
            <NavLink to={isAuthenticated ? '/dashboard' : '/register'} className="w-full sm:w-auto">
              <Button size="lg" variant="primary" className="w-full sm:w-auto text-base">
                Start Journaling Free <ArrowRight className="h-5 w-5 ml-1" />
              </Button>
            </NavLink>

            {!isAuthenticated && (
              <NavLink to="/login" className="w-full sm:w-auto">
                <Button size="lg" variant="outline" className="w-full sm:w-auto text-base">
                  Sign In to Your Space
                </Button>
              </NavLink>
            )}
          </div>
        </div>

        {/* Abstract Emotional Orbit Preview Card */}
        <div className="max-w-4xl mx-auto mt-16 relative z-10">
          <div className="rounded-2xl bg-gradient-to-b from-slate-800/60 to-slate-900/90 border border-slate-700/60 p-6 sm:p-8 backdrop-blur-2xl shadow-2xl">
            <div className="flex items-center justify-between pb-4 border-b border-slate-800">
              <div className="flex items-center gap-2">
                <div className="h-3 w-3 rounded-full bg-rose-500/60" />
                <div className="h-3 w-3 rounded-full bg-amber-500/60" />
                <div className="h-3 w-3 rounded-full bg-emerald-500/60" />
                <span className="text-xs text-slate-400 ml-2 font-mono">reflection_preview.tsx</span>
              </div>
              <span className="text-xs font-semibold px-2.5 py-0.5 rounded-full bg-cyan-500/10 text-cyan-400 border border-cyan-500/20">
                Persona: Supportive
              </span>
            </div>

            <div className="grid grid-cols-1 md:grid-cols-2 gap-6 pt-6">
              <div className="space-y-3">
                <span className="text-xs uppercase font-bold tracking-wider text-slate-400 flex items-center gap-1.5">
                  <Feather className="h-3.5 w-3.5 text-cyan-400" /> Journal Entry
                </span>
                <p className="text-sm text-slate-300 italic bg-slate-950/40 p-4 rounded-xl border border-slate-800 leading-relaxed">
                  “I felt overwhelmed tackling the migration today, but once I broke it into
                  individual steps, things clicked into place. Grateful for the calm that followed.”
                </p>
                <div className="flex gap-2">
                  <span className="text-xs px-2.5 py-0.5 rounded-full bg-emerald-500/10 text-emerald-400 border border-emerald-500/20">
                    Gratitude
                  </span>
                  <span className="text-xs px-2.5 py-0.5 rounded-full bg-indigo-500/10 text-indigo-400 border border-indigo-500/20">
                    Perseverance
                  </span>
                </div>
              </div>

              <div className="space-y-3">
                <span className="text-xs uppercase font-bold tracking-wider text-slate-400 flex items-center gap-1.5">
                  <Sparkles className="h-3.5 w-3.5 text-indigo-400" /> AI Reflection
                </span>
                <p className="text-sm text-slate-200 bg-slate-900/60 p-4 rounded-xl border border-cyan-500/20 leading-relaxed">
                  “You recognized the power of incremental progress when overwhelm took over.
                  Acknowledging your relief builds resilience for future challenges.”
                </p>
                <span className="inline-block text-[11px] text-slate-400">
                  Mood: 8.0/10 • Calibrating emotional equilibrium
                </span>
              </div>
            </div>
          </div>
        </div>
      </section>

      {/* Feature Grid */}
      <section className="py-20 px-4 sm:px-6 lg:px-8 max-w-7xl mx-auto w-full">
        <div className="text-center max-w-2xl mx-auto mb-16">
          <h2 className="text-2xl sm:text-3xl font-bold tracking-tight text-white">
            Designed for Mindfulness and Clarity
          </h2>
          <p className="mt-3 text-sm sm:text-base text-slate-400">
            A seamless bridge between freeform writing and deep emotional awareness.
          </p>
        </div>

        <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-6">
          {features.map((feat, idx) => {
            const Icon = feat.icon;
            return (
              <Card key={idx} hover className={`p-6 border ${feat.border}`}>
                <div className={`p-3 rounded-xl bg-slate-800/80 w-fit mb-4 ${feat.color}`}>
                  <Icon className="h-6 w-6" />
                </div>
                <h3 className="text-base font-semibold text-white">{feat.title}</h3>
                <p className="mt-2 text-xs sm:text-sm text-slate-400 leading-relaxed">
                  {feat.description}
                </p>
              </Card>
            );
          })}
        </div>
      </section>

      {/* Privacy Guarantee Banner */}
      <section className="py-16 px-4 sm:px-6 lg:px-8 max-w-5xl mx-auto w-full">
        <div className="rounded-2xl border border-slate-800 bg-gradient-to-r from-slate-900/80 via-[#0e1424] to-slate-900/80 p-8 sm:p-10 text-center space-y-4">
          <div className="mx-auto flex h-12 w-12 items-center justify-center rounded-xl bg-cyan-500/10 text-cyan-400 border border-cyan-500/20">
            <Lock className="h-6 w-6" />
          </div>
          <h3 className="text-xl font-bold text-white">Zero Tracking. Total Agency.</h3>
          <p className="text-sm text-slate-400 max-w-xl mx-auto leading-relaxed">
            Your journals are encrypted and tied strictly to your account. Your thoughts
            are never sold, exposed to ad networks, or broadcast publicly.
          </p>
        </div>
      </section>

      {/* Footer */}
      <footer className="mt-auto border-t border-slate-800/80 py-8 px-4 text-center text-xs text-slate-400">
        <p>© 2026 ConsciousAI Journal. Understand your thoughts. Reflect with intention.</p>
      </footer>
    </div>
  );
};
