import React, { useState } from 'react';
import { NavLink, useNavigate } from 'react-router-dom';
import { Eye, EyeOff, Sparkles, Mail, Lock, AlertCircle, Check } from 'lucide-react';
import { useAuth } from '../context/AuthContext';
import { formatApiError } from '../api/client';
import { Input } from '../components/ui/Input';
import { Button } from '../components/ui/Button';

export const RegisterPage: React.FC = () => {
  const { register, isAuthenticated } = useAuth();
  const navigate = useNavigate();

  const [email, setEmail] = useState('');
  const [password, setPassword] = useState('');
  const [showPassword, setShowPassword] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [isSubmitting, setIsSubmitting] = useState(false);

  React.useEffect(() => {
    if (isAuthenticated) {
      navigate('/dashboard', { replace: true });
    }
  }, [isAuthenticated, navigate]);

  // Password strength indicators
  const hasLength = password.length >= 8;
  const hasUpper = /[A-Z]/.test(password);
  const hasNumber = /[0-9]/.test(password);
  const hasSpecial = /[^A-Za-z0-9]/.test(password);

  const strengthScore = [hasLength, hasUpper, hasNumber, hasSpecial].filter(Boolean).length;

  const getStrengthLabel = () => {
    if (strengthScore <= 1) return { text: 'Weak', color: 'bg-rose-500' };
    if (strengthScore <= 3) return { text: 'Moderate', color: 'bg-amber-500' };
    return { text: 'Strong', color: 'bg-emerald-500' };
  };

  const handleSubmit = async (e: React.FormEvent) => {
    e.preventDefault();
    setError(null);

    if (!hasLength) {
      setError('Password must be at least 8 characters long.');
      return;
    }

    try {
      setIsSubmitting(true);
      await register({ email: email.trim(), password });
      navigate('/dashboard');
    } catch (err) {
      setError(formatApiError(err));
    } finally {
      setIsSubmitting(false);
    }
  };

  return (
    <div className="min-h-screen bg-[#090d16] text-slate-100 flex flex-col justify-center py-12 px-4 sm:px-6 lg:px-8 relative overflow-hidden">
      <div className="absolute top-1/3 left-1/2 -translate-x-1/2 -translate-y-1/2 w-96 h-96 bg-indigo-500/10 blur-[120px] rounded-full pointer-events-none" />

      <div className="sm:mx-auto sm:w-full sm:max-w-md relative z-10 text-center">
        <NavLink to="/" className="inline-flex items-center gap-2.5 mb-6 group">
          <div className="flex h-11 w-11 items-center justify-center rounded-2xl bg-gradient-to-tr from-cyan-500 to-indigo-600 shadow-glow-cyan">
            <Sparkles className="h-5 w-5 text-white" />
          </div>
        </NavLink>
        <h2 className="text-2xl sm:text-3xl font-bold tracking-tight text-white">
          Create your sanctuary
        </h2>
        <p className="mt-2 text-sm text-slate-400">
          Begin your journey with private AI-guided emotional self-reflection
        </p>
      </div>

      <div className="mt-8 sm:mx-auto sm:w-full sm:max-w-md relative z-10">
        <div className="rounded-2xl bg-slate-900/80 border border-slate-800 p-8 shadow-2xl backdrop-blur-xl">
          {error && (
            <div className="mb-6 p-3.5 rounded-xl bg-rose-500/10 border border-rose-500/20 text-xs text-rose-300 flex items-start gap-2.5 animate-fadeIn">
              <AlertCircle className="h-4 w-4 shrink-0 mt-0.5 text-rose-400" />
              <span>{error}</span>
            </div>
          )}

          <form onSubmit={handleSubmit} className="space-y-5">
            <Input
              label="Email Address"
              type="email"
              placeholder="you@example.com"
              autoComplete="email"
              required
              value={email}
              onChange={(e) => setEmail(e.target.value)}
              leftIcon={<Mail className="h-4 w-4" />}
            />

            <Input
              label="Password (min 8 characters)"
              type={showPassword ? 'text' : 'password'}
              placeholder="••••••••••••"
              autoComplete="new-password"
              required
              value={password}
              onChange={(e) => setPassword(e.target.value)}
              leftIcon={<Lock className="h-4 w-4" />}
              rightIcon={
                <button
                  type="button"
                  onClick={() => setShowPassword(!showPassword)}
                  className="text-slate-400 hover:text-slate-200 focus:outline-none"
                  aria-label={showPassword ? 'Hide password' : 'Show password'}
                >
                  {showPassword ? <EyeOff className="h-4 w-4" /> : <Eye className="h-4 w-4" />}
                </button>
              }
            />

            {/* Password strength bar */}
            {password.length > 0 && (
              <div className="space-y-2 pt-1 animate-fadeIn">
                <div className="flex items-center justify-between text-[11px]">
                  <span className="text-slate-400">Strength:</span>
                  <span className="font-semibold text-slate-200">{getStrengthLabel().text}</span>
                </div>
                <div className="h-1.5 w-full bg-slate-800 rounded-full overflow-hidden flex gap-1">
                  {[1, 2, 3, 4].map((step) => (
                    <div
                      key={step}
                      className={`h-full flex-1 rounded-full transition-all duration-300 ${
                        step <= strengthScore ? getStrengthLabel().color : 'bg-slate-800'
                      }`}
                    />
                  ))}
                </div>
                <div className="grid grid-cols-2 gap-1.5 text-[11px] text-slate-400 pt-1">
                  <span className={`flex items-center gap-1 ${hasLength ? 'text-emerald-400' : ''}`}>
                    <Check className="h-3 w-3" /> 8+ characters
                  </span>
                  <span className={`flex items-center gap-1 ${hasUpper ? 'text-emerald-400' : ''}`}>
                    <Check className="h-3 w-3" /> Uppercase letter
                  </span>
                  <span className={`flex items-center gap-1 ${hasNumber ? 'text-emerald-400' : ''}`}>
                    <Check className="h-3 w-3" /> Number
                  </span>
                  <span className={`flex items-center gap-1 ${hasSpecial ? 'text-emerald-400' : ''}`}>
                    <Check className="h-3 w-3" /> Special character
                  </span>
                </div>
              </div>
            )}

            <Button
              type="submit"
              variant="primary"
              size="lg"
              className="w-full mt-2"
              isLoading={isSubmitting}
            >
              Create Account
            </Button>
          </form>

          <div className="mt-6 text-center text-xs text-slate-400 border-t border-slate-800/80 pt-5">
            Already have an account?{' '}
            <NavLink
              to="/login"
              className="text-cyan-400 hover:text-cyan-300 font-semibold transition-colors"
            >
              Sign in
            </NavLink>
          </div>
        </div>
      </div>
    </div>
  );
};
