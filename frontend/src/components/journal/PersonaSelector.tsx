import React from 'react';
import { HeartHandshake, Compass, Brain, Sparkles } from 'lucide-react';
import type { PersonaEnum } from '../../api/types';

export interface PersonaSelectorProps {
  value: PersonaEnum;
  onChange: (persona: PersonaEnum) => void;
  disabled?: boolean;
}

export const PersonaSelector: React.FC<PersonaSelectorProps> = ({
  value,
  onChange,
  disabled = false,
}) => {
  const personas: Array<{
    id: PersonaEnum;
    title: string;
    description: string;
    icon: React.ElementType;
    color: string;
  }> = [
    {
      id: 'Supportive',
      title: 'Supportive',
      description: 'Warm, empathetic, validating companion focused on emotional well-being.',
      icon: HeartHandshake,
      color: 'text-rose-400',
    },
    {
      id: 'Coach',
      title: 'Coach',
      description: 'Action-driven, motivating, encouraging personal growth and accountability.',
      icon: Compass,
      color: 'text-cyan-400',
    },
    {
      id: 'Therapist-Style Reflection',
      title: 'Therapist-Style Reflection',
      description: 'Introspective, deep questioning to unlock subconscious patterns.',
      icon: Brain,
      color: 'text-violet-400',
    },
    {
      id: 'Neutral',
      title: 'Neutral',
      description: 'Objective, grounded, concise reflection without prescriptive advice.',
      icon: Sparkles,
      color: 'text-emerald-400',
    },
  ];

  return (
    <div className="space-y-2">
      <label className="block text-xs font-semibold uppercase tracking-wider text-slate-300">
        AI Reflection Persona
      </label>
      <div className="grid grid-cols-1 sm:grid-cols-2 gap-3">
        {personas.map((p) => {
          const isSelected = value === p.id;
          const Icon = p.icon;
          return (
            <button
              key={p.id}
              type="button"
              disabled={disabled}
              onClick={() => onChange(p.id)}
              className={`flex items-start gap-3 p-3.5 rounded-xl border text-left transition-all duration-200 select-none ${
                isSelected
                  ? 'bg-cyan-500/10 border-cyan-500/50 shadow-glow-cyan'
                  : 'bg-slate-900/40 border-slate-800/80 hover:bg-slate-800/50 hover:border-slate-700'
              }`}
            >
              <div
                className={`p-2 rounded-lg shrink-0 ${
                  isSelected ? 'bg-cyan-500/20 text-cyan-300' : 'bg-slate-800 text-slate-400'
                }`}
              >
                <Icon className={`h-4 w-4 ${p.color}`} />
              </div>
              <div>
                <span className="block text-xs font-bold text-slate-100">
                  {p.title}
                </span>
                <span className="block text-[11px] text-slate-400 mt-0.5 leading-relaxed">
                  {p.description}
                </span>
              </div>
            </button>
          );
        })}
      </div>
    </div>
  );
};
