import React from 'react';
import { Badge } from '../ui/Badge';
import { Sparkles } from 'lucide-react';

export interface EmotionBadgeProps {
  emotion: string;
}

export const EmotionBadge: React.FC<EmotionBadgeProps> = ({ emotion }) => {
  // Map common emotions to harmonious color tokens
  const lower = emotion.toLowerCase();
  let variant: 'cyan' | 'emerald' | 'amber' | 'rose' | 'violet' | 'indigo' = 'cyan';

  if (lower.includes('joy') || lower.includes('happy') || lower.includes('gratitude') || lower.includes('peace')) {
    variant = 'emerald';
  } else if (lower.includes('anxiety') || lower.includes('fear') || lower.includes('anger') || lower.includes('sad')) {
    variant = 'rose';
  } else if (lower.includes('curious') || lower.includes('excited') || lower.includes('hope')) {
    variant = 'amber';
  } else if (lower.includes('love') || lower.includes('compassion') || lower.includes('calm')) {
    variant = 'violet';
  }

  return (
    <Badge variant={variant} size="sm">
      {emotion}
    </Badge>
  );
};

export interface ValueBadgeProps {
  value: string;
}

export const ValueBadge: React.FC<ValueBadgeProps> = ({ value }) => {
  return (
    <Badge
      variant="indigo"
      size="sm"
      icon={<Sparkles className="h-2.5 w-2.5" />}
    >
      {value}
    </Badge>
  );
};
