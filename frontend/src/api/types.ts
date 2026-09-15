/**
 * Strict TypeScript types mapped 1:1 from FastAPI backend OpenAPI specification.
 */

export type PersonaEnum =
  | 'Supportive'
  | 'Coach'
  | 'Therapist-Style Reflection'
  | 'Neutral';

export type JournalExportFormat = 'csv' | 'json';

export type RegionEnum = 'US' | 'CA' | 'GB' | 'AU' | 'IN' | 'GLOBAL';

export interface UserRegister {
  email: string;
  password: string; // minLength 8, maxLength 72
}

export interface UserLogin {
  email: string;
  password: string;
}

export interface UserRead {
  id: string;
  email: string;
  is_active: boolean;
  created_at: string;
}

export interface TokenResponse {
  access_token: string;
  token_type: string;
  expires_in: number;
}

export interface JournalEntryCreate {
  text: string;
  mood_score?: number | null; // 1.0 to 10.0
  tags?: string[];
  persona?: PersonaEnum;
  region?: string;
  is_private?: boolean;
}

export interface JournalEntryUpdate {
  tags?: string[] | null;
  mood_score?: number | null; // 1.0 to 10.0
  is_private?: boolean | null;
}

export interface JournalEntryRead {
  id: number;
  user_id?: string | null;
  text: string;
  mood_score?: number | null;
  top_emotion?: string | null;
  top_value?: string | null;
  detected_emotions: string[];
  detected_values: string[];
  tags: string[];
  ai_response?: string | null;
  is_private: boolean;
  created_at: string;
  updated_at: string;
}

export interface JournalListResponse {
  items: JournalEntryRead[];
  total: number;
  skip: number;
  limit: number;
  has_more: boolean;
}

export interface ReflectionResult {
  response: string;
  persona: PersonaEnum;
  model_name: string;
  safety_flag: boolean;
  fallback_used: boolean;
}

export interface SafetyCheckResult {
  is_safe: boolean;
  reason?: string | null;
  crisis_response?: string | null;
  region?: string | null;
}

export interface OutputSafetyCheckResult {
  is_safe: boolean;
  violation_type?: string | null;
  reason?: string | null;
}

export interface JournalEntryCreateResponse {
  entry: JournalEntryRead | null;
  reflection: ReflectionResult;
  input_safety: SafetyCheckResult;
  output_safety?: OutputSafetyCheckResult | null;
}

export interface FeedbackCreate {
  feedback_type: string;
  comment?: string | null;
}

export interface FeedbackRead {
  id: number;
  journal_id: number;
  user_id: string;
  feedback_type: string;
  comment?: string | null;
  created_at: string;
}

export interface HealthResponse {
  status: string;
  service: string;
}

export interface ReadyResponse {
  status: string;
  database?: string | null;
  redis?: string | null;
}

export interface ValidationErrorItem {
  loc: (string | number)[];
  msg: string;
  type: string;
}

export interface ApiErrorResponse {
  detail?: string | ValidationErrorItem[];
}
