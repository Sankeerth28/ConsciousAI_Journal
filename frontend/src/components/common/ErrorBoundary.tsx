import { Component } from 'react';
import type { ErrorInfo, ReactNode } from 'react';
import { AlertTriangle, RefreshCw } from 'lucide-react';
import { Button } from '../ui/Button';

interface Props {
  children: ReactNode;
}

interface State {
  hasError: boolean;
  error?: Error;
}

export class ErrorBoundary extends Component<Props, State> {
  public state: State = {
    hasError: false,
  };

  public static getDerivedStateFromError(error: Error): State {
    return { hasError: true, error };
  }

  public componentDidCatch(error: Error, errorInfo: ErrorInfo) {
    console.error('Uncaught component error in ConsciousAI:', error, errorInfo);
  }

  private handleReload = () => {
    this.setState({ hasError: false });
    window.location.reload();
  };

  public render() {
    if (this.state.hasError) {
      return (
        <div className="min-h-screen flex items-center justify-center p-6 bg-[#090d16] text-slate-100">
          <div className="max-w-md w-full rounded-2xl bg-slate-900/90 border border-slate-800 p-8 text-center shadow-2xl">
            <div className="mx-auto flex h-12 w-12 items-center justify-center rounded-xl bg-rose-500/10 text-rose-400 border border-rose-500/20 mb-4">
              <AlertTriangle className="h-6 w-6" />
            </div>
            <h2 className="text-xl font-bold">Something went wrong</h2>
            <p className="mt-2 text-sm text-slate-400">
              An unexpected interface error occurred. You can safely reload the page to continue.
            </p>
            {this.state.error?.message && (
              <pre className="mt-4 p-3 rounded-lg bg-black/40 text-left text-xs font-mono text-rose-300 overflow-x-auto">
                {this.state.error.message}
              </pre>
            )}
            <div className="mt-6 flex justify-center">
              <Button onClick={this.handleReload} icon={<RefreshCw className="h-4 w-4" />}>
                Reload Application
              </Button>
            </div>
          </div>
        </div>
      );
    }

    return this.props.children;
  }
}
