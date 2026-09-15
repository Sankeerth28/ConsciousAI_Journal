import React, { useEffect, useState, useMemo } from 'react';
import { NavLink } from 'react-router-dom';
import {
  Search,
  Download,
  BookOpen,
  ArrowUpDown,
  PenSquare,
} from 'lucide-react';
import { listJournals, exportJournals } from '../api/journals';
import type { JournalEntryRead, JournalExportFormat } from '../api/types';
import { Button } from '../components/ui/Button';
import { Card } from '../components/ui/Card';
import { Skeleton } from '../components/ui/Skeleton';
import { EmptyState } from '../components/common/EmptyState';
import { JournalCard } from '../components/journal/JournalCard';

export const JournalHistoryPage: React.FC = () => {
  const [entries, setEntries] = useState<JournalEntryRead[]>([]);
  const [total, setTotal] = useState(0);
  const [isLoading, setIsLoading] = useState(true);
  const [isExporting, setIsExporting] = useState(false);

  // Filters
  const [searchQuery, setSearchQuery] = useState('');
  const [selectedTag, setSelectedTag] = useState<string>('all');
  const [privacyFilter, setPrivacyFilter] = useState<'all' | 'private' | 'public'>('all');
  const [sortOrder, setSortOrder] = useState<'newest' | 'oldest'>('newest');

  // Pagination
  const [page, setPage] = useState(1);
  const pageSize = 20;

  useEffect(() => {
    let isMounted = true;
    const fetchEntries = async () => {
      try {
        setIsLoading(true);
        const data = await listJournals({
          skip: (page - 1) * pageSize,
          limit: pageSize,
        });
        if (isMounted) {
          setEntries(data.items || []);
          setTotal(data.total || 0);
        }
      } catch {
        // Error handling
      } finally {
        if (isMounted) {
          setIsLoading(false);
        }
      }
    };

    fetchEntries();
    return () => {
      isMounted = false;
    };
  }, [page]);

  // Extract all unique tags
  const allTags = useMemo(() => {
    const tagSet = new Set<string>();
    entries.forEach((e) => e.tags?.forEach((t) => tagSet.add(t)));
    return Array.from(tagSet);
  }, [entries]);

  // Client-side search and filtering
  const filteredEntries = useMemo(() => {
    return entries
      .filter((e) => {
        // Text search
        if (searchQuery.trim()) {
          const q = searchQuery.toLowerCase();
          const matchesText = e.text.toLowerCase().includes(q);
          const matchesEmotion = e.top_emotion?.toLowerCase().includes(q);
          const matchesValue = e.top_value?.toLowerCase().includes(q);
          const matchesTag = e.tags?.some((t) => t.toLowerCase().includes(q));
          if (!matchesText && !matchesEmotion && !matchesValue && !matchesTag) {
            return false;
          }
        }

        // Tag filter
        if (selectedTag !== 'all' && !e.tags?.includes(selectedTag)) {
          return false;
        }

        // Privacy filter
        if (privacyFilter === 'private' && !e.is_private) return false;
        if (privacyFilter === 'public' && e.is_private) return false;

        return true;
      })
      .sort((a, b) => {
        const timeA = new Date(a.created_at).getTime();
        const timeB = new Date(b.created_at).getTime();
        return sortOrder === 'newest' ? timeB - timeA : timeA - timeB;
      });
  }, [entries, searchQuery, selectedTag, privacyFilter, sortOrder]);

  // Handle CSV / JSON export from backend endpoint
  const handleExport = async (format: JournalExportFormat) => {
    try {
      setIsExporting(true);
      const blob = await exportJournals(format);
      const url = window.URL.createObjectURL(blob);
      const a = document.createElement('a');
      a.href = url;
      a.download = `consciousai_journals_${new Date().toISOString().slice(0, 10)}.${format}`;
      document.body.appendChild(a);
      a.click();
      window.URL.revokeObjectURL(url);
      document.body.removeChild(a);
    } catch {
      alert('Unable to export entries at this time. Please try again.');
    } finally {
      setIsExporting(false);
    }
  };

  return (
    <div className="space-y-6 animate-fadeIn">
      {/* Header with Title and Export Actions */}
      <div className="flex flex-col sm:flex-row sm:items-center justify-between gap-4 pb-4 border-b border-slate-800/80">
        <div>
          <h1 className="text-2xl font-bold text-white tracking-tight">Journal History</h1>
          <p className="text-xs text-slate-400 mt-0.5">
            Search, filter, and review all your past reflections
          </p>
        </div>

        <div className="flex items-center gap-2">
          <Button
            size="sm"
            variant="outline"
            disabled={isExporting || entries.length === 0}
            onClick={() => handleExport('csv')}
            icon={<Download className="h-3.5 w-3.5" />}
          >
            Export CSV
          </Button>
          <Button
            size="sm"
            variant="outline"
            disabled={isExporting || entries.length === 0}
            onClick={() => handleExport('json')}
            icon={<Download className="h-3.5 w-3.5" />}
          >
            Export JSON
          </Button>
          <NavLink to="/journal/new">
            <Button size="sm" variant="primary" icon={<PenSquare className="h-3.5 w-3.5" />}>
              New Entry
            </Button>
          </NavLink>
        </div>
      </div>

      {/* Filter and Search Bar */}
      <Card className="p-4 sm:p-5">
        <div className="flex flex-col md:flex-row gap-3">
          {/* Search input */}
          <div className="relative flex-1">
            <Search className="absolute left-3.5 top-1/2 -translate-y-1/2 h-4 w-4 text-slate-400" />
            <input
              type="text"
              placeholder="Search by keywords, emotions, or values..."
              value={searchQuery}
              onChange={(e) => setSearchQuery(e.target.value)}
              className="w-full pl-10 pr-4 py-2 rounded-xl bg-slate-900/80 border border-slate-800 text-xs sm:text-sm text-slate-100 placeholder:text-slate-500 focus:outline-none focus:ring-2 focus:ring-cyan-500/30 focus:border-cyan-500"
            />
          </div>

          {/* Filter dropdowns */}
          <div className="flex flex-wrap items-center gap-2">
            {/* Tag filter */}
            {allTags.length > 0 && (
              <select
                value={selectedTag}
                onChange={(e) => setSelectedTag(e.target.value)}
                className="text-xs rounded-xl bg-slate-900/80 border border-slate-800 text-slate-300 px-3 py-2 focus:outline-none focus:ring-2 focus:ring-cyan-500/30"
              >
                <option value="all">All Tags</option>
                {allTags.map((t) => (
                  <option key={t} value={t}>
                    Tag: {t}
                  </option>
                ))}
              </select>
            )}

            {/* Privacy filter */}
            <select
              value={privacyFilter}
              onChange={(e) => setPrivacyFilter(e.target.value as any)}
              className="text-xs rounded-xl bg-slate-900/80 border border-slate-800 text-slate-300 px-3 py-2 focus:outline-none focus:ring-2 focus:ring-cyan-500/30"
            >
              <option value="all">All Visibility</option>
              <option value="public">Standard</option>
              <option value="private">Private Only</option>
            </select>

            {/* Sort order */}
            <button
              onClick={() => setSortOrder(sortOrder === 'newest' ? 'oldest' : 'newest')}
              className="flex items-center gap-1.5 text-xs px-3 py-2 rounded-xl bg-slate-900/80 border border-slate-800 text-slate-300 hover:text-white transition-colors"
            >
              <ArrowUpDown className="h-3.5 w-3.5 text-cyan-400" />
              <span>{sortOrder === 'newest' ? 'Newest First' : 'Oldest First'}</span>
            </button>
          </div>
        </div>
      </Card>

      {/* Journal Cards Grid */}
      {isLoading ? (
        <div className="space-y-4">
          <Skeleton className="h-36 w-full" />
          <Skeleton className="h-36 w-full" />
          <Skeleton className="h-36 w-full" />
        </div>
      ) : filteredEntries.length === 0 ? (
        <EmptyState
          icon={<BookOpen className="h-7 w-7" />}
          title={searchQuery ? 'No Matching Entries' : 'No Journals Found'}
          description={
            searchQuery
              ? 'Try adjusting your search terms or clearing your filters.'
              : 'You have not created any journal reflections yet.'
          }
          actionLabel={searchQuery ? 'Clear Filters' : 'Write First Entry'}
          onAction={() => {
            if (searchQuery) {
              setSearchQuery('');
              setSelectedTag('all');
              setPrivacyFilter('all');
            } else {
              window.location.assign('/journal/new');
            }
          }}
        />
      ) : (
        <div className="grid grid-cols-1 md:grid-cols-2 gap-4 sm:gap-6">
          {filteredEntries.map((entry) => (
            <JournalCard key={entry.id} entry={entry} />
          ))}
        </div>
      )}

      {/* Pagination Controls */}
      {total > pageSize && (
        <div className="flex items-center justify-between pt-6 border-t border-slate-800">
          <span className="text-xs text-slate-400">
            Showing {(page - 1) * pageSize + 1}–{Math.min(page * pageSize, total)} of {total} entries
          </span>
          <div className="flex items-center gap-2">
            <Button
              size="sm"
              variant="outline"
              disabled={page === 1}
              onClick={() => setPage(page - 1)}
            >
              Previous
            </Button>
            <Button
              size="sm"
              variant="outline"
              disabled={page * pageSize >= total}
              onClick={() => setPage(page + 1)}
            >
              Next
            </Button>
          </div>
        </div>
      )}
    </div>
  );
};
