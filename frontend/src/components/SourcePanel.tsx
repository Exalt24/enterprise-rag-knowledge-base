"use client";

import { useEffect, useRef, useState } from "react";
import { api, type ChunkPassage, type ChunkView } from "@/lib/api";

interface SourcePanelProps {
  docId: string;
  chunkId: number;
  fileName: string;
  onClose: () => void;
}

// A side panel that shows one cited passage where it sits in its document: the cited chunk highlighted, with the
// chunks before and after it. It follows the pattern NotebookLM uses for a citation: the click opens the source at the
// passage, the passage is marked, Escape closes it. It is a labelled region rather than an aside because it renders
// inside the page's main landmark, and a complementary landmark must not sit inside another landmark. Focus moves to
// the Close button on open and goes back to the citation that was clicked on close.
//
// The parent keys this component on the citation, so every citation starts in the loading state and the effect below
// only sets state once a request has answered.
export function SourcePanel({ docId, chunkId, fileName, onClose }: SourcePanelProps) {
  // COPY-OK: type and initial value in code, not screen text
  const [view, setView] = useState<ChunkView | null>(null);
  // COPY-OK: type and initial value in code, not screen text
  const [error, setError] = useState<string | null>(null);
  // COPY-OK: initial value of a ref in code, not screen text
  const closeButton = useRef<HTMLButtonElement>(null);

  useEffect(() => {
    let cancelled = false;
    api
      .getChunk(docId, chunkId, 2)
      .then((data) => {
        if (!cancelled) setView(data);
      })
      .catch((e) => {
        if (!cancelled) setError(e instanceof Error ? e.message : "Could not load the passage.");
      });
    return () => {
      cancelled = true;
    };
  }, [docId, chunkId]);

  useEffect(() => {
    // COPY-OK: a focus lookup in code, not screen text
    const opener = document.activeElement instanceof HTMLElement ? document.activeElement : null;
    closeButton.current?.focus();
    return () => opener?.focus();
  }, []);

  useEffect(() => {
    const onKey = (event: KeyboardEvent) => {
      if (event.key === "Escape") onClose();
    };
    window.addEventListener("keydown", onKey);
    return () => window.removeEventListener("keydown", onKey);
  }, [onClose]);

  const passage = (item: ChunkPassage, testId: string) => (
    <p
      key={item.chunk_id}
      data-testid={testId}
      className="text-sm text-slate-400 whitespace-pre-wrap border-l-2 border-slate-700 pl-3"
    >
      {item.text}
    </p>
  );

  return (
    <section
      aria-label="Source passage"
      data-testid="source-panel"
      className="fixed inset-y-0 right-0 z-50 w-full max-w-md bg-slate-900 border-l border-slate-700 shadow-2xl flex flex-col"
    >
      <header className="p-4 border-b border-slate-700 flex items-start justify-between gap-3">
        <div className="min-w-0">
          <h3 className="text-white font-semibold break-words" data-testid="source-file">
            {view?.file_name ?? fileName}
          </h3>
          {view && (
            <p className="text-xs text-slate-400 mt-1" data-testid="source-location">
              {view.chunk.location.label}
              {view.total_chunks ? `, chunk ${view.chunk.chunk_id + 1} of ${view.total_chunks}` : ""}
            </p>
          )}
        </div>
        <button
          ref={closeButton}
          onClick={onClose}
          aria-label="Close source panel"
          className="text-slate-400 hover:text-white px-2 py-1 rounded hover:bg-slate-700"
        >
          Close
        </button>
      </header>

      <div className="flex-1 overflow-y-auto p-4 space-y-3">
        {!view && !error && <p className="text-sm text-slate-400">Loading the passage...</p>}
        {error && <p className="text-sm text-red-400">{error}</p>}
        {view && (
          <>
            {view.before.length === 0 && <p className="text-xs text-slate-600">Start of the document</p>}
            {view.before.map((item) => passage(item, "context-before"))}
            <mark
              data-testid="cited-passage"
              className="block text-sm text-slate-900 bg-amber-200 rounded px-3 py-2 whitespace-pre-wrap"
            >
              {view.chunk.text}
            </mark>
            {view.after.map((item) => passage(item, "context-after"))}
            {view.after.length === 0 && <p className="text-xs text-slate-600">End of the document</p>}
          </>
        )}
      </div>
    </section>
  );
}
