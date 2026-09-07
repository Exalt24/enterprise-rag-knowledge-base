"use client";

import { useState } from "react";
import { DocumentUpload } from "@/components/DocumentUpload";
import { ChatInterface } from "@/components/ChatInterface";
import { Stats } from "@/components/Stats";
import { FileList } from "@/components/FileList";

export default function Home() {
  const [refreshKey, setRefreshKey] = useState(0);

  const handleDocumentUploaded = () => {
    setRefreshKey((prev) => prev + 1);
  };

  return (
    <main className="min-h-screen bg-linear-to-br from-slate-900 via-slate-800 to-slate-900">
      <div className="container mx-auto px-4 py-8 max-w-7xl">
        <div className="mb-8 text-center">
          <h1 className="text-4xl font-bold text-white mb-2">
            Enterprise RAG Knowledge Base
          </h1>
          {/* No model name here, deliberately.
              This read "Production RAG with Llama 3.3" while the stats bar three
              inches below it reported the real model, so the page contradicted
              itself on screen. Llama 3.3 had been retired by the provider and the
              backend had already moved on; a name hardcoded into marketing copy
              cannot track a model that changes, so the copy names the ARCHITECTURE
              and the Stats component reports whatever is actually answering. */}
          <p className="text-slate-400">
            Hybrid retrieval over Qdrant, with cited answers
          </p>
        </div>

        <Stats key={refreshKey} />

        {/* min-w-0 on both columns is what stops the page scrolling sideways on a phone.
            A grid item defaults to min-width:auto, so it refuses to shrink below the
            widest thing inside it and pushes past its track. Measured at 390px: the
            document panel and the chat panel each rendered 385px wide against 358px of
            available space, so the document scrolled 11px horizontally. Nothing looked
            broken, which is why it survived: you only see it by dragging the page. */}
        <div className="grid md:grid-cols-3 gap-6 mt-8">
          <div className="md:col-span-1 space-y-6 min-w-0">
            <DocumentUpload onUploadSuccess={handleDocumentUploaded} />
            <FileList refreshKey={refreshKey} />
          </div>

          <div className="md:col-span-2 min-w-0">
            <ChatInterface />
          </div>
        </div>

        <div className="mt-12 text-center text-slate-500 text-sm">
          <p>Next.js • FastAPI • LangChain • Qdrant</p>
          <p className="mt-1">100% Free & Open Source</p>
        </div>
      </div>
    </main>
  );
}
