"use client";

import { useState } from "react";
import { DocumentUpload } from "@/components/DocumentUpload";
import { ChatInterface } from "@/components/ChatInterface";
import { Stats } from "@/components/Stats";

export default function Home() {
  const [refreshKey, setRefreshKey] = useState(0);

  const handleDocumentUploaded = () => {
    setRefreshKey(prev => prev + 1);
  };

  return (
    <main className="min-h-screen bg-gradient-to-br from-slate-900 via-slate-800 to-slate-900">
      <div className="container mx-auto px-4 py-8 max-w-7xl">
        <div className="mb-8 text-center">
          <h1 className="text-4xl font-bold text-white mb-2">
            Enterprise RAG Knowledge Base
          </h1>
          <p className="text-slate-400">
            Production RAG with Llama 3, Chroma & Advanced Retrieval
          </p>
        </div>

        <Stats key={refreshKey} />

        <div className="grid md:grid-cols-3 gap-6 mt-8">
          <div className="md:col-span-1">
            <DocumentUpload onUploadSuccess={handleDocumentUploaded} />
          </div>

          <div className="md:col-span-2">
            <ChatInterface />
          </div>
        </div>

        <div className="mt-12 text-center text-slate-500 text-sm">
          <p>Next.js • FastAPI • LangChain • Llama 3 • Chroma</p>
          <p className="mt-1">100% Free & Open Source</p>
        </div>
      </div>
    </main>
  );
}
