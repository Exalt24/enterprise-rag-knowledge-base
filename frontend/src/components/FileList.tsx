"use client";

import { useEffect, useState } from "react";
import { api } from "@/lib/api";

interface Document {
  file_name: string;
  file_type: string;
  file_size_kb: number;
  upload_date: string;
  chunk_count: number;
}

export function FileList({ refreshKey }: { refreshKey: number }) {
  const [documents, setDocuments] = useState<Document[]>([]);
  const [loading, setLoading] = useState(true);
  const [deleting, setDeleting] = useState<string | null>(null);

  useEffect(() => {
    fetchDocuments();
  }, [refreshKey]);

  const fetchDocuments = async () => {
    try {
      const data = await api.listDocuments();
      setDocuments(data.documents);
    } catch (error) {
      console.error("Failed to fetch documents:", error);
    } finally {
      setLoading(false);
    }
  };

  const handleDelete = async (fileName: string) => {
    if (!confirm(`Delete ${fileName}? This will remove it from the knowledge base.`)) {
      return;
    }

    setDeleting(fileName);

    try {
      await api.deleteDocument(fileName);
      await fetchDocuments(); // Refresh list
    } catch (error) {
      alert(`Failed to delete: ${error instanceof Error ? error.message : "Unknown error"}`);
    } finally {
      setDeleting(null);
    }
  };

  if (loading) {
    return (
      <div className="bg-slate-800/50 rounded-lg p-6 border border-slate-700">
        <h3 className="text-lg font-semibold text-white mb-4">Documents</h3>
        <p className="text-slate-400 text-sm">Loading...</p>
      </div>
    );
  }

  return (
    <div className="bg-slate-800/50 rounded-lg p-6 border border-slate-700">
      <h3 className="text-lg font-semibold text-white mb-4">
        Documents ({documents.length})
      </h3>

      {documents.length === 0 ? (
        <p className="text-slate-400 text-sm">No documents uploaded yet</p>
      ) : (
        <div className="space-y-2 max-h-[300px] overflow-y-auto">
          {documents.map((doc) => (
            <div
              key={doc.file_name}
              className="bg-slate-700/30 rounded-lg p-3 border border-slate-600 hover:border-slate-500 transition-colors"
            >
              <div className="flex justify-between items-start">
                <div className="flex-1 min-w-0">
                  <p className="text-white font-medium truncate">{doc.file_name}</p>
                  <div className="flex gap-4 text-xs text-slate-400 mt-1">
                    <span>{doc.file_type.toUpperCase()}</span>
                    <span>{doc.file_size_kb} KB</span>
                    <span>{doc.chunk_count} chunks</span>
                  </div>
                  <p className="text-xs text-slate-500 mt-1">
                    {new Date(doc.upload_date).toLocaleDateString()}
                  </p>
                </div>

                <button
                  onClick={() => handleDelete(doc.file_name)}
                  disabled={deleting === doc.file_name}
                  className="ml-3 text-red-400 hover:text-red-300 disabled:opacity-50 text-sm px-2 py-1 rounded hover:bg-red-900/20 transition-colors"
                >
                  {deleting === doc.file_name ? "..." : "Delete"}
                </button>
              </div>
            </div>
          ))}
        </div>
      )}
    </div>
  );
}
