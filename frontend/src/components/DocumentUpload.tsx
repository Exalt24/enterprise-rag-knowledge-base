"use client";

import { useState, useRef } from "react";
import { api } from "@/lib/api";

interface DocumentUploadProps {
  onUploadSuccess?: () => void;
}

export function DocumentUpload({ onUploadSuccess }: DocumentUploadProps) {
  const [uploading, setUploading] = useState(false);
  const [message, setMessage] = useState("");
  const fileInputRef = useRef<HTMLInputElement>(null);

  const handleFileUpload = async (e: React.ChangeEvent<HTMLInputElement>) => {
    const file = e.target.files?.[0];
    if (!file) return;

    // Validate file type
    const allowedTypes = [".pdf", ".docx", ".txt", ".md"];
    const fileExt = file.name.substring(file.name.lastIndexOf(".")).toLowerCase();

    if (!allowedTypes.includes(fileExt)) {
      setMessage(`Unsupported file type. Allowed: ${allowedTypes.join(", ")}`);
      return;
    }

    setUploading(true);
    setMessage("");

    try {
      const result = await api.uploadDocument(file);

      if (result.success) {
        setMessage(`✓ ${file.name} uploaded successfully!`);
        onUploadSuccess?.();

        // Reset file input
        if (fileInputRef.current) {
          fileInputRef.current.value = "";
        }
      } else {
        setMessage(`✗ Upload failed: ${result.message}`);
      }
    } catch (error) {
      setMessage(`✗ Error: ${error instanceof Error ? error.message : "Upload failed"}`);
    } finally {
      setUploading(false);
    }
  };

  return (
    <div className="bg-slate-800/50 rounded-lg p-6 border border-slate-700">
      <h2 className="text-xl font-semibold text-white mb-4">Upload Documents</h2>

      <div className="space-y-4">
        <div
          className="border-2 border-dashed border-slate-600 rounded-lg p-8 text-center hover:border-slate-500 transition-colors cursor-pointer"
          onClick={() => fileInputRef.current?.click()}
        >
          <input
            ref={fileInputRef}
            type="file"
            accept=".pdf,.docx,.txt,.md"
            onChange={handleFileUpload}
            className="hidden"
            disabled={uploading}
          />

          {uploading ? (
            <div className="space-y-2">
              <div className="animate-spin rounded-full h-8 w-8 border-b-2 border-white mx-auto"></div>
              <p className="text-slate-400">Processing...</p>
            </div>
          ) : (
            <div className="space-y-2">
              <svg
                className="mx-auto h-12 w-12 text-slate-500"
                fill="none"
                viewBox="0 0 24 24"
                stroke="currentColor"
              >
                <path
                  strokeLinecap="round"
                  strokeLinejoin="round"
                  strokeWidth={2}
                  d="M7 16a4 4 0 01-.88-7.903A5 5 0 1115.9 6L16 6a5 5 0 011 9.9M15 13l-3-3m0 0l-3 3m3-3v12"
                />
              </svg>
              <p className="text-slate-300">Click to upload document</p>
              <p className="text-slate-500 text-sm">PDF, DOCX, TXT, MD</p>
            </div>
          )}
        </div>

        {message && (
          <div
            className={`p-3 rounded-lg text-sm ${
              message.startsWith("✓")
                ? "bg-green-900/30 text-green-400 border border-green-800"
                : "bg-red-900/30 text-red-400 border border-red-800"
            }`}
          >
            {message}
          </div>
        )}

        <div className="text-xs text-slate-500 space-y-1">
          <p>• Supported formats: PDF, DOCX, TXT, Markdown</p>
          <p>• Files are chunked and embedded automatically</p>
          <p>• Searchable within seconds</p>
        </div>
      </div>
    </div>
  );
}
