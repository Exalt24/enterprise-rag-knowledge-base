"use client";

import { useState, useRef, useEffect } from "react";
import { api, type QueryResponse, type Source } from "@/lib/api";

interface Message {
  type: "user" | "assistant";
  content: string;
  sources?: Source[];
  model?: string;
}

export function ChatInterface() {
  const [messages, setMessages] = useState<Message[]>([]);
  const [input, setInput] = useState("");
  const [loading, setLoading] = useState(false);
  const [useHybrid, setUseHybrid] = useState(true);
  const [useReranking, setUseReranking] = useState(false);
  const messagesEndRef = useRef<HTMLDivElement>(null);

  const scrollToBottom = () => {
    messagesEndRef.current?.scrollIntoView({ behavior: "smooth" });
  };

  useEffect(() => {
    scrollToBottom();
  }, [messages]);

  const handleSubmit = async (e: React.FormEvent) => {
    e.preventDefault();

    if (!input.trim() || loading) return;

    const userMessage = input.trim();
    setInput("");

    // Add user message
    setMessages(prev => [...prev, { type: "user", content: userMessage }]);
    setLoading(true);

    try {
      const response = await api.query({
        question: userMessage,
        k: 3,
        include_sources: true,
        use_hybrid_search: useHybrid,
        optimize_query: false,
        use_reranking: useReranking,
      });

      // Add assistant message
      setMessages(prev => [
        ...prev,
        {
          type: "assistant",
          content: response.answer,
          sources: response.sources,
          model: response.model_used,
        },
      ]);
    } catch (error) {
      // Add error message
      setMessages(prev => [
        ...prev,
        {
          type: "assistant",
          content: `Error: ${error instanceof Error ? error.message : "Failed to get response"}`,
        },
      ]);
    } finally {
      setLoading(false);
    }
  };

  return (
    <div className="bg-slate-800/50 rounded-lg border border-slate-700 flex flex-col h-[600px]">
      {/* Header */}
      <div className="p-4 border-b border-slate-700">
        <h2 className="text-xl font-semibold text-white mb-3">Ask Questions</h2>

        {/* Options */}
        <div className="flex gap-4 text-sm">
          <label className="flex items-center gap-2 cursor-pointer">
            <input
              type="checkbox"
              checked={useHybrid}
              onChange={(e) => setUseHybrid(e.target.checked)}
              className="rounded"
            />
            <span className="text-slate-300">Hybrid Search</span>
          </label>

          <label className="flex items-center gap-2 cursor-pointer">
            <input
              type="checkbox"
              checked={useReranking}
              onChange={(e) => setUseReranking(e.target.checked)}
              className="rounded"
            />
            <span className="text-slate-300">Reranking</span>
          </label>
        </div>
      </div>

      {/* Messages */}
      <div className="flex-1 overflow-y-auto p-4 space-y-4">
        {messages.length === 0 && (
          <div className="text-center text-slate-500 mt-8">
            <p className="text-lg mb-2">No messages yet</p>
            <p className="text-sm">Ask a question about your documents!</p>
          </div>
        )}

        {messages.map((message, index) => (
          <div key={index}>
            {message.type === "user" ? (
              <div className="flex justify-end">
                <div className="bg-blue-600 text-white rounded-lg px-4 py-2 max-w-[80%]">
                  {message.content}
                </div>
              </div>
            ) : (
              <div className="space-y-2">
                <div className="bg-slate-700/50 text-slate-100 rounded-lg px-4 py-3 max-w-[90%]">
                  {message.content}
                </div>

                {/* Sources */}
                {message.sources && message.sources.length > 0 && (
                  <div className="ml-4 space-y-1">
                    <p className="text-xs text-slate-500 uppercase tracking-wide">
                      Sources ({message.sources.length})
                    </p>
                    {message.sources.map((source, idx) => (
                      <div
                        key={idx}
                        className="text-xs bg-slate-800/70 rounded px-3 py-2 text-slate-300 border border-slate-700"
                      >
                        <div className="flex justify-between items-start">
                          <span className="font-medium">{source.file_name}</span>
                          {source.relevance_score !== undefined && (
                            <span className="text-slate-500 ml-2">
                              {source.relevance_score.toFixed(3)}
                            </span>
                          )}
                        </div>
                        {source.page && (
                          <p className="text-slate-500 mt-1">Page {source.page}</p>
                        )}
                      </div>
                    ))}
                  </div>
                )}

                {/* Model info */}
                {message.model && (
                  <p className="text-xs text-slate-600 ml-4">Model: {message.model}</p>
                )}
              </div>
            )}
          </div>
        ))}

        {loading && (
          <div className="flex items-center gap-2 text-slate-400">
            <div className="animate-spin rounded-full h-4 w-4 border-b-2 border-slate-400"></div>
            <span className="text-sm">Thinking...</span>
          </div>
        )}

        <div ref={messagesEndRef} />
      </div>

      {/* Input */}
      <form onSubmit={handleSubmit} className="p-4 border-t border-slate-700">
        <div className="flex gap-2">
          <input
            type="text"
            value={input}
            onChange={(e) => setInput(e.target.value)}
            placeholder="Ask a question about your documents..."
            className="flex-1 bg-slate-900/50 text-white rounded-lg px-4 py-2 border border-slate-700 focus:outline-none focus:ring-2 focus:ring-blue-500"
            disabled={loading}
          />
          <button
            type="submit"
            disabled={loading || !input.trim()}
            className="bg-blue-600 text-white px-6 py-2 rounded-lg hover:bg-blue-700 disabled:opacity-50 disabled:cursor-not-allowed transition-colors font-medium"
          >
            {loading ? "..." : "Ask"}
          </button>
        </div>
      </form>
    </div>
  );
}
