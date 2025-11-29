"use client";

import { useEffect, useState } from "react";
import { api, type StatsResponse } from "@/lib/api";

export function Stats() {
  const [stats, setStats] = useState<StatsResponse | null>(null);
  const [loading, setLoading] = useState(true);

  useEffect(() => {
    const fetchStats = async () => {
      try {
        const data = await api.getStats();
        setStats(data);
      } catch (error) {
        console.error("Failed to fetch stats:", error);
      } finally {
        setLoading(false);
      }
    };

    fetchStats();
  }, []);

  if (loading) {
    return (
      <div className="bg-slate-800/50 rounded-lg p-6 border border-slate-700">
        <p className="text-slate-400">Loading stats...</p>
      </div>
    );
  }

  if (!stats) {
    return (
      <div className="bg-slate-800/50 rounded-lg p-6 border border-red-900/50">
        <p className="text-red-400">Failed to load stats. Is the API running?</p>
      </div>
    );
  }

  return (
    <div className="bg-slate-800/50 rounded-lg p-6 border border-slate-700">
      <div className="grid grid-cols-2 md:grid-cols-5 gap-4">
        <StatItem label="Documents" value={stats.total_documents.toString()} />
        <StatItem label="LLM" value={stats.llm_model} />
        <StatItem label="Embeddings" value={stats.embedding_model.split("-").slice(0, 2).join("-")} />
        <StatItem label="Dimension" value={stats.embedding_dimension.toString()} />
        <StatItem label="Vector DB" value="Chroma" />
      </div>
    </div>
  );
}

function StatItem({ label, value }: { label: string; value: string }) {
  return (
    <div className="text-center">
      <p className="text-slate-500 text-xs uppercase tracking-wide mb-1">{label}</p>
      <p className="text-white font-semibold">{value}</p>
    </div>
  );
}
