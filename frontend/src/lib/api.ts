/**
 * API Service for Enterprise RAG Backend
 */

const API_BASE_URL = process.env.NEXT_PUBLIC_API_URL || "http://localhost:8001/api";

export interface QueryRequest {
  question: string;
  k?: number;
  include_sources?: boolean;
  use_hybrid_search?: boolean;
  optimize_query?: boolean;
  use_reranking?: boolean;
  conversation_id?: string;  // For multi-turn conversations with memory
}

export interface Source {
  file_name: string;
  page?: number;
  content_preview: string;
  relevance_score?: number;
}

export interface QueryResponse {
  answer: string;
  query: string;
  sources: Source[];
  num_sources: number;
  model_used: string;
}

export interface StatsResponse {
  total_documents: number;
  collection_name: string;
  embedding_model: string;
  embedding_dimension: number;
  llm_model: string;
}

export interface HealthResponse {
  status: string;
  ollama_connected: boolean;
  vector_db_connected: boolean;
  total_documents: number;
}

export const api = {
  async query(request: QueryRequest): Promise<QueryResponse> {
    const response = await fetch(`${API_BASE_URL}/query`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(request),
    });

    if (!response.ok) {
      throw new Error(`Query failed: ${response.statusText}`);
    }

    return response.json();
  },

  async uploadDocument(file: File): Promise<{ success: boolean; message: string }> {
    const formData = new FormData();
    formData.append("file", file);

    const response = await fetch(`${API_BASE_URL}/ingest`, {
      method: "POST",
      body: formData,
    });

    if (!response.ok) {
      throw new Error(`Upload failed: ${response.statusText}`);
    }

    return response.json();
  },

  async getStats(): Promise<StatsResponse> {
    const response = await fetch(`${API_BASE_URL}/stats`);

    if (!response.ok) {
      throw new Error(`Stats fetch failed: ${response.statusText}`);
    }

    return response.json();
  },

  async getHealth(): Promise<HealthResponse> {
    const response = await fetch(`${API_BASE_URL}/health`);

    if (!response.ok) {
      throw new Error(`Health check failed: ${response.statusText}`);
    }

    return response.json();
  },

  async listDocuments(): Promise<{
    documents: Array<{
      file_name: string;
      file_type: string;
      file_size_kb: number;
      upload_date: string;
      chunk_count: number;
    }>;
    total_documents: number;
    total_chunks: number;
  }> {
    const response = await fetch(`${API_BASE_URL}/documents`);

    if (!response.ok) {
      throw new Error(`Failed to list documents: ${response.statusText}`);
    }

    return response.json();
  },

  async deleteDocument(fileName: string): Promise<{ success: boolean; message: string }> {
    const response = await fetch(`${API_BASE_URL}/documents/${fileName}`, {
      method: "DELETE",
    });

    if (!response.ok) {
      throw new Error(`Failed to delete document: ${response.statusText}`);
    }

    return response.json();
  },
};
