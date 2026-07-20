// Mirrors QueryRequest/QueryResponse in src/graphrag/api/main.py

export interface QueryRequest {
  question: string;
  top_k_vector?: number;
  top_k_final?: number;
}

export interface QueryResponse {
  answer: string;
  supporting_passage_ids: string[];
  reasoning_path: string[];
}

export interface ChatMessage {
  id: string;
  role: "user" | "assistant";
  content: string;
  reasoningPath?: string[];
  supportingPassageIds?: string[];
  isError?: boolean;
}
