import { ChatWindow } from "@/components/chat/chat-window";

export default function Home() {
  return (
    <div className="mx-auto flex h-dvh w-full max-w-2xl flex-col">
      <header className="border-b px-4 py-3">
        <h1 className="text-sm font-semibold">GraphRAG Chat</h1>
        <p className="text-xs text-muted-foreground">
          Hybrid graph + vector retrieval over 2WikiMultiHopQA
        </p>
      </header>
      <ChatWindow />
    </div>
  );
}
