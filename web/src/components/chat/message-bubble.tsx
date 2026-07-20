"use client";

import { ChevronDown, AlertCircle } from "lucide-react";
import { Avatar, AvatarFallback } from "@/components/ui/avatar";
import {
  Collapsible,
  CollapsibleContent,
  CollapsibleTrigger,
} from "@/components/ui/collapsible";
import { cn } from "@/lib/utils";
import type { ChatMessage } from "@/lib/types";

export function MessageBubble({ message }: { message: ChatMessage }) {
  const isUser = message.role === "user";
  const hasReasoning = !!message.reasoningPath?.length;

  return (
    <div className={cn("flex gap-3", isUser && "flex-row-reverse")}>
      <Avatar size="sm" className="mt-0.5 shrink-0">
        <AvatarFallback>{isUser ? "You" : "GR"}</AvatarFallback>
      </Avatar>

      <div className={cn("flex max-w-[80%] flex-col gap-1", isUser && "items-end")}>
        <div
          className={cn(
            "rounded-2xl px-4 py-2.5 text-sm whitespace-pre-wrap",
            isUser
              ? "bg-primary text-primary-foreground"
              : message.isError
                ? "bg-destructive/10 text-destructive"
                : "bg-muted text-foreground",
          )}
        >
          {message.isError && (
            <AlertCircle className="mb-1 inline size-4 align-text-bottom" />
          )}{" "}
          {message.content}
        </div>

        {hasReasoning && (
          <Collapsible className="w-full">
            <CollapsibleTrigger className="group flex items-center gap-1 px-1 text-xs text-muted-foreground hover:text-foreground">
              <ChevronDown className="size-3 transition-transform group-data-[panel-open]:rotate-180" />
              Show reasoning ({message.reasoningPath!.length} passages)
            </CollapsibleTrigger>
            <CollapsibleContent className="mt-1 space-y-1 rounded-lg border bg-card p-2 text-xs text-muted-foreground">
              {message.reasoningPath!.map((step, i) => {
                const passageId = message.supportingPassageIds?.find((id) =>
                  step.includes(id),
                );
                return (
                  <div
                    key={i}
                    className={cn(
                      "rounded px-1.5 py-1",
                      passageId && "bg-primary/5 text-foreground",
                    )}
                  >
                    {step}
                  </div>
                );
              })}
            </CollapsibleContent>
          </Collapsible>
        )}
      </div>
    </div>
  );
}
