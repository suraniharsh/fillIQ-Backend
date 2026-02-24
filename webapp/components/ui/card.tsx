import * as React from "react";

import { cn } from "@/lib/utils";

export function Card({ className, ...props }: React.HTMLAttributes<HTMLDivElement>) {
  return (
    <div
      className={cn(
        "rounded-3xl border border-white/10 bg-gradient-to-br from-white/10 via-white/05 to-white/0 text-white",
        className
      )}
      {...props}
    />
  );
}
