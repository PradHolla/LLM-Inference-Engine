import * as TooltipPrimitive from "@radix-ui/react-tooltip";
import type { ReactNode } from "react";

export function TooltipProvider({ children }: { children: ReactNode }) {
  return <TooltipPrimitive.Provider delayDuration={350}>{children}</TooltipPrimitive.Provider>;
}

export function Tooltip({ children }: { children: ReactNode }) {
  return <TooltipPrimitive.Root>{children}</TooltipPrimitive.Root>;
}

export function TooltipTrigger({ children, asChild = true }: {
  children: ReactNode;
  asChild?: boolean;
}) {
  return <TooltipPrimitive.Trigger asChild={asChild}>{children}</TooltipPrimitive.Trigger>;
}

export function TooltipContent({ children, side = "top" }: {
  children: ReactNode;
  side?: "top" | "right" | "bottom" | "left";
}) {
  return <TooltipPrimitive.Portal>
    <TooltipPrimitive.Content className="tooltip-content" side={side} sideOffset={6}>
      {children}
    </TooltipPrimitive.Content>
  </TooltipPrimitive.Portal>;
}
