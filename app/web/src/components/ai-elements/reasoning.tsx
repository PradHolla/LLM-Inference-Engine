import * as CollapsiblePrimitive from "@radix-ui/react-collapsible";
import { useControllableState } from "@radix-ui/react-use-controllable-state";
import { ChevronDown } from "lucide-react";
import {
  createContext, memo, useCallback, useContext, useEffect, useMemo, useRef, useState,
  type ComponentProps, type ReactNode,
} from "react";
import { cn } from "../../lib/utils";

type ReasoningContextValue = {
  isStreaming: boolean;
  isOpen: boolean;
  duration?: number;
};

const ReasoningContext = createContext<ReasoningContextValue | null>(null);

export type ReasoningProps = ComponentProps<typeof CollapsiblePrimitive.Root> & {
  isStreaming?: boolean;
  open?: boolean;
  defaultOpen?: boolean;
  onOpenChange?: (open: boolean) => void;
  duration?: number;
};

export const Reasoning = memo(function Reasoning({
  className, isStreaming = false, open, defaultOpen, onOpenChange, duration: durationProp,
  children, ...props
}: ReasoningProps) {
  const [isOpen, setIsOpen] = useControllableState<boolean>({
    defaultProp: defaultOpen ?? isStreaming, onChange: onOpenChange, prop: open,
  });
  const [duration, setDuration] = useControllableState<number | undefined>({
    defaultProp: undefined, prop: durationProp,
  });
  const startedAt = useRef<number | null>(null);
  const [closedAfterStream, setClosedAfterStream] = useState(false);

  useEffect(() => {
    if (isStreaming) {
      if (startedAt.current == null) startedAt.current = Date.now();
      if (defaultOpen !== false && !isOpen) setIsOpen(true);
    } else if (startedAt.current != null) {
      setDuration(Math.ceil((Date.now() - startedAt.current) / 1000));
      startedAt.current = null;
    }
  }, [isStreaming, isOpen, defaultOpen, setDuration, setIsOpen]);

  useEffect(() => {
    if (!isStreaming && isOpen && !closedAfterStream && duration !== undefined) {
      const timer = window.setTimeout(() => {
        setIsOpen(false);
        setClosedAfterStream(true);
      }, 850);
      return () => window.clearTimeout(timer);
    }
  }, [isStreaming, isOpen, closedAfterStream, duration, setIsOpen]);

  const change = useCallback((value: boolean) => setIsOpen(value), [setIsOpen]);
  const context = useMemo(() => ({ isStreaming, isOpen, duration }),
    [isStreaming, isOpen, duration]);

  return <ReasoningContext.Provider value={context}>
    <CollapsiblePrimitive.Root className={cn("reasoning", className)} open={isOpen}
      onOpenChange={change} {...props}>
      {children}
    </CollapsiblePrimitive.Root>
  </ReasoningContext.Provider>;
});

export function ReasoningTrigger({ children, ...props }: ComponentProps<typeof CollapsiblePrimitive.Trigger>) {
  const context = useContext(ReasoningContext);
  if (!context) throw new Error("ReasoningTrigger must be inside Reasoning");
  const label: ReactNode = context.isStreaming ? "Thinking" : context.duration
    ? `Thought for ${context.duration} ${context.duration === 1 ? "second" : "seconds"}` : "Thought";
  return <CollapsiblePrimitive.Trigger className="reasoning-trigger" {...props}>
    <span className={context.isStreaming ? "thinking-pulse" : "thinking-mark"} aria-hidden="true" />
    <span>{children ?? label}</span>
    <ChevronDown className={context.isOpen ? "reasoning-chevron open" : "reasoning-chevron"} size={15} />
  </CollapsiblePrimitive.Trigger>;
}

export function ReasoningContent({ className, children, ...props }: ComponentProps<typeof CollapsiblePrimitive.Content> & {
  children: string;
}) {
  return <CollapsiblePrimitive.Content className={cn("reasoning-content", className)} {...props}>
    <div className="reasoning-copy">{children}</div>
  </CollapsiblePrimitive.Content>;
}
