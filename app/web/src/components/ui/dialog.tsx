import * as DialogPrimitive from "@radix-ui/react-dialog";
import type { ComponentProps } from "react";
import { X } from "lucide-react";
import { Button } from "./button";

export const Dialog = DialogPrimitive.Root;
export const DialogTrigger = DialogPrimitive.Trigger;

export function DialogContent({ className, children, ...props }: ComponentProps<typeof DialogPrimitive.Content>) {
  // a caller's class is added to dialog-content, never in place of it: without it the box sits under the overlay
  return <DialogPrimitive.Portal>
    <DialogPrimitive.Overlay className="dialog-overlay" />
    <DialogPrimitive.Content className={className ? `dialog-content ${className}` : "dialog-content"} {...props}>
      {children}
      <DialogPrimitive.Close asChild>
        <Button className="dialog-close" variant="ghost" size="iconSm" aria-label="Close dialog">
          <X size={16} />
        </Button>
      </DialogPrimitive.Close>
    </DialogPrimitive.Content>
  </DialogPrimitive.Portal>;
}
