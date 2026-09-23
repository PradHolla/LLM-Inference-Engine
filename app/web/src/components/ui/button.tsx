import * as React from "react";
import { cva, type VariantProps } from "class-variance-authority";
import { cn } from "../../lib/utils";

const buttonStyles = cva(
  "inline-flex shrink-0 items-center justify-center gap-2 whitespace-nowrap font-medium transition-colors focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-[var(--focus)] disabled:pointer-events-none disabled:opacity-55",
  {
    variants: {
      variant: {
        default: "bg-[var(--accent-strong)] text-[var(--button-text)] hover:bg-[var(--accent)]",
        outline: "border border-[var(--line)] bg-transparent hover:bg-[var(--surface-hover)]",
        ghost: "bg-transparent hover:bg-[var(--surface-hover)]",
        quiet: "bg-[var(--surface-muted)] text-[var(--text)] hover:bg-[var(--surface-hover)]",
        danger: "bg-[var(--danger)] text-[var(--button-text)] hover:brightness-95",
      },
      size: {
        default: "h-10 rounded-[9px] px-4 text-sm",
        sm: "h-8 rounded-[8px] px-3 text-xs",
        icon: "size-9 rounded-[8px] p-0",
        iconSm: "size-8 rounded-[7px] p-0",
      },
    },
    defaultVariants: { variant: "outline", size: "default" },
  },
);

export interface ButtonProps extends React.ButtonHTMLAttributes<HTMLButtonElement>,
  VariantProps<typeof buttonStyles> {}

export function Button({ className, variant, size, type = "button", ...props }: ButtonProps) {
  return <button className={cn(buttonStyles({ variant, size }), className)} type={type} {...props} />;
}
