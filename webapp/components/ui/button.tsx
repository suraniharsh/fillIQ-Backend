import * as React from "react";

import { cn } from "@/lib/utils";

type ButtonVariant = "default" | "ghost";
type ButtonSize = "default" | "sm";

interface ButtonProps extends React.ButtonHTMLAttributes<HTMLButtonElement> {
  variant?: ButtonVariant;
  size?: ButtonSize;
}

const variantClasses: Record<ButtonVariant, string> = {
  default:
    "bg-ember text-midnight shadow-[0_10px_30px_rgba(251,107,60,0.35)] hover:bg-white hover:text-midnight",
  ghost: "bg-white/5 text-white/70 shadow-none hover:bg-white/10"
};

const sizeClasses: Record<ButtonSize, string> = {
  default: "px-5 py-3 text-[0.7rem] font-semibold uppercase tracking-[0.4em]",
  sm: "px-3 py-2 text-[0.55rem] font-semibold uppercase tracking-[0.35em]"
};

export const Button = React.forwardRef<HTMLButtonElement, ButtonProps>(
  ({ className, variant = "default", size = "default", ...props }, ref) => {
    return (
      <button
        ref={ref}
        className={cn(
          "inline-flex items-center justify-center rounded-2xl border border-white/20 transition focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-offset-2 focus-visible:ring-ember/60",
          variantClasses[variant],
          sizeClasses[size],
          className
        )}
        {...props}
      />
    );
  }
);
Button.displayName = "Button";
