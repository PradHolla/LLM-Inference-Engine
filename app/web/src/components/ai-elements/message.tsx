import { memo } from "react";
import { Streamdown } from "streamdown";
import { codeHighlighter } from "../../lib/shiki";
import type { Theme } from "../../types";

export const MessageResponse = memo(function MessageResponse({
  children, isAnimating = false, theme,
}: { children: string; isAnimating?: boolean; theme: Theme }) {
  const shiki = theme === "dark" ? "github-dark" : "github-light";
  return <Streamdown className="markdown-body" plugins={{ code: codeHighlighter }} isAnimating={isAnimating}
    controls={{ code: { copy: true } }} skipHtml shikiTheme={[shiki, shiki]} components={{ img: () => null }}>
    {children}
  </Streamdown>;
}, (previous, next) => previous.children === next.children &&
  previous.isAnimating === next.isAnimating && previous.theme === next.theme);

MessageResponse.displayName = "MessageResponse";
