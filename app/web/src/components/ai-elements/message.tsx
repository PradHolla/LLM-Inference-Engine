import { memo, useMemo, type ComponentProps, type ReactNode } from "react";
import { Streamdown, defaultRemarkPlugins, type Components } from "streamdown";
import { remarkCitations } from "../../lib/citations";
import { codeHighlighter } from "../../lib/shiki";
import type { Source, Theme } from "../../types";

type RemarkPlugins = NonNullable<ComponentProps<typeof Streamdown>["remarkPlugins"]>;
export type CiteHandler = (index: number | null) => void;

const citationPlugins = new Map<number, RemarkPlugins>();

function pluginsFor(count: number): RemarkPlugins | undefined {
  if (count <= 0) return undefined;
  let plugins = citationPlugins.get(count);
  if (!plugins) {
    plugins = [...Object.values(defaultRemarkPlugins), [remarkCitations, { count }]];
    citationPlugins.set(count, plugins);
  }
  return plugins;
}

const noImage = () => null;
const baseComponents: Components = { img: noImage };

function citationComponents(sources: Source[], onCite?: CiteHandler): Components {
  return { img: noImage, sup: ({ children, node: _node, ...rest }: { children?: ReactNode; node?: unknown }) => {
    const n = Number(typeof children === "string" ? children : Array.isArray(children) ? children.join("") : NaN);
    const source = Number.isInteger(n) ? sources[n - 1] : undefined;
    if (!source) return <sup {...rest}>{children}</sup>;
    const name = source.title || source.site;
    return <sup className="citation">
      <a href={source.url} target="_blank" rel="noreferrer" title={name} aria-label={`Source ${n}: ${name}`}
        onMouseEnter={() => onCite?.(n)} onMouseLeave={() => onCite?.(null)}
        onFocus={() => onCite?.(n)} onBlur={() => onCite?.(null)}>{n}</a>
    </sup>;
  } } as Components;
}

export const MessageResponse = memo(function MessageResponse({
  children, isAnimating = false, theme, sources, onCite,
}: { children: string; isAnimating?: boolean; theme: Theme; sources?: Source[] | null; onCite?: CiteHandler }) {
  const shiki = theme === "dark" ? "github-dark" : "github-light";
  const count = sources?.length ?? 0;
  const components = useMemo(() => count ? citationComponents(sources!, onCite) : baseComponents,
    [count, sources, onCite]);
  return <Streamdown className="markdown-body" plugins={{ code: codeHighlighter }} isAnimating={isAnimating}
    controls={{ code: { copy: true } }} skipHtml shikiTheme={[shiki, shiki]} components={components}
    remarkPlugins={pluginsFor(count)}>
    {children}
  </Streamdown>;
}, (previous, next) => previous.children === next.children &&
  previous.isAnimating === next.isAnimating && previous.theme === next.theme &&
  previous.sources === next.sources && previous.onCite === next.onCite);

MessageResponse.displayName = "MessageResponse";
