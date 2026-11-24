import { ListChecks } from "lucide-react";
import { humanize, type CrmExample } from "../../lib/crm";
import { Tooltip, TooltipContent, TooltipTrigger } from "../ui/tooltip";

/** Group by job when the examples carry one (service agent), otherwise by skill. */
function group(examples: CrmExample[]): [string, string, CrmExample[]][] {
  const groups = new Map<string, [string, CrmExample[]]>();
  for (const example of examples) {
    const key = example.job ?? example.skill;
    const entry = groups.get(key) ?? [example.job ?? humanize(example.skill), []];
    entry[1].push(example);
    groups.set(key, entry);
  }
  return [...groups.entries()].map(([key, [label, items]]) => [key, label, items]);
}

/** Showcase questions grouped by job or skill; clicking one runs it. */
export function ExampleCards({ examples, disabled, onRun }: {
  examples: CrmExample[]; disabled: boolean; onRun: (example: CrmExample) => void;
}) {
  return <div className="crm-examples">
    {group(examples).map(([key, label, items]) => <section key={key} className="crm-example-group">
      <h3>{label}</h3>
      <div className="crm-example-grid">
        {items.map((example) => <button key={example.id} type="button" className="crm-example" disabled={disabled}
          title={example.context_summary || undefined} onClick={() => onRun(example)}>
          <span className="crm-example-meta">
            {!example.job && <span className="crm-tag">{humanize(example.task_type)}</span>}
            {example.mode === "customer" && <span className="crm-tag soft">Customer</span>}
            {example.has_expected && <Tooltip>
              <TooltipTrigger><span className="crm-graded" aria-label="Checked against an expected answer">
                <ListChecks size={13} aria-hidden="true" /></span></TooltipTrigger>
              <TooltipContent>Checked against an expected answer</TooltipContent>
            </Tooltip>}
          </span>
          <strong>{example.title}</strong>
          <span className="crm-example-q">{example.question}</span>
        </button>)}
      </div>
    </section>)}
  </div>;
}
