import { ListChecks } from "lucide-react";
import { humanize, type CrmExample } from "../../lib/crm";
import { Tooltip, TooltipContent, TooltipTrigger } from "../ui/tooltip";

function groupBySkill(examples: CrmExample[]): [string, CrmExample[]][] {
  const groups = new Map<string, CrmExample[]>();
  for (const example of examples) groups.set(example.skill, [...(groups.get(example.skill) ?? []), example]);
  return [...groups.entries()];
}

/** Showcase questions grouped by skill; clicking one runs it. */
export function ExampleCards({ examples, disabled, onRun }: {
  examples: CrmExample[]; disabled: boolean; onRun: (example: CrmExample) => void;
}) {
  return <div className="crm-examples">
    {groupBySkill(examples).map(([skill, items]) => <section key={skill} className="crm-example-group">
      <h3>{humanize(skill)}</h3>
      <div className="crm-example-grid">
        {items.map((example) => <button key={example.id} type="button" className="crm-example" disabled={disabled}
          title={example.context_summary || undefined} onClick={() => onRun(example)}>
          <span className="crm-example-meta">
            <span className="crm-tag">{humanize(example.task_type)}</span>
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
