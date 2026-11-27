import { humanize, type CrmExample } from "../../lib/crm";

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

/** One helpful line under the title: the case description for routing, the question for policy, none for knowledge. */
function subline(example: CrmExample): string | null {
  if (example.note) return example.note;  // a planted example: what was planted, in plain words
  if (example.task_type === "knowledge_qa") return null;
  if (example.task_type !== "case_routing") return example.question;
  const detail = example.detail ?? "";
  const lead = `${example.title}.`;
  return (detail.startsWith(lead) ? detail.slice(lead.length) : detail).trim() || null;
}

/** Showcase questions grouped by job or skill; clicking one runs it. */
export function ExampleCards({ examples, disabled, onRun }: {
  examples: CrmExample[]; disabled: boolean; onRun: (example: CrmExample) => void;
}) {
  return <div className="crm-examples">
    {group(examples).map(([key, label, items]) => <section key={key} className="crm-example-group">
      <h3>{label}</h3>
      <div className="crm-example-grid">
        {items.map((example) => {
          const line = subline(example);
          return <button key={example.id} type="button" className={example.note ? "crm-example planted" : "crm-example"} disabled={disabled}
            title={example.note || example.detail || example.question} onClick={() => onRun(example)}>
            <strong>{example.title}</strong>
            {line && <span className="crm-example-q">{line}</span>}
          </button>;
        })}
      </div>
    </section>)}
  </div>;
}
