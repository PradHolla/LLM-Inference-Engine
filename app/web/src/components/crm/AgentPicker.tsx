import type { AgentInfo, CrmAgent } from "../../lib/crm";

/** Segmented control to choose which agent answers; the description shows as a tooltip. */
export function AgentPicker({ agents, value, disabled, onChange }: {
  agents: AgentInfo[]; value: CrmAgent; disabled: boolean; onChange: (agent: CrmAgent) => void;
}) {
  return <div className="crm-segments crm-agent-picker" role="radiogroup" aria-label="Agent">
    {agents.map((agent) => <button key={agent.id} type="button" role="radio" aria-checked={agent.id === value}
      disabled={disabled} title={agent.description || undefined}
      onClick={() => { if (agent.id !== value) onChange(agent.id); }}>{agent.label}</button>)}
  </div>;
}
