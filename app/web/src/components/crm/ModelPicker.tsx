import { useCallback, useEffect, useRef, useState } from "react";
import { LoaderCircle } from "lucide-react";
import { getCrmModels, setCrmModel, type CrmModels } from "../../lib/crm";

const POLL_MS = 2000;
const POLL_LIMIT = 180;

/** Shows the served model and its note; picking the other one switches the server and polls until ready. */
export function ModelPicker({ onBusyChange }: { onBusyChange?: (busy: boolean) => void }) {
  const [models, setModels] = useState<CrmModels | null>(null);
  const [target, setTarget] = useState<string | null>(null);
  const [error, setError] = useState<string | null>(null);
  const timer = useRef<number | null>(null);
  const alive = useRef(true);

  const stopPolling = () => {
    if (timer.current != null) window.clearTimeout(timer.current);
    timer.current = null;
  };

  const poll = useCallback((attempt: number) => {
    stopPolling();
    timer.current = window.setTimeout(async () => {
      try {
        const next = await getCrmModels();
        if (!alive.current) return;
        setModels(next);
        if (!next.switching) { setTarget(null); setError(null); return; }
      } catch { /* the server may be restarting; keep polling */ }
      if (!alive.current) return;
      if (attempt >= POLL_LIMIT) { setTarget(null); setError("The model switch is taking too long."); return; }
      poll(attempt + 1);
    }, POLL_MS);
  }, []);

  useEffect(() => {
    alive.current = true;
    getCrmModels().then((next) => {
      if (!alive.current) return;
      setModels(next);
      if (next.switching) poll(0);
    }).catch((reason: unknown) => {
      if (alive.current) setError(reason instanceof Error ? reason.message : "Could not load the model list.");
    });
    return () => { alive.current = false; stopPolling(); };
  }, [poll]);

  const switching = Boolean(models?.switching) || target !== null;
  useEffect(() => { onBusyChange?.(switching); }, [switching, onBusyChange]);

  const choose = async (id: string) => {
    if (!models || switching || id === models.current) return;
    setError(null);
    setTarget(id);
    try {
      const next = await setCrmModel(id);
      if (!alive.current) return;
      setModels(next);
      if (next.switching) poll(0); else setTarget(null);
    } catch (reason) {
      setTarget(null);
      setError(reason instanceof Error ? reason.message : "Could not switch the model.");
    }
  };

  if (!models) {
    return <div className="model-picker"><p className="model-note">{error ?? "Loading models"}</p></div>;
  }
  const selected = target ?? models.current;
  const shown = models.options.find((option) => option.id === selected);
  if (!models.picker) {  // single-model demo: name the model, no switch
    return <div className="model-picker"><span className="model-fixed" title={shown?.note}>{shown?.label ?? models.current}</span>
      <p className="model-note">{shown?.note ?? ""}</p></div>;
  }
  return <div className="model-picker">
    <div className="crm-segments" role="radiogroup" aria-label="Model">
      {models.options.map((option) => <button key={option.id} type="button" role="radio"
        aria-checked={option.id === selected} disabled={switching} title={option.note}
        onClick={() => void choose(option.id)}>{option.label}</button>)}
    </div>
    {switching
      ? <p className="model-note switching" role="status"><LoaderCircle className="crm-spin" size={12} aria-hidden="true" />
        Switching model, about 1-2 minutes</p>
      : <p className="model-note" title={shown?.note}>{error ?? shown?.note ?? ""}</p>}
  </div>;
}
