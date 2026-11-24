import { formatCell, isSalesforceId } from "../../lib/crm";

function Cell({ value }: { value: unknown }) {
  const text = formatCell(value);
  if (value === null || value === undefined) return <span className="crm-null">null</span>;
  if (isSalesforceId(value)) return <span className="crm-id" title={text}>{text}</span>;
  return <span title={text.length > 40 ? text : undefined}>{text}</span>;
}

/** A compact scrollable table of preview rows; one wide row becomes a field list instead. */
export function ResultTable({ columns, rows }: { columns: string[]; rows: Record<string, unknown>[] }) {
  const names = columns.length ? columns : rows[0] ? Object.keys(rows[0]) : [];
  if (!rows.length || !names.length) return null;
  if (rows.length === 1 && names.length > 5) {
    return <dl className="crm-record" tabIndex={0} aria-label="Record fields">
      {names.map((name) => <div key={name}>
        <dt>{name}</dt><dd><Cell value={rows[0][name]} /></dd>
      </div>)}
    </dl>;
  }
  return <div className="crm-table-wrap" tabIndex={0} role="region" aria-label="Result rows">
    <table className="crm-table">
      <thead><tr>{names.map((name) => <th key={name} scope="col" className={typeof rows[0][name] === "number" ? "num" : undefined}>
        {name}</th>)}</tr></thead>
      <tbody>{rows.map((row, index) => <tr key={index}>
        {names.map((name) => <td key={name} className={typeof row[name] === "number" ? "num" : undefined}>
          <Cell value={row[name]} />
        </td>)}
      </tr>)}</tbody>
    </table>
  </div>;
}
