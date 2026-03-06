import { useEffect, useRef, useState } from 'react'
import { Link, useParams } from 'react-router-dom'

interface TraceSummary {
  trace_id: string
  started_at: string
  intent: string
  message_text: string
  sender: string
  step_counts: Record<string, number>
  has_error: boolean
}

export default function TraceList() {
  const { dbName } = useParams<{ dbName: string }>()
  const [traces, setTraces] = useState<TraceSummary[]>([])
  const [total, setTotal] = useState(0)
  const [page, setPage] = useState(0)
  const [loading, setLoading] = useState(true)
  const perPage = 30
  const loadingRef = useRef(true)

  useEffect(() => {
    loadingRef.current = true
    fetch(`/api/traces/${dbName}?page=${page}&per_page=${perPage}`)
      .then(r => r.json())
      .then(data => {
        setTraces(data.traces || [])
        setTotal(data.total || 0)
      })
      .catch(console.error)
      .finally(() => {
        loadingRef.current = false
        setLoading(false)
      })
  }, [dbName, page])

  const fmtTime = (iso: string) => {
    try {
      return new Date(iso).toLocaleString()
    } catch {
      return iso
    }
  }

  const stepBadges = (counts: Record<string, number>) =>
    Object.entries(counts).map(([type, count]) => (
      <span key={type} className={`badge badge-${type === 'llm_call' ? 'llm' : type === 'tool_exec' ? 'tool' : type === 'git_commit' ? 'git' : 'intent'}`}>
        {type.replace('_', ' ')} ({count})
      </span>
    ))

  return (
    <div>
      <div className="nav">
        <Link to="/">Databases</Link>
        <span>/</span>
        <strong>{dbName}</strong>
        <span style={{ marginLeft: 'auto', display: 'flex', gap: '1rem' }}>
          <Link to={`/${dbName}/timeline`}>Timeline</Link>
          <Link to={`/${dbName}/people`}>People</Link>
          <Link to={`/${dbName}/digests`}>Digests</Link>
          <Link to={`/${dbName}/integrity`}>Integrity</Link>
        </span>
      </div>
      <h1>Traces</h1>

      {loading ? (
        <div className="loading">Loading...</div>
      ) : traces.length === 0 ? (
        <p>No traces yet.</p>
      ) : (
        <>
          <table>
            <thead>
              <tr>
                <th>Time</th>
                <th>Intent</th>
                <th>Message</th>
                <th>Steps</th>
                <th>Status</th>
              </tr>
            </thead>
            <tbody>
              {traces.map(t => (
                <tr key={t.trace_id}>
                  <td style={{ whiteSpace: 'nowrap' }}>{fmtTime(t.started_at)}</td>
                  <td><span className="badge badge-intent">{t.intent || '...'}</span></td>
                  <td>
                    <Link to={`/${dbName}/${t.trace_id}`}>
                      {t.message_text || '(empty)'}
                    </Link>
                  </td>
                  <td style={{ display: 'flex', gap: '0.25rem', flexWrap: 'wrap' }}>
                    {stepBadges(t.step_counts)}
                  </td>
                  <td>
                    {t.has_error && <span className="badge badge-error">error</span>}
                  </td>
                </tr>
              ))}
            </tbody>
          </table>
          <div className="pagination">
            <button disabled={page === 0} onClick={() => setPage(p => p - 1)}>Prev</button>
            <span>Page {page + 1} of {Math.ceil(total / perPage)}</span>
            <button disabled={(page + 1) * perPage >= total} onClick={() => setPage(p => p + 1)}>Next</button>
          </div>
        </>
      )}
    </div>
  )
}
