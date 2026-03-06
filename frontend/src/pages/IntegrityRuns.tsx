import { useEffect, useState, useCallback } from 'react'
import { Link, useParams, useNavigate } from 'react-router-dom'

interface IntegrityRunSummary {
  run_id: string
  started_at: string
  issues_found: number
  auto_fixed: number
  questions_created: number
  has_error: boolean
}

interface DryRunResult {
  run_id: string
  started_at: string
  finished_at: string | null
  issues_found: number
  findings: {
    category: string
    severity: string
    message: string
    action: string
    details: Record<string, string>
    explanation: string | null
  }[]
  error: string | null
  trace_id: string | null
}

const LLM_CATEGORIES = new Set(['semantic_duplicate', 'contradiction'])

export default function IntegrityRuns() {
  const { dbName } = useParams<{ dbName: string }>()
  const navigate = useNavigate()
  const [runs, setRuns] = useState<IntegrityRunSummary[]>([])
  const [total, setTotal] = useState(0)
  const [page, setPage] = useState(0)
  const [loading, setLoading] = useState(true)
  const [running, setRunning] = useState(false)
  const [dryResult, setDryResult] = useState<DryRunResult | null>(null)

  const perPage = 20

  const loadRuns = useCallback(() => {
    setLoading(true)
    const params = new URLSearchParams({
      page: String(page),
      per_page: String(perPage),
    })

    fetch(`/api/integrity/${dbName}?${params}`)
      .then(r => r.json())
      .then(data => {
        setRuns(data.runs || [])
        setTotal(data.total || 0)
      })
      .catch(console.error)
      .finally(() => setLoading(false))
  }, [dbName, page])

  useEffect(() => { loadRuns() }, [loadRuns])

  const triggerRun = async (dryRun: boolean) => {
    setRunning(true)
    setDryResult(null)
    try {
      const qs = dryRun ? '?dry_run=true' : ''
      const resp = await fetch(`/api/integrity/${dbName}/run${qs}`, { method: 'POST' })
      const data = await resp.json()
      if (dryRun && data.record) {
        setDryResult(data.record)
      } else if (!dryRun && data.record) {
        // Navigate to the new run's detail page
        navigate(`/${dbName}/integrity/${data.record.run_id}`)
      }
    } catch (e) {
      console.error(e)
    } finally {
      setRunning(false)
      if (!dryRun) loadRuns()
    }
  }

  const formatDate = (iso: string) => {
    const d = new Date(iso)
    return d.toLocaleDateString(undefined, {
      weekday: 'short',
      year: 'numeric',
      month: 'short',
      day: 'numeric',
      hour: '2-digit',
      minute: '2-digit',
    })
  }

  const actionColor: Record<string, string> = {
    would_auto_fix: 'badge-git',
    would_create_question: 'badge-llm',
    logged: 'badge-tool',
  }

  return (
    <div>
      <div className="nav">
        <Link to="/">Databases</Link>
        <span>/</span>
        <Link to={`/${dbName}`}>{dbName}</Link>
        <span>/</span>
        <strong>Integrity</strong>
      </div>

      <div style={{ display: 'flex', alignItems: 'center', gap: '1rem', marginBottom: '1rem' }}>
        <h1 style={{ margin: 0 }}>Integrity Checks</h1>
        <button disabled={running} onClick={() => triggerRun(true)}>
          {running ? 'Running...' : 'Dry Run'}
        </button>
        <button disabled={running} onClick={() => triggerRun(false)}>
          {running ? 'Running...' : 'Run Now'}
        </button>
      </div>

      {dryResult && (() => {
        const ruleFindings = dryResult.findings.filter(f => !LLM_CATEGORIES.has(f.category))
        const llmFindings = dryResult.findings.filter(f => LLM_CATEGORIES.has(f.category))
        return (
          <div className="card" style={{ marginBottom: '1rem' }}>
            <h2 style={{ marginTop: 0 }}>Dry Run Result</h2>
            <p>
              <strong>Issues:</strong> {dryResult.issues_found}
              {llmFindings.length > 0 && (
                <span style={{ marginLeft: '0.5rem' }}>
                  (<strong>{llmFindings.length} LLM</strong>)
                </span>
              )}
              {dryResult.error && (
                <span style={{ color: 'var(--red)', marginLeft: '1rem' }}>
                  Error: {dryResult.error}
                </span>
              )}
            </p>
            {dryResult.trace_id && (
              <p>
                <Link to={`/${dbName}/${dryResult.trace_id}`}>View LLM trace</Link>
              </p>
            )}
            {dryResult.findings.length === 0 ? (
              <p>No issues found.</p>
            ) : (
              <>
                {ruleFindings.length > 0 && (
                  <div style={{ marginBottom: '0.5rem' }}>
                    <strong>Rule-based ({ruleFindings.length})</strong>
                    <div style={{ display: 'flex', flexDirection: 'column', gap: '0.4rem' }}>
                      {ruleFindings.map((f, i) => (
                        <div key={i} style={{ padding: '0.4rem 0', borderBottom: '1px solid var(--border)' }}>
                          <span className="badge badge-intent">{f.category}</span>{' '}
                          <span className={`badge ${f.severity === 'error' ? 'badge-error' : 'badge-tool'}`}>
                            {f.severity}
                          </span>{' '}
                          <span className={`badge ${actionColor[f.action] || 'badge-tool'}`}>
                            {f.action.replace(/_/g, ' ')}
                          </span>{' '}
                          <span>{f.message}</span>
                          {f.explanation && (
                            <div style={{ fontStyle: 'italic', opacity: 0.8, marginTop: '0.2rem' }}>
                              {f.explanation}
                            </div>
                          )}
                        </div>
                      ))}
                    </div>
                  </div>
                )}
                {llmFindings.length > 0 && (
                  <div style={{ marginBottom: '0.5rem' }}>
                    <strong>LLM findings ({llmFindings.length})</strong>
                    <div style={{ display: 'flex', flexDirection: 'column', gap: '0.4rem' }}>
                      {llmFindings.map((f, i) => (
                        <div key={i} style={{ padding: '0.4rem 0', borderBottom: '1px solid var(--border)' }}>
                          <span className="badge badge-llm">{f.category.replace(/_/g, ' ')}</span>{' '}
                          <span className={`badge ${f.severity === 'error' ? 'badge-error' : 'badge-tool'}`}>
                            {f.severity}
                          </span>{' '}
                          <span className={`badge ${actionColor[f.action] || 'badge-tool'}`}>
                            {f.action.replace(/_/g, ' ')}
                          </span>{' '}
                          <span>{f.message}</span>
                          {f.explanation && (
                            <div style={{ fontStyle: 'italic', opacity: 0.8, marginTop: '0.2rem' }}>
                              {f.explanation}
                            </div>
                          )}
                        </div>
                      ))}
                    </div>
                  </div>
                )}
              </>
            )}
            <button style={{ marginTop: '0.5rem' }} onClick={() => setDryResult(null)}>
              Dismiss
            </button>
          </div>
        )
      })()}

      {loading ? (
        <div className="loading">Loading...</div>
      ) : runs.length === 0 ? (
        <p>No integrity checks run yet.</p>
      ) : (
        <>
          <table>
            <thead>
              <tr>
                <th>Time</th>
                <th>Issues Found</th>
                <th>Auto-fixed</th>
                <th>Questions</th>
                <th>Status</th>
              </tr>
            </thead>
            <tbody>
              {runs.map(r => (
                <tr key={r.run_id}>
                  <td style={{ whiteSpace: 'nowrap' }}>
                    <Link to={`/${dbName}/integrity/${r.run_id}`}>
                      {formatDate(r.started_at)}
                    </Link>
                  </td>
                  <td>{r.issues_found}</td>
                  <td>{r.auto_fixed}</td>
                  <td>{r.questions_created}</td>
                  <td>
                    {r.has_error ? (
                      <span className="badge badge-error">error</span>
                    ) : (
                      <span className="badge badge-intent">ok</span>
                    )}
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
