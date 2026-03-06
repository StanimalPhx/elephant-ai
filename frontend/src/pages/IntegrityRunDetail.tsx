import { useEffect, useState } from 'react'
import { Link, useParams } from 'react-router-dom'
import StepCard from '../components/StepCard'

interface Finding {
  category: string
  severity: string
  message: string
  action: string
  details: Record<string, string>
  explanation: string | null
}

interface TraceStep {
  step_type: string
  timestamp: string
  [key: string]: unknown
}

interface IntegrityRun {
  run_id: string
  started_at: string
  finished_at: string | null
  issues_found: number
  auto_fixed: number
  questions_created: number
  findings: Finding[]
  error: string | null
  trace_id: string | null
  trace_steps?: TraceStep[]
}

const actionColor: Record<string, string> = {
  auto_fixed: 'badge-git',
  question_created: 'badge-llm',
  logged: 'badge-tool',
  would_auto_fix: 'badge-git',
  would_create_question: 'badge-llm',
}

const LLM_CATEGORIES = new Set(['semantic_duplicate', 'contradiction'])

export default function IntegrityRunDetail() {
  const { dbName, runId } = useParams<{ dbName: string; runId: string }>()
  const [run, setRun] = useState<IntegrityRun | null>(null)
  const [loading, setLoading] = useState(true)

  useEffect(() => {
    setLoading(true)
    fetch(`/api/integrity/${dbName}/${runId}`)
      .then(r => r.json())
      .then(data => setRun(data))
      .catch(console.error)
      .finally(() => setLoading(false))
  }, [dbName, runId])

  const formatDate = (iso: string) => {
    const d = new Date(iso)
    return d.toLocaleDateString(undefined, {
      weekday: 'short',
      year: 'numeric',
      month: 'short',
      day: 'numeric',
      hour: '2-digit',
      minute: '2-digit',
      second: '2-digit',
    })
  }

  const duration = (start: string, end: string | null) => {
    if (!end) return '...'
    const ms = new Date(end).getTime() - new Date(start).getTime()
    if (ms < 1000) return `${ms}ms`
    return `${(ms / 1000).toFixed(1)}s`
  }

  if (loading) return <div className="loading">Loading...</div>
  if (!run) return <p>Run not found.</p>

  const ruleFindings = run.findings.filter(f => !LLM_CATEGORIES.has(f.category))
  const llmFindings = run.findings.filter(f => LLM_CATEGORIES.has(f.category))
  const llmTraceSteps = (run.trace_steps || []).filter(s => s.step_type === 'llm_call')

  return (
    <div>
      <div className="nav">
        <Link to="/">Databases</Link>
        <span>/</span>
        <Link to={`/${dbName}`}>{dbName}</Link>
        <span>/</span>
        <Link to={`/${dbName}/integrity`}>Integrity</Link>
        <span>/</span>
        <strong>{run.run_id.slice(0, 20)}...</strong>
      </div>

      <h1>Integrity Run</h1>
      <div className="card" style={{ marginBottom: '1rem' }}>
        <p><strong>Run ID:</strong> {run.run_id}</p>
        <p><strong>Started:</strong> {formatDate(run.started_at)}</p>
        <p><strong>Duration:</strong> {duration(run.started_at, run.finished_at)}</p>
        <p>
          <strong>Issues:</strong> {run.issues_found}
          {' | '}
          <strong>Auto-fixed:</strong> {run.auto_fixed}
          {' | '}
          <strong>Questions:</strong> {run.questions_created}
          {llmFindings.length > 0 && (
            <>
              {' | '}
              <strong>LLM findings:</strong> {llmFindings.length}
            </>
          )}
        </p>
        {run.trace_id && (
          <p>
            <strong>Trace:</strong>{' '}
            <Link to={`/${dbName}/${run.trace_id}`}>View full trace</Link>
          </p>
        )}
        {run.error && (
          <p style={{ color: 'var(--red)' }}><strong>Error:</strong> {run.error}</p>
        )}
      </div>

      {ruleFindings.length > 0 && (
        <>
          <h2>Rule-based findings ({ruleFindings.length})</h2>
          <div style={{ display: 'flex', flexDirection: 'column', gap: '0.5rem' }}>
            {ruleFindings.map((f, i) => (
              <div key={i} className="card">
                <div style={{ display: 'flex', gap: '0.5rem', alignItems: 'center', marginBottom: '0.25rem' }}>
                  <span className="badge badge-intent">{f.category}</span>
                  <span className={`badge ${f.severity === 'error' ? 'badge-error' : 'badge-tool'}`}>
                    {f.severity}
                  </span>
                  <span className={`badge ${actionColor[f.action] || 'badge-tool'}`}>
                    {f.action.replace(/_/g, ' ')}
                  </span>
                </div>
                <p style={{ margin: '0.25rem 0' }}>{f.message}</p>
                {f.explanation && (
                  <p style={{ margin: '0.25rem 0', fontStyle: 'italic', opacity: 0.8 }}>
                    {f.explanation}
                  </p>
                )}
                {Object.keys(f.details).length > 0 && (
                  <pre style={{ fontSize: '0.8rem', margin: '0.25rem 0', opacity: 0.7 }}>
                    {JSON.stringify(f.details, null, 2)}
                  </pre>
                )}
              </div>
            ))}
          </div>
        </>
      )}

      {llmFindings.length > 0 && (
        <>
          <h2>LLM findings ({llmFindings.length})</h2>
          <div style={{ display: 'flex', flexDirection: 'column', gap: '0.5rem' }}>
            {llmFindings.map((f, i) => (
              <div key={i} className="card">
                <div style={{ display: 'flex', gap: '0.5rem', alignItems: 'center', marginBottom: '0.25rem' }}>
                  <span className="badge badge-llm">{f.category.replace(/_/g, ' ')}</span>
                  <span className={`badge ${f.severity === 'error' ? 'badge-error' : 'badge-tool'}`}>
                    {f.severity}
                  </span>
                  <span className={`badge ${actionColor[f.action] || 'badge-tool'}`}>
                    {f.action.replace(/_/g, ' ')}
                  </span>
                </div>
                <p style={{ margin: '0.25rem 0' }}>{f.message}</p>
                {f.explanation && (
                  <p style={{ margin: '0.25rem 0', fontStyle: 'italic', opacity: 0.8 }}>
                    {f.explanation}
                  </p>
                )}
                {Object.keys(f.details).length > 0 && (
                  <pre style={{ fontSize: '0.8rem', margin: '0.25rem 0', opacity: 0.7 }}>
                    {JSON.stringify(f.details, null, 2)}
                  </pre>
                )}
              </div>
            ))}
          </div>
        </>
      )}

      {llmTraceSteps.length > 0 && (
        <>
          <h2>LLM Analysis ({llmTraceSteps.length} calls)</h2>
          <div style={{ display: 'flex', flexDirection: 'column', gap: '0.5rem' }}>
            {llmTraceSteps.map((step, i) => (
              <StepCard key={i} step={step} index={i} dbName={dbName} />
            ))}
          </div>
        </>
      )}

      {run.findings.length === 0 && (
        <p>No issues found.</p>
      )}
    </div>
  )
}
