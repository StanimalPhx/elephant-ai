import { useEffect, useState } from 'react'
import { Link, useParams } from 'react-router-dom'

interface MemorySummary {
  id: string
  date: string
  title: string
  type: string
  people: string[]
  description: string
  nostalgia_score: number
  location: string | null
}

interface PersonOption {
  person_id: string
  display_name: string
}

const MEMORY_TYPES = [
  '', 'milestone', 'daily', 'outing', 'celebration',
  'health', 'travel', 'mundane', 'other',
]

export default function MemoryTimeline() {
  const { dbName } = useParams<{ dbName: string }>()
  const [memories, setMemories] = useState<MemorySummary[]>([])
  const [total, setTotal] = useState(0)
  const [page, setPage] = useState(0)
  const [loading, setLoading] = useState(true)
  const [people, setPeople] = useState<PersonOption[]>([])

  // Filters
  const [person, setPerson] = useState('')
  const [type, setType] = useState('')
  const [year, setYear] = useState('')

  const perPage = 50

  // Load people for the filter dropdown
  useEffect(() => {
    fetch(`/api/people/${dbName}`)
      .then(r => r.json())
      .then(data => {
        const names = (data.people || []).map((p: PersonOption) => ({
          person_id: p.person_id,
          display_name: p.display_name,
        }))
        setPeople(names)
      })
      .catch(console.error)
  }, [dbName])

  // Fetch memories whenever filters or page change
  useEffect(() => {
    setLoading(true)
    const params = new URLSearchParams({
      page: String(page),
      per_page: String(perPage),
    })
    if (person) params.set('person', person)
    if (type) params.set('type', type)
    if (year) params.set('year', year)

    fetch(`/api/memories/${dbName}?${params}`)
      .then(r => r.json())
      .then(data => {
        setMemories(data.memories || [])
        setTotal(data.total || 0)
      })
      .catch(console.error)
      .finally(() => setLoading(false))
  }, [dbName, page, person, type, year])

  // Reset page when filters change
  const updateFilter = (setter: (v: string) => void, value: string) => {
    setter(value)
    setPage(0)
  }

  // Compute year range for dropdown
  const currentYear = new Date().getFullYear()
  const years = Array.from({ length: currentYear - 2019 }, (_, i) => String(currentYear - i))

  const typeBadgeClass = (t: string) => {
    if (t === 'milestone' || t === 'celebration') return 'badge badge-llm'
    if (t === 'daily' || t === 'mundane') return 'badge badge-intent'
    if (t === 'outing' || t === 'travel') return 'badge badge-tool'
    return 'badge badge-git'
  }

  return (
    <div>
      <div className="nav">
        <Link to="/">Databases</Link>
        <span>/</span>
        <Link to={`/${dbName}`}>{dbName}</Link>
        <span>/</span>
        <strong>Timeline</strong>
      </div>
      <h1>Memory Timeline</h1>

      <div style={{ display: 'flex', gap: '1rem', marginBottom: '1rem', flexWrap: 'wrap' }}>
        <select value={person} onChange={e => updateFilter(setPerson, e.target.value)}>
          <option value="">All people</option>
          {people.map(p => (
            <option key={p.person_id} value={p.display_name}>{p.display_name}</option>
          ))}
        </select>

        <select value={type} onChange={e => updateFilter(setType, e.target.value)}>
          <option value="">All types</option>
          {MEMORY_TYPES.filter(Boolean).map(t => (
            <option key={t} value={t}>{t}</option>
          ))}
        </select>

        <select value={year} onChange={e => updateFilter(setYear, e.target.value)}>
          <option value="">All years</option>
          {years.map(y => (
            <option key={y} value={y}>{y}</option>
          ))}
        </select>
      </div>

      {loading ? (
        <div className="loading">Loading...</div>
      ) : memories.length === 0 ? (
        <p>No memories found.</p>
      ) : (
        <>
          <div style={{ display: 'flex', flexDirection: 'column', gap: '0.75rem' }}>
            {memories.map(m => (
              <div key={m.id} className="card">
                <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginBottom: '0.5rem' }}>
                  <strong>{m.title}</strong>
                  <span className={typeBadgeClass(m.type)}>{m.type}</span>
                </div>
                <p style={{ margin: '0.25rem 0', color: '#666', fontSize: '0.9em' }}>
                  {m.date}{m.location ? ` — ${m.location}` : ''}
                </p>
                <p style={{ margin: '0.25rem 0' }}>{m.description}</p>
                {m.people.length > 0 && (
                  <div style={{ display: 'flex', gap: '0.25rem', flexWrap: 'wrap', marginTop: '0.25rem' }}>
                    {m.people.map(p => (
                      <span key={p} className="badge badge-intent">{p}</span>
                    ))}
                  </div>
                )}
              </div>
            ))}
          </div>

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
