import { Routes, Route } from 'react-router-dom'
import DatabaseList from './pages/DatabaseList'
import TraceList from './pages/TraceList'
import TraceDetail from './pages/TraceDetail'
import PeopleGraph from './pages/PeopleGraph'
import MemoryTimeline from './pages/MemoryTimeline'
import DigestHistory from './pages/DigestHistory'
import IntegrityRuns from './pages/IntegrityRuns'
import IntegrityRunDetail from './pages/IntegrityRunDetail'

function App() {
  return (
    <Routes>
      <Route path="/" element={<DatabaseList />} />
      <Route path="/:dbName" element={<TraceList />} />
      <Route path="/:dbName/people" element={<PeopleGraph />} />
      <Route path="/:dbName/timeline" element={<MemoryTimeline />} />
      <Route path="/:dbName/digests" element={<DigestHistory />} />
      <Route path="/:dbName/integrity" element={<IntegrityRuns />} />
      <Route path="/:dbName/integrity/:runId" element={<IntegrityRunDetail />} />
      <Route path="/:dbName/:traceId" element={<TraceDetail />} />
    </Routes>
  )
}

export default App
