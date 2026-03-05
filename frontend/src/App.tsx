import { Routes, Route } from 'react-router-dom'
import DatabaseList from './pages/DatabaseList'
import TraceList from './pages/TraceList'
import TraceDetail from './pages/TraceDetail'
import PeopleGraph from './pages/PeopleGraph'
import MemoryTimeline from './pages/MemoryTimeline'

function App() {
  return (
    <Routes>
      <Route path="/" element={<DatabaseList />} />
      <Route path="/:dbName" element={<TraceList />} />
      <Route path="/:dbName/people" element={<PeopleGraph />} />
      <Route path="/:dbName/timeline" element={<MemoryTimeline />} />
      <Route path="/:dbName/:traceId" element={<TraceDetail />} />
    </Routes>
  )
}

export default App
