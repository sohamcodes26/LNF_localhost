import { Routes, Route } from 'react-router-dom';

import LandingPage from './pages/Landing';
import PostFoundPage from './pages/PostFound';
import FindLostItemPage from './pages/FindLostItem';
import MyAccountPage from './pages/MyAccount';
import FoundItemsPage from './pages/FoundItems';
import ReceivedItemsPage from './pages/ReceivedItems';
import HelpPage from './pages/Help';

import ProtectedRoute from './components/auth/ProtectedRoute';
import ResultsPage from './pages/Results';
import StatisticsDashboard from './pages/StatisticsDashboard';

function App() {
  return (
    <Routes>
      {}
      <Route path="/" element={<LandingPage />} />

      {}
      <Route element={<ProtectedRoute />}>
        <Route path="/post-found" element={<PostFoundPage />} />
        <Route path="/find-lost-item" element={<FindLostItemPage />} />
        <Route path="/account" element={<MyAccountPage />} />
        <Route path="/my-found-items" element={<FoundItemsPage />} />
        <Route path="/my-received-items" element={<ReceivedItemsPage />} />
        <Route path="/help" element={<HelpPage />} />
        <Route path="/results" element={<ResultsPage />} />
        <Route path="/statistics" element={<StatisticsDashboard />} />
        
      </Route>
    </Routes>
  );
}

export default App;
