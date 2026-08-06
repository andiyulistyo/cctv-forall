import { Navigate, Route, Routes, Link, useLocation } from "react-router-dom";
import { useAuth } from "./auth";
import Login from "./pages/Login";
import Dashboard from "./pages/Dashboard";
import SourceDetail from "./pages/SourceDetail";
import Plates from "./pages/Plates";

function RequireAuth({ children }: { children: JSX.Element }) {
  const { isAuthed } = useAuth();
  const loc = useLocation();
  if (!isAuthed) return <Navigate to="/login" state={{ from: loc }} replace />;
  return children;
}

function NavBar() {
  const { logout } = useAuth();
  return (
    <nav className="flex items-center gap-4 border-b border-slate-800 bg-slate-900 px-6 py-3">
      <span className="text-lg font-semibold text-sky-400">🎥 Detection</span>
      <Link className="hover:text-sky-300" to="/">
        Dashboard
      </Link>
      <Link className="hover:text-sky-300" to="/plates">
        Plat Nomor
      </Link>
      <button
        onClick={logout}
        className="ml-auto rounded bg-slate-800 px-3 py-1 text-sm hover:bg-slate-700"
      >
        Logout
      </button>
    </nav>
  );
}

function Shell({ children }: { children: JSX.Element }) {
  return (
    <div className="min-h-screen">
      <NavBar />
      <main className="mx-auto max-w-7xl px-6 py-6">{children}</main>
    </div>
  );
}

export default function App() {
  return (
    <Routes>
      <Route path="/login" element={<Login />} />
      <Route
        path="/"
        element={
          <RequireAuth>
            <Shell>
              <Dashboard />
            </Shell>
          </RequireAuth>
        }
      />
      <Route
        path="/sources/:id"
        element={
          <RequireAuth>
            <Shell>
              <SourceDetail />
            </Shell>
          </RequireAuth>
        }
      />
      <Route
        path="/plates"
        element={
          <RequireAuth>
            <Shell>
              <Plates />
            </Shell>
          </RequireAuth>
        }
      />
      <Route path="*" element={<Navigate to="/" replace />} />
    </Routes>
  );
}
