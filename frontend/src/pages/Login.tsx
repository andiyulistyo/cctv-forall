import { useState } from "react";
import { useNavigate } from "react-router-dom";
import { useAuth } from "../auth";

export default function Login() {
  const { login } = useAuth();
  const nav = useNavigate();
  const [username, setUsername] = useState("admin");
  const [password, setPassword] = useState("");
  const [error, setError] = useState("");
  const [busy, setBusy] = useState(false);

  const submit = async (e: React.FormEvent) => {
    e.preventDefault();
    setBusy(true);
    setError("");
    try {
      await login(username, password);
      nav("/");
    } catch (err: any) {
      setError(err.message || "Login gagal");
    } finally {
      setBusy(false);
    }
  };

  return (
    <div className="flex min-h-screen items-center justify-center">
      <form
        onSubmit={submit}
        className="w-80 rounded-xl border border-slate-800 bg-slate-900 p-6 shadow-xl"
      >
        <h1 className="mb-1 text-xl font-semibold text-sky-400">Detection Dashboard</h1>
        <p className="mb-5 text-sm text-slate-400">Masuk untuk melanjutkan</p>
        <label className="mb-1 block text-sm text-slate-300">Username</label>
        <input
          className="mb-3 w-full rounded border border-slate-700 bg-slate-800 px-3 py-2"
          value={username}
          onChange={(e) => setUsername(e.target.value)}
          autoFocus
        />
        <label className="mb-1 block text-sm text-slate-300">Password</label>
        <input
          type="password"
          className="mb-4 w-full rounded border border-slate-700 bg-slate-800 px-3 py-2"
          value={password}
          onChange={(e) => setPassword(e.target.value)}
        />
        {error && <div className="mb-3 text-sm text-red-400">{error}</div>}
        <button
          disabled={busy}
          className="w-full rounded bg-sky-600 px-3 py-2 font-medium hover:bg-sky-500 disabled:opacity-50"
        >
          {busy ? "Memproses..." : "Login"}
        </button>
      </form>
    </div>
  );
}
