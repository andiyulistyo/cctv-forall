import { createContext, useContext, useState, ReactNode } from "react";
import { api } from "./api";

interface AuthCtx {
  isAuthed: boolean;
  login: (u: string, p: string) => Promise<void>;
  logout: () => void;
}

const Ctx = createContext<AuthCtx>(null!);

export function AuthProvider({ children }: { children: ReactNode }) {
  const [isAuthed, setAuthed] = useState<boolean>(!!localStorage.getItem("token"));

  const login = async (u: string, p: string) => {
    await api.login(u, p);
    setAuthed(true);
  };
  const logout = () => {
    api.logout();
    setAuthed(false);
  };

  return <Ctx.Provider value={{ isAuthed, login, logout }}>{children}</Ctx.Provider>;
}

export const useAuth = () => useContext(Ctx);
