import { create } from 'zustand';

interface ThemeStoreState {
  theme: 'dark';
  setTheme: (theme: 'dark') => void;
}

export const useThemeStore = create<ThemeStoreState>((set) => ({
  theme: 'dark',
  setTheme: (theme) => set({ theme }),
}));
