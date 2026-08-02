/** @type {import('tailwindcss').Config} */
export default {
  content: ['./index.html', './src/**/*.{js,ts,jsx,tsx}'],
  theme: {
    extend: {
      colors: {
        background: '#090d16',
        surface: {
          DEFAULT: '#111827',
          dark: '#0c101d',
          light: '#1f293d',
          border: '#1e293b',
        },
        accent: {
          DEFAULT: '#3b82f6',
          hover: '#2563eb',
          glow: '#60a5fa',
        },
        electric: '#00f0ff',
      },
      fontFamily: {
        sans: ['Inter', 'system-ui', 'sans-serif'],
        mono: ['Fira Code', 'monospace'],
      },
      boxShadow: {
        glow: '0 0 25px -5px rgba(59, 130, 246, 0.5)',
        electric: '0 0 20px -3px rgba(0, 240, 255, 0.4)',
      },
    },
  },
  plugins: [],
};
