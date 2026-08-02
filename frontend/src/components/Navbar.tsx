import React, { useState } from 'react';
import { NavLink } from 'react-router-dom';
import { Bot, Cpu, Activity, Play, FileText, Info, Menu, X, Github } from 'lucide-react';
import { Container } from './Container';

const navItems = [
  { name: 'Home', path: '/', icon: Bot },
  { name: 'Simulation', path: '/simulation', icon: Play },
  { name: 'Algorithms', path: '/algorithms', icon: Cpu },
  { name: 'Experiments', path: '/experiments', icon: Activity },
  { name: 'Research', path: '/research', icon: FileText },
  { name: 'Docs', path: '/docs', icon: FileText },
  { name: 'About', path: '/about', icon: Info },
];

export const Navbar: React.FC = () => {
  const [mobileMenuOpen, setMobileMenuOpen] = useState(false);

  return (
    <header className="sticky top-0 z-50 glass-panel border-b border-surface-border bg-background/80 backdrop-blur-md">
      <Container>
        <div className="flex items-center justify-between h-16">
          {/* Left: Brand Logo */}
          <NavLink to="/" className="flex items-center space-x-3 group">
            <div className="p-2 rounded-lg bg-accent/10 border border-accent/30 text-accent group-hover:bg-accent group-hover:text-white transition-all duration-200">
              <Bot className="w-5 h-5" />
            </div>
            <div>
              <span className="text-base font-extrabold text-white tracking-wider uppercase block">
                Warehouse <span className="text-accent">Twin</span>
              </span>
              <span className="text-[10px] text-slate-400 font-mono tracking-widest block uppercase -mt-1">
                MARL Platform
              </span>
            </div>
          </NavLink>

          {/* Center: Desktop Navigation */}
          <nav className="hidden md:flex items-center space-x-1">
            {navItems.map((item) => {
              const Icon = item.icon;
              return (
                <NavLink
                  key={item.path}
                  to={item.path}
                  className={({ isActive }) =>
                    `flex items-center space-x-2 px-3 py-1.5 rounded-lg text-xs font-medium transition-all duration-200 ${
                      isActive
                        ? 'bg-accent/15 text-accent border border-accent/30 shadow-glow'
                        : 'text-slate-300 hover:text-white hover:bg-slate-800/60'
                    }`
                  }
                >
                  <Icon className="w-3.5 h-3.5" />
                  <span>{item.name}</span>
                </NavLink>
              );
            })}
          </nav>

          {/* Right: GitHub Placeholder Link */}
          <div className="hidden md:flex items-center space-x-3">
            <a
              href="https://github.com"
              target="_blank"
              rel="noreferrer"
              className="flex items-center space-x-2 px-3 py-1.5 rounded-lg text-xs font-medium text-slate-300 bg-surface-light border border-surface-border hover:text-white hover:border-slate-600 transition-all duration-200"
            >
              <Github className="w-4 h-4" />
              <span>GitHub</span>
            </a>
          </div>

          {/* Mobile Hamburger Button */}
          <div className="md:hidden flex items-center">
            <button
              onClick={() => setMobileMenuOpen(!mobileMenuOpen)}
              className="p-2 rounded-lg text-slate-300 hover:text-white hover:bg-surface-light focus:outline-none"
            >
              {mobileMenuOpen ? <X className="w-6 h-6" /> : <Menu className="w-6 h-6" />}
            </button>
          </div>
        </div>
      </Container>

      {/* Mobile Navigation Drawer */}
      {mobileMenuOpen && (
        <div className="md:hidden border-b border-surface-border bg-surface-dark px-4 pt-2 pb-4 space-y-1">
          {navItems.map((item) => {
            const Icon = item.icon;
            return (
              <NavLink
                key={item.path}
                to={item.path}
                onClick={() => setMobileMenuOpen(false)}
                className={({ isActive }) =>
                  `flex items-center space-x-3 px-3 py-2 rounded-lg text-sm font-medium ${
                    isActive
                      ? 'bg-accent/20 text-accent font-semibold'
                      : 'text-slate-300 hover:bg-slate-800'
                  }`
                }
              >
                <Icon className="w-4 h-4" />
                <span>{item.name}</span>
              </NavLink>
            );
          })}
        </div>
      )}
    </header>
  );
};
