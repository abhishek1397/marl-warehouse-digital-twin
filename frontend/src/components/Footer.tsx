import React from 'react';
import { Container } from './Container';

export const Footer: React.FC = () => {
  return (
    <footer className="mt-20 border-t border-surface-border bg-surface-dark/80 py-8 text-xs text-slate-400">
      <Container>
        <div className="flex flex-col md:flex-row items-center justify-between space-y-4 md:space-y-0">
          <div>
            <span className="font-semibold text-slate-200">Warehouse Digital Twin MARL Platform</span>
            <p className="text-[11px] text-slate-400 mt-0.5">
              Reinforcement Learning Research Platform for Autonomous Robot Fleets
            </p>
          </div>
          <div className="flex items-center space-x-6">
            <span className="inline-flex items-center space-x-1.5">
              <span className="w-2 h-2 rounded-full bg-emerald-400 animate-pulse"></span>
              <span>Platform Core Active</span>
            </span>
            <span>Python 3.11 (marl_env)</span>
            <span>PyTorch 2.x</span>
          </div>
        </div>
      </Container>
    </footer>
  );
};
