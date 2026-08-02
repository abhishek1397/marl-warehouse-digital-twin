import React from 'react';
import { StatItem } from '../types';

interface StatCardProps extends StatItem {
  iconNode?: React.ReactNode;
}

export const StatCard: React.FC<StatCardProps> = ({
  label,
  value,
  change,
  trend,
  iconNode,
}) => {
  return (
    <div className="glass-card p-5 rounded-xl border border-surface-border">
      <div className="flex items-center justify-between">
        <span className="text-xs font-medium text-slate-400 uppercase tracking-wider">
          {label}
        </span>
        {iconNode && <div className="text-accent p-2 rounded-lg bg-blue-500/10">{iconNode}</div>}
      </div>
      <div className="mt-3 flex items-baseline justify-between">
        <span className="text-2xl font-bold text-white font-mono">{value}</span>
        {change && (
          <span
            className={`text-xs font-semibold px-2 py-0.5 rounded ${
              trend === 'up'
                ? 'text-emerald-400 bg-emerald-500/10'
                : trend === 'down'
                ? 'text-rose-400 bg-rose-500/10'
                : 'text-slate-400 bg-slate-800'
            }`}
          >
            {change}
          </span>
        )}
      </div>
    </div>
  );
};
