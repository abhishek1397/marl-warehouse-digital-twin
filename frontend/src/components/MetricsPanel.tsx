import React from 'react';
import { Activity, ShieldAlert, Zap, Award, Package, Clock, Battery, Gauge } from 'lucide-react';
import { useSimulationStore } from '../store/useSimulationStore';

export const MetricsPanel: React.FC = () => {
  const { metrics } = useSimulationStore();

  const items = [
    { label: 'Episode', val: metrics.episode, icon: Clock, color: 'text-slate-400' },
    { label: 'Step', val: metrics.step, icon: Activity, color: 'text-accent' },
    { label: 'Mean Reward', val: metrics.reward.toFixed(2), icon: Award, color: 'text-blue-400' },
    { label: 'Throughput', val: `${metrics.throughput.toFixed(2)}/s`, icon: Zap, color: 'text-electric' },
    { label: 'Total Collisions', val: metrics.collisions, icon: ShieldAlert, color: 'text-emerald-400' },
    { label: 'Idle Robots', val: metrics.idleRobots, icon: Clock, color: 'text-amber-400' },
    { label: 'Battery Avg', val: `${metrics.batteryAvg}%`, icon: Battery, color: 'text-emerald-400' },
    { label: 'Deliveries', val: metrics.packagesDelivered, icon: Package, color: 'text-purple-400' },
    { label: 'Simulation FPS', val: metrics.fps, icon: Gauge, color: 'text-slate-400' },
    { label: 'Policy Entropy', val: metrics.policyEntropy.toFixed(2), icon: Activity, color: 'text-indigo-400' },
  ];

  return (
    <div className="glass-panel p-5 rounded-xl border border-surface-border space-y-4">
      <div className="flex items-center justify-between border-b border-surface-border pb-3">
        <h3 className="text-sm font-bold uppercase tracking-wider text-white font-mono flex items-center">
          <Activity className="w-4 h-4 mr-2 text-accent" />
          Live Metrics Telemetry
        </h3>
      </div>

      <div className="grid grid-cols-2 gap-3">
        {items.map((item, idx) => {
          const Icon = item.icon;
          return (
            <div key={idx} className="glass-card p-3 rounded-lg border border-surface-border space-y-1">
              <div className="flex items-center space-x-1.5 text-[10px] text-slate-400 font-mono uppercase tracking-wider">
                <Icon className={`w-3 h-3 ${item.color}`} />
                <span>{item.label}</span>
              </div>
              <div className={`text-base font-bold font-mono ${item.color}`}>
                {item.val}
              </div>
            </div>
          );
        })}
      </div>
    </div>
  );
};
