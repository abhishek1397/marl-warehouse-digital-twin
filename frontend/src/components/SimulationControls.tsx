import React, { useEffect } from 'react';
import { Play, Pause, RotateCcw, FastForward, StepForward, Sliders } from 'lucide-react';
import { useSimulationStore } from '../store/useSimulationStore';
import { Button } from './Button';
import { AlgorithmSelector } from './AlgorithmSelector';

export const SimulationControls: React.FC = () => {
  const {
    isRunning,
    isPaused,
    speed,
    gridSize,
    robotCount,
    packageCount,
    setRunning,
    setPaused,
    setSpeed,
    setGridSize,
    setRobotCount,
    setPackageCount,
    resetSimulation,
    stepSimulation,
  } = useSimulationStore();

  useEffect(() => {
    let interval: any = null;
    if (isRunning && !isPaused) {
      interval = setInterval(() => {
        stepSimulation();
      }, Math.max(100, 1000 / speed));
    } else {
      clearInterval(interval);
    }
    return () => clearInterval(interval);
  }, [isRunning, isPaused, speed, stepSimulation]);

  return (
    <div className="glass-panel p-5 rounded-xl border border-surface-border space-y-6">
      <div className="flex items-center justify-between border-b border-surface-border pb-3">
        <h3 className="text-sm font-bold uppercase tracking-wider text-white font-mono flex items-center">
          <Sliders className="w-4 h-4 mr-2 text-accent" />
          Simulation Control
        </h3>
      </div>

      {/* Algorithm Selector */}
      <AlgorithmSelector />

      {/* Playback Action Buttons */}
      <div className="space-y-2">
        <label className="text-xs font-semibold text-slate-300 uppercase tracking-wider font-mono">
          Playback Controls
        </label>
        <div className="grid grid-cols-2 gap-2">
          {!isRunning ? (
            <Button
              variant="primary"
              size="sm"
              icon={<Play className="w-4 h-4 fill-current" />}
              onClick={() => setRunning(true)}
            >
              Start
            </Button>
          ) : (
            <Button
              variant="secondary"
              size="sm"
              icon={<Pause className="w-4 h-4" />}
              onClick={() => setPaused(!isPaused)}
            >
              {isPaused ? 'Resume' : 'Pause'}
            </Button>
          )}

          <Button
            variant="outline"
            size="sm"
            icon={<RotateCcw className="w-4 h-4" />}
            onClick={resetSimulation}
          >
            Reset
          </Button>

          <Button
            variant="ghost"
            size="sm"
            icon={<StepForward className="w-4 h-4" />}
            onClick={stepSimulation}
          >
            Step
          </Button>

          <Button
            variant="ghost"
            size="sm"
            icon={<FastForward className="w-4 h-4" />}
            onClick={() => setRunning(true)}
          >
            Replay
          </Button>
        </div>
      </div>

      {/* Simulation Speed Slider */}
      <div className="space-y-2">
        <div className="flex justify-between items-center text-xs">
          <span className="font-semibold text-slate-300 uppercase tracking-wider font-mono">Speed Multiplier</span>
          <span className="font-mono text-accent font-bold">{speed}x</span>
        </div>
        <input
          type="range"
          min="0.5"
          max="5"
          step="0.5"
          value={speed}
          onChange={(e) => setSpeed(parseFloat(e.target.value))}
          className="w-full h-1.5 bg-slate-800 rounded-lg appearance-none cursor-pointer accent-accent"
        />
      </div>

      {/* Grid Size Selector */}
      <div className="space-y-2">
        <label className="text-xs font-semibold text-slate-300 uppercase tracking-wider font-mono">
          Warehouse Size
        </label>
        <select
          value={gridSize}
          onChange={(e) => setGridSize(parseInt(e.target.value))}
          className="w-full bg-surface-dark border border-surface-border text-white text-xs rounded-lg p-2 focus:ring-1 focus:ring-accent"
        >
          <option value={10}>10 x 10 Grid</option>
          <option value={15}>15 x 15 Grid</option>
          <option value={20}>20 x 20 Grid (Default)</option>
          <option value={25}>25 x 25 Grid</option>
        </select>
      </div>

      {/* Robot Fleet Count Selector */}
      <div className="space-y-2">
        <label className="text-xs font-semibold text-slate-300 uppercase tracking-wider font-mono">
          Fleet Robots
        </label>
        <select
          value={robotCount}
          onChange={(e) => setRobotCount(parseInt(e.target.value))}
          className="w-full bg-surface-dark border border-surface-border text-white text-xs rounded-lg p-2 focus:ring-1 focus:ring-accent"
        >
          <option value={1}>1 Robot</option>
          <option value={2}>2 Robots</option>
          <option value={4}>4 Robots (Default)</option>
          <option value={8}>8 Robots</option>
          <option value={16}>16 Robots</option>
          <option value={32}>32 Robots</option>
        </select>
      </div>

      {/* Package Count Selector */}
      <div className="space-y-2">
        <label className="text-xs font-semibold text-slate-300 uppercase tracking-wider font-mono">
          Package Deliveries
        </label>
        <select
          value={packageCount}
          onChange={(e) => setPackageCount(parseInt(e.target.value))}
          className="w-full bg-surface-dark border border-surface-border text-white text-xs rounded-lg p-2 focus:ring-1 focus:ring-accent"
        >
          <option value={5}>5 Packages</option>
          <option value={10}>10 Packages (Default)</option>
          <option value={20}>20 Packages</option>
          <option value={50}>50 Packages</option>
        </select>
      </div>
    </div>
  );
};
