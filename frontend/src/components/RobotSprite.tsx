import React, { useState } from 'react';
import { motion } from 'framer-motion';
import { Bot, Battery, AlertTriangle } from 'lucide-react';
import { RobotEntity, useSimulationStore } from '../store/useSimulationStore';

interface RobotSpriteProps {
  robot: RobotEntity;
  cellSize: number;
}

export const RobotSprite: React.FC<RobotSpriteProps> = ({ robot, cellSize }) => {
  const [hovered, setHovered] = useState(false);
  const showDebugOverlay = useSimulationStore((state) => state.showDebugOverlay);

  return (
    <motion.div
      initial={false}
      animate={{
        x: robot.x * cellSize,
        y: robot.y * cellSize,
      }}
      transition={{ type: 'spring', stiffness: 300, damping: 25 }}
      style={{
        width: cellSize,
        height: cellSize,
        position: 'absolute',
        top: 0,
        left: 0,
        zIndex: 20,
      }}
      onMouseEnter={() => setHovered(true)}
      onMouseLeave={() => setHovered(false)}
      className="flex items-center justify-center cursor-pointer relative group"
    >
      {/* Enhanced Above-Robot Canvas Debug Overlay */}
      {showDebugOverlay && (
        <div className="absolute -top-10 left-1/2 -translate-x-1/2 px-1.5 py-0.5 bg-slate-950/95 border border-slate-700/90 rounded text-[8px] font-mono whitespace-nowrap shadow-xl z-30 pointer-events-none flex flex-col items-center leading-tight">
          <div className="flex items-center gap-1 font-bold" style={{ color: robot.color }}>
            {robot.isCollision && <AlertTriangle className="w-2.5 h-2.5 text-rose-500 animate-pulse" />}
            <span>{robot.id}</span>
            {robot.assignedPackageId && <span className="text-amber-300">[{robot.assignedPackageId}]</span>}
          </div>
          <div className="text-slate-300 text-[7.5px] flex items-center gap-1">
            <span className="text-cyan-300">{robot.state}</span>
            {robot.currentAction && <span className="text-indigo-300">({robot.currentAction})</span>}
            {robot.targetPosition && <span className="text-emerald-300">→({robot.targetPosition[0]},{robot.targetPosition[1]})</span>}
          </div>
        </div>
      )}

      {/* Glowing Robot Body */}
      <div
        className={`w-4/5 h-4/5 rounded-lg flex items-center justify-center border transition-all shadow-glow ${
          robot.isCollision ? 'ring-2 ring-rose-500 animate-pulse' : ''
        }`}
        style={{
          backgroundColor: `${robot.color}22`,
          borderColor: robot.isCollision ? '#ef4444' : robot.color,
          boxShadow: `0 0 15px ${robot.color}66`,
        }}
      >
        <Bot className="w-4 h-4" style={{ color: robot.color }} />
      </div>

      {/* Battery Indicator Bar */}
      <div className="absolute -bottom-1 left-1 right-1 h-1 bg-slate-800 rounded-full overflow-hidden border border-slate-700">
        <div
          className={`h-full ${robot.battery > 50 ? 'bg-emerald-400' : robot.battery > 20 ? 'bg-amber-400' : 'bg-rose-500'}`}
          style={{ width: `${robot.battery}%` }}
        />
      </div>

      {/* Hover Tooltip */}
      {hovered && (
        <div className="absolute bottom-full mb-2 left-1/2 -translate-x-1/2 w-52 p-3 glass-panel rounded-xl border border-accent/40 shadow-2xl z-50 text-[11px] text-slate-200 pointer-events-none space-y-1 font-sans">
          <div className="flex items-center justify-between border-b border-slate-700 pb-1 mb-1">
            <span className="font-bold text-white font-mono">{robot.id}</span>
            <span className="text-[10px] px-1.5 py-0.5 rounded bg-blue-500/20 text-accent font-semibold">
              {robot.state}
            </span>
          </div>
          <div className="flex justify-between items-center text-slate-400">
            <span>Position:</span>
            <span className="font-mono text-white">({robot.x}, {robot.y})</span>
          </div>
          {robot.targetPosition && (
            <div className="flex justify-between items-center text-slate-400">
              <span>Target Goal:</span>
              <span className="font-mono text-cyan-400">({robot.targetPosition[0]}, {robot.targetPosition[1]})</span>
            </div>
          )}
          {robot.assignedPackageId && (
            <div className="flex justify-between items-center text-slate-400">
              <span>Assigned Task:</span>
              <span className="font-mono text-amber-300">{robot.assignedPackageId}</span>
            </div>
          )}
          {robot.currentAction && (
            <div className="flex justify-between items-center text-slate-400">
              <span>Current Action:</span>
              <span className="font-mono text-indigo-300">{robot.currentAction}</span>
            </div>
          )}
          <div className="flex justify-between items-center text-slate-400">
            <span>Battery Level:</span>
            <span className="font-mono text-emerald-400 flex items-center">
              <Battery className="w-3 h-3 mr-1" />
              {robot.battery}%
            </span>
          </div>
        </div>
      )}
    </motion.div>
  );
};
