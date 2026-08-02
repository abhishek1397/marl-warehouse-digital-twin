import React, { useState } from 'react';
import { motion } from 'framer-motion';
import { Bot, Battery, Zap } from 'lucide-react';
import { RobotEntity } from '../store/useSimulationStore';

interface RobotSpriteProps {
  robot: RobotEntity;
  cellSize: number;
}

export const RobotSprite: React.FC<RobotSpriteProps> = ({ robot, cellSize }) => {
  const [hovered, setHovered] = useState(false);

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
      {/* Glowing Robot Body */}
      <div
        className="w-4/5 h-4/5 rounded-lg flex items-center justify-center border transition-all shadow-glow"
        style={{
          backgroundColor: `${robot.color}22`,
          borderColor: robot.color,
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
        <div className="absolute bottom-full mb-2 left-1/2 -translate-x-1/2 w-48 p-3 glass-panel rounded-xl border border-accent/40 shadow-2xl z-50 text-[11px] text-slate-200 pointer-events-none space-y-1 font-sans">
          <div className="flex items-center justify-between border-b border-slate-700 pb-1 mb-1">
            <span className="font-bold text-white font-mono">{robot.id}</span>
            <span className="text-[10px] px-1.5 py-0.5 rounded bg-blue-500/20 text-accent font-semibold">
              {robot.state}
            </span>
          </div>
          <div className="flex justify-between items-center text-slate-400">
            <span>Coordinates:</span>
            <span className="font-mono text-white">({robot.x}, {robot.y})</span>
          </div>
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
