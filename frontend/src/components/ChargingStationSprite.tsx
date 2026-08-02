import React from 'react';
import { Zap } from 'lucide-react';
import { GridEntity } from '../store/useSimulationStore';

interface EntitySpriteProps {
  entity: GridEntity;
  cellSize: number;
}

export const ChargingStationSprite: React.FC<EntitySpriteProps> = ({ entity, cellSize }) => {
  return (
    <div
      style={{
        width: cellSize,
        height: cellSize,
        position: 'absolute',
        top: entity.y * cellSize,
        left: entity.x * cellSize,
        zIndex: 10,
      }}
      className="flex items-center justify-center p-1"
    >
      <div className="w-full h-full rounded bg-emerald-500/10 border border-emerald-500/40 shadow-glow flex items-center justify-center text-emerald-400">
        <Zap className="w-3.5 h-3.5" />
      </div>
    </div>
  );
};
