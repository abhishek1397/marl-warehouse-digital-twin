import React from 'react';
import { Square } from 'lucide-react';
import { GridEntity } from '../store/useSimulationStore';

interface EntitySpriteProps {
  entity: GridEntity;
  cellSize: number;
}

export const ObstacleSprite: React.FC<EntitySpriteProps> = ({ entity, cellSize }) => {
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
      <div className="w-full h-full rounded bg-rose-900/30 border border-rose-700/50 flex items-center justify-center text-rose-500">
        <Square className="w-3.5 h-3.5 fill-current" />
      </div>
    </div>
  );
};
