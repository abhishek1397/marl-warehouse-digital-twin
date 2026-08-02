import React from 'react';
import { PackageCheck } from 'lucide-react';
import { GridEntity } from '../store/useSimulationStore';

interface EntitySpriteProps {
  entity: GridEntity;
  cellSize: number;
}

export const ShelfSprite: React.FC<EntitySpriteProps> = ({ entity, cellSize }) => {
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
      <div className="w-full h-full rounded bg-slate-800/80 border border-slate-700 flex items-center justify-center text-slate-400">
        <PackageCheck className="w-3.5 h-3.5" />
      </div>
    </div>
  );
};
