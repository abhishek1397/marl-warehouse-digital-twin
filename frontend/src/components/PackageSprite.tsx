import React from 'react';
import { Package } from 'lucide-react';
import { GridEntity } from '../store/useSimulationStore';

interface EntitySpriteProps {
  entity: GridEntity;
  cellSize: number;
}

export const PackageSprite: React.FC<EntitySpriteProps> = ({ entity, cellSize }) => {
  return (
    <div
      style={{
        width: cellSize,
        height: cellSize,
        position: 'absolute',
        top: entity.y * cellSize,
        left: entity.x * cellSize,
        zIndex: 12,
      }}
      className="flex items-center justify-center p-1"
    >
      <div className="w-full h-full rounded bg-amber-500/10 border border-amber-500/40 shadow-glow flex items-center justify-center text-amber-400">
        <Package className="w-3.5 h-3.5" />
      </div>
    </div>
  );
};
