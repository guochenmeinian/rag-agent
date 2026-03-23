import React from 'react';
import { Vehicle } from '../types';
import { Language } from '../lib/translations';

interface VehicleCardProps {
  vehicle: Vehicle;
  isActive?: boolean;
  isCollapsed?: boolean;
  lang?: Language;
  onClick?: () => void;
}

export const VehicleCard: React.FC<VehicleCardProps> = ({ vehicle, isActive, isCollapsed, lang = 'zh', onClick }) => {
  const displayType = lang === 'zh' && vehicle.typeZh ? vehicle.typeZh : vehicle.type;

  if (isCollapsed) {
    return (
      <div
        onClick={onClick}
        className={`relative group cursor-pointer w-10 h-10 rounded-xl flex items-center justify-center transition-all duration-300 overflow-hidden ${
          isActive ? 'bg-white shadow-md' : 'hover:bg-white/20'
        }`}
      >
        <img
          src={vehicle.image}
          alt={vehicle.name}
          className="w-full h-full object-cover object-center"
        />
        <div className="absolute left-14 px-3 py-2 bg-slate-900 text-white text-[10px] font-bold rounded-lg opacity-0 group-hover:opacity-100 pointer-events-none transition-opacity whitespace-nowrap z-50 shadow-xl">
          {vehicle.name}
          <div className="absolute left-[-4px] top-1/2 -translate-y-1/2 w-2 h-2 bg-slate-900 rotate-45" />
        </div>
      </div>
    );
  }

  return (
    <div
      onClick={onClick}
      className={`group cursor-pointer transition-all duration-300 rounded-xl border overflow-hidden ${
        isActive
          ? 'bg-white border-white shadow-sm'
          : 'bg-transparent border-transparent hover:bg-white/10'
      }`}
    >
      {/* Car image strip */}
      <div
        className="w-full h-16 relative overflow-hidden"
        style={{ backgroundColor: vehicle.color + '18' }}
      >
        <img
          src={vehicle.image}
          alt={vehicle.name}
          className="absolute inset-0 w-full h-full object-contain object-center scale-110"
        />
      </div>

      {/* Name / type row */}
      <div className="flex items-center justify-between px-3 py-2.5">
        <div className="flex items-center space-x-2.5">
          <div
            className="w-1 h-6 rounded-full flex-shrink-0"
            style={{ backgroundColor: vehicle.color }}
          />
          <div className="flex flex-col gap-0.5">
            <h3 className="text-xs font-black tracking-tight text-slate-800 leading-none">{vehicle.name}</h3>
            <span className="text-[11px] text-slate-500 font-medium leading-none">{displayType}</span>
          </div>
        </div>
        <div className="opacity-0 group-hover:opacity-100 transition-opacity">
          <div className="w-5 h-5 rounded-lg bg-slate-900 flex items-center justify-center">
            <div className="w-1 h-1 bg-white rounded-full" />
          </div>
        </div>
      </div>
    </div>
  );
};
