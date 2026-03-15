import React from 'react';
import { motion, AnimatePresence } from 'motion/react';
import { X, Zap, Battery, Gauge, Cpu } from 'lucide-react';
import { Vehicle } from '../types';
import { Language, translations } from '../lib/translations';

interface VehicleDetailProps {
  vehicle: Vehicle | null;
  onClose: () => void;
  lang: Language;
}

export const VehicleDetail: React.FC<VehicleDetailProps> = ({ vehicle, onClose, lang }) => {
  const t = translations[lang].vehicle;

  if (!vehicle) return null;

  const displayType = lang === 'zh' && vehicle.typeZh ? vehicle.typeZh : vehicle.type;
  const displayDesc = lang === 'zh' && vehicle.descriptionZh ? vehicle.descriptionZh : vehicle.description;
  const displayFeatures = lang === 'zh' && vehicle.featuresZh ? vehicle.featuresZh : vehicle.features;

  return (
    <AnimatePresence>
      <div className="fixed inset-0 z-50 flex items-center justify-center p-4 md:p-8">
        <motion.div
          initial={{ opacity: 0 }}
          animate={{ opacity: 1 }}
          exit={{ opacity: 0 }}
          onClick={onClose}
          className="absolute inset-0 bg-slate-900/60 backdrop-blur-sm"
        />

        <motion.div
          initial={{ opacity: 0, scale: 0.9, y: 20 }}
          animate={{ opacity: 1, scale: 1, y: 0 }}
          exit={{ opacity: 0, scale: 0.9, y: 20 }}
          className="relative w-full max-w-4xl bg-white rounded-[2.5rem] overflow-hidden shadow-2xl flex flex-col md:flex-row"
        >
          <button
            onClick={onClose}
            className="absolute top-6 right-6 z-10 p-2 bg-slate-100 hover:bg-slate-200 rounded-full transition-colors"
          >
            <X size={20} className="text-slate-600" />
          </button>

          {/* Image Section */}
          <div className="md:w-1/2 relative bg-slate-50 flex items-center justify-center p-12 overflow-hidden">
            <div
              className="absolute inset-0 opacity-10"
              style={{ backgroundColor: vehicle.color }}
            />
            <motion.img
              initial={{ x: 40, opacity: 0 }}
              animate={{ x: 0, opacity: 1 }}
              transition={{ delay: 0.2 }}
              src={vehicle.image}
              alt={vehicle.name}
              className="relative z-10 w-full h-auto object-contain drop-shadow-2xl"
            />
            <div className="absolute bottom-12 left-12">
              <h2 className="text-8xl font-black text-slate-900/5 select-none">{vehicle.name}</h2>
            </div>
          </div>

          {/* Content Section */}
          <div className="md:w-1/2 p-12 flex flex-col">
            <div className="mb-8">
              <div className="flex items-center space-x-2 mb-2">
                <div className="w-2 h-2 rounded-full" style={{ backgroundColor: vehicle.color }} />
                <span className="text-[10px] uppercase tracking-[0.3em] font-black text-slate-400">
                  {displayType}
                </span>
              </div>
              <h2 className="text-4xl font-black text-slate-900 tracking-tight">{vehicle.name}</h2>
            </div>

            <p className="text-slate-500 leading-relaxed mb-10 font-medium">
              {displayDesc}
            </p>

            <div className="grid grid-cols-2 gap-8 mb-10">
              <div className="space-y-1">
                <div className="flex items-center space-x-2 text-slate-400">
                  <Zap size={14} />
                  <span className="text-[10px] uppercase tracking-widest font-bold">{t.acceleration}</span>
                </div>
                <p className="text-xl font-black text-slate-900">{vehicle.specs.acceleration}</p>
              </div>
              <div className="space-y-1">
                <div className="flex items-center space-x-2 text-slate-400">
                  <Battery size={14} />
                  <span className="text-[10px] uppercase tracking-widest font-bold">{t.range}</span>
                </div>
                <p className="text-xl font-black text-slate-900">{vehicle.specs.range}</p>
              </div>
              <div className="space-y-1">
                <div className="flex items-center space-x-2 text-slate-400">
                  <Gauge size={14} />
                  <span className="text-[10px] uppercase tracking-widest font-bold">{t.power}</span>
                </div>
                <p className="text-xl font-black text-slate-900">{vehicle.specs.power}</p>
              </div>
              <div className="space-y-1">
                <div className="flex items-center space-x-2 text-slate-400">
                  <Cpu size={14} />
                  <span className="text-[10px] uppercase tracking-widest font-bold">{t.torque}</span>
                </div>
                <p className="text-xl font-black text-slate-900">{vehicle.specs.torque}</p>
              </div>
            </div>

            <div className="mt-auto flex flex-wrap gap-2">
              {displayFeatures.map((f, i) => (
                <span
                  key={i}
                  className="text-[10px] tracking-wide px-3 py-1.5 rounded-lg font-semibold border border-slate-100 bg-slate-50 text-slate-500"
                >
                  {f}
                </span>
              ))}
            </div>
          </div>
        </motion.div>
      </div>
    </AnimatePresence>
  );
};
