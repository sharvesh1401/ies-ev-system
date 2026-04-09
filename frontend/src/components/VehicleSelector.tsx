import { useState, useRef, useEffect, CSSProperties } from 'react';
import ReactDOM from 'react-dom';
import { useVehicle } from '../contexts/VehicleContext';
import './VehicleSelector.scss';

function VehicleSelector() {
  const { vehicle, switchVehicle, allVehicles } = useVehicle();
  const [isOpen, setIsOpen] = useState(false);
  const [justSwitched, setJustSwitched] = useState(false);
  const [dropdownPos, setDropdownPos] = useState<CSSProperties>({});
  const buttonRef = useRef<HTMLButtonElement>(null);
  const dropdownRef = useRef<HTMLDivElement>(null);

  // Close dropdown on outside click
  useEffect(() => {
    function handleClickOutside(event: MouseEvent) {
      const target = event.target as Node;
      const clickedButton = buttonRef.current?.contains(target);
      const clickedDropdown = dropdownRef.current?.contains(target);
      if (!clickedButton && !clickedDropdown) {
        setIsOpen(false);
      }
    }
    document.addEventListener('mousedown', handleClickOutside);
    return () => document.removeEventListener('mousedown', handleClickOutside);
  }, []);

  // Reposition on scroll / resize while open
  useEffect(() => {
    if (!isOpen) return;
    function reposition() {
      if (!buttonRef.current) return;
      const rect = buttonRef.current.getBoundingClientRect();
      setDropdownPos({
        position: 'fixed',
        top: rect.bottom + 8,
        left: rect.left,
        width: rect.width,
        zIndex: 9999,
      });
    }
    reposition();
    window.addEventListener('scroll', reposition, true);
    window.addEventListener('resize', reposition);
    return () => {
      window.removeEventListener('scroll', reposition, true);
      window.removeEventListener('resize', reposition);
    };
  }, [isOpen]);

  const handleToggle = () => {
    if (!isOpen && buttonRef.current) {
      const rect = buttonRef.current.getBoundingClientRect();
      setDropdownPos({
        position: 'fixed',
        top: rect.bottom + 8,
        left: rect.left,
        width: rect.width,
        zIndex: 9999,
      });
    }
    setIsOpen((prev) => !prev);
  };

  const handleSelect = (vehicleId: string) => {
    switchVehicle(vehicleId);
    setIsOpen(false);
    setJustSwitched(true);
    setTimeout(() => setJustSwitched(false), 2000);
  };

  return (
    <div className="vehicle-selector">
      <button
        ref={buttonRef}
        className="vehicle-selector-button"
        onClick={handleToggle}
        aria-label="Select vehicle"
        aria-expanded={isOpen}
        aria-haspopup="listbox"
      >
        <div className="vehicle-info">
          <div className="vehicle-icon">{vehicle.icon}</div>
          <div className="vehicle-details">
            <div className="vehicle-name">{vehicle.name}</div>
            <div className="vehicle-subtitle">{vehicle.subtitle}</div>
          </div>
        </div>
        <div className="dropdown-arrow">
          <svg
            className={isOpen ? 'rotated' : ''}
            width="16"
            height="16"
            viewBox="0 0 16 16"
          >
            <path
              d="M4 6l4 4 4-4"
              stroke="currentColor"
              strokeWidth="2"
              fill="none"
              strokeLinecap="round"
              strokeLinejoin="round"
            />
          </svg>
        </div>
      </button>

      {isOpen &&
        ReactDOM.createPortal(
          <div
            className="vehicle-dropdown"
            ref={dropdownRef}
            style={dropdownPos}
            role="listbox"
            aria-label="Select vehicle"
          >
            {Object.values(allVehicles).filter(v => !v.isCustom).map((v) => (
              <button
                key={v.id}
                role="option"
                aria-selected={v.id === vehicle.id}
                className={`vehicle-option ${v.id === vehicle.id ? 'selected' : ''}`}
                onClick={() => handleSelect(v.id)}
                style={{ '--vehicle-color': v.color } as CSSProperties}
              >
                <div className="option-icon">{v.icon}</div>
                <div className="option-details">
                  <div className="option-name">{v.name}</div>
                  <div className="option-subtitle">{v.subtitle}</div>
                  <div className="option-stats">
                    <span>SoH: {v.battery.soh_percent}%</span>
                    <span>•</span>
                    <span>{v.specs.mass_kg} kg</span>
                    <span>•</span>
                    <span>{v.range_km} km range</span>
                  </div>
                </div>
                {v.id === vehicle.id && (
                  <div className="selected-check">✓</div>
                )}
              </button>
            ))}

            <div className="px-4 py-2 flex items-center gap-3">
              <div className="flex-1 h-px bg-outline-variant/20"></div>
              <span className="text-[9px] text-slate-500 uppercase tracking-widest font-mono">
                Research
              </span>
              <div className="flex-1 h-px bg-outline-variant/20"></div>
            </div>

            {Object.values(allVehicles).filter(v => v.isCustom).map((v) => (
              <button
                key={v.id}
                className={`vehicle-option w-full flex items-center gap-3 px-4 py-3
                  border-b border-outline-variant/10 last:border-0 text-left
                  transition-all duration-200 cursor-pointer
                  ${v.id === vehicle.id 
                    ? 'bg-surface-container-highest border-l-[3px] border-[#A855F7]' 
                    : 'bg-transparent hover:bg-surface-container'
                  }`}
                onClick={() => handleSelect(v.id)}
              >
                <div className="text-2xl">⚗️</div>
                <div className="flex-1">
                  <div className="text-sm font-semibold text-white">Custom Lab</div>
                  <div className="text-[11px] text-slate-400 mb-1">
                    Configurable • Research Profile
                  </div>
                  <div className="text-[10px] text-[#A855F7] font-mono">
                    Click to configure any parameter →
                  </div>
                </div>
                {v.id === vehicle.id && (
                  <div className="text-[#A855F7] text-lg">✓</div>
                )}
              </button>
            ))}
          </div>,
          document.body
        )}

      {justSwitched && (
        <div className="saved-indicator">✓ Vehicle switched</div>
      )}
    </div>
  );
}

export default VehicleSelector;
