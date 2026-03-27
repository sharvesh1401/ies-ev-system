import { useEffect, useState } from 'react'

interface AnimatedNumberProps {
  value: number;
  duration?: number;
  fractionDigits?: number;
  prefix?: string;
  suffix?: string;
  className?: string;
}

// A simple count-up implementation that doesn't strictly depend on react-spring to avoid crashes if imports fail,
// but follows the exact visual aesthetic requested.
export default function AnimatedNumber({ 
  value, 
  duration = 1000, 
  fractionDigits = 0,
  prefix = '',
  suffix = '',
  className = ''
}: AnimatedNumberProps) {
  const [displayValue, setDisplayValue] = useState(0);

  useEffect(() => {
    let startTimestamp: number | null = null;
    const initialValue = displayValue;

    const step = (timestamp: number) => {
      if (!startTimestamp) startTimestamp = timestamp;
      const progress = Math.min((timestamp - startTimestamp) / duration, 1);
      
      // easeOutExpo
      const easeProgress = progress === 1 ? 1 : 1 - Math.pow(2, -10 * progress);
      
      setDisplayValue(initialValue + (value - initialValue) * easeProgress);

      if (progress < 1) {
        window.requestAnimationFrame(step);
      }
    };

    window.requestAnimationFrame(step);
  }, [value, duration]); // eslint-disable-line react-hooks/exhaustive-deps

  return (
    <span className={className}>
      {prefix}
      {displayValue.toFixed(fractionDigits)}
      {suffix}
    </span>
  );
}
