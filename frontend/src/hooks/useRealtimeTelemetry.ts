import { useEffect, useState } from 'react';

export function useRealtimeTelemetry() {
  // Replace with a real interface for strict typing in a production state
  const [data, setData] = useState<any>(null);
  const [connected, setConnected] = useState(false);
  
  useEffect(() => {
    // Determine WS protocol based on current HTTP protocol (wss for https, ws for http)
    const wsUrl = import.meta.env.VITE_WS_URL || 'wss://api.ies-ev.com/telemetry';
    
    // In local dev, we might not have a running WebSocket server, so we'll 
    // mock behavior, but build in the physical connection attempt.
    let ws: WebSocket | null = null;
    
    try {
      ws = new WebSocket(wsUrl);
      
      ws.onopen = () => {
        setConnected(true);
        console.log('[WebSocket] Connected to Telemetry Stream');
      };
      
      ws.onmessage = (event) => {
        try {
          const telemetry = JSON.parse(event.data);
          setData(telemetry);
        } catch (e) {
          console.warn('[WebSocket] Malformed payload received');
        }
      };
      
      ws.onerror = (error) => {
        console.error('[WebSocket] Error connecting to stream:', error);
        setConnected(false);
      };
      
      ws.onclose = () => {
        setConnected(false);
        console.log('[WebSocket] Disconnected from Telemetry Stream');
      };
    } catch (err) {
      console.warn('[WebSocket] Initialization failed. Running in standalone mode.');
    }
    
    return () => {
      if (ws && ws.readyState === WebSocket.OPEN) {
        ws.close();
      }
    };
  }, []);
  
  return { data, connected };
}
