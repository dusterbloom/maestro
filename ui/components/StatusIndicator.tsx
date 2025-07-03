'use client';

import { useEffect, useState } from 'react';

interface StatusIndicatorProps {
  status: 'idle' | 'connecting' | 'connected' | 'recording' | 'processing' | 'error';
  error?: string;
}

export default function StatusIndicator({ status, error }: StatusIndicatorProps) {
  const [dots, setDots] = useState('');
  
  useEffect(() => {
    if (status === 'connecting' || status === 'processing') {
      const interval = setInterval(() => {
        setDots(prev => prev.length >= 3 ? '' : prev + '.');
      }, 500);
      
      return () => clearInterval(interval);
    } else {
      setDots('');
    }
  }, [status]);
  
  const getStatusDisplay = () => {
    switch (status) {
      case 'idle':
        return { text: 'Not Connected', color: 'text-gray-500', bg: 'bg-gray-100', icon: '⭕' };
      case 'connecting':
        return { text: `Connecting${dots}`, color: 'text-yellow-600', bg: 'bg-yellow-50', icon: '🔄' };
      case 'connected':
        return { text: 'Ready', color: 'text-green-600', bg: 'bg-green-50', icon: '✅' };
      case 'recording':
        return { text: 'Recording', color: 'text-red-600', bg: 'bg-red-50', icon: '🎤' };
      case 'processing':
        return { text: `Processing${dots}`, color: 'text-blue-600', bg: 'bg-blue-50', icon: '⚡' };
      case 'error':
        return { text: 'Error', color: 'text-red-700', bg: 'bg-red-100', icon: '❌' };
      default:
        return { text: 'Unknown', color: 'text-gray-500', bg: 'bg-gray-100', icon: '❓' };
    }
  };
  
  const { text, color, bg, icon } = getStatusDisplay();
  
  return (
    <div className={`inline-flex items-center gap-2 px-4 py-2 rounded-full ${bg} ${color} font-medium`}>
      <span className="text-lg">{icon}</span>
      <span>{text}</span>
      {error && status === 'error' && (
        <div className="ml-2 text-sm text-red-600 max-w-xs truncate" title={error}>
          {error}
        </div>
      )}
    </div>
  );
}