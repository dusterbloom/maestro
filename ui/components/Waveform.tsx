'use client';

import { useEffect, useRef, useState } from 'react';

interface WaveformProps {
  isRecording: boolean;
  audioLevel?: number;
}

export default function Waveform({ isRecording, audioLevel = 0 }: WaveformProps) {
  const canvasRef = useRef<HTMLCanvasElement>(null);
  const animationRef = useRef<number>();
  const [waveData, setWaveData] = useState<number[]>(new Array(50).fill(0));
  
  useEffect(() => {
    const canvas = canvasRef.current;
    if (!canvas) return;
    
    const ctx = canvas.getContext('2d');
    if (!ctx) return;
    
    const animate = () => {
      // Clear canvas
      ctx.clearRect(0, 0, canvas.width, canvas.height);
      
      if (isRecording) {
        // Generate wave animation based on audio level
        setWaveData(prev => {
          const newData = [...prev.slice(1)];
          const level = audioLevel + Math.random() * 0.3;
          newData.push(level);
          return newData;
        });
      } else {
        // Fade out animation
        setWaveData(prev => prev.map(val => val * 0.95));
      }
      
      // Draw waveform
      const barWidth = canvas.width / waveData.length;
      const centerY = canvas.height / 2;
      
      ctx.fillStyle = isRecording ? '#ef4444' : '#3b82f6';
      
      waveData.forEach((level, index) => {
        const barHeight = level * centerY;
        const x = index * barWidth;
        
        // Draw bar (mirrored above and below center)
        ctx.fillRect(x, centerY - barHeight, barWidth - 1, barHeight * 2);
      });
      
      animationRef.current = requestAnimationFrame(animate);
    };
    
    animate();
    
    return () => {
      if (animationRef.current) {
        cancelAnimationFrame(animationRef.current);
      }
    };
  }, [isRecording, audioLevel, waveData]);
  
  if (!isRecording && waveData.every(val => val < 0.01)) {
    return null; // Don't show when not recording and no activity
  }
  
  return (
    <div className="absolute -bottom-20 left-1/2 transform -translate-x-1/2">
      <canvas
        ref={canvasRef}
        width={200}
        height={60}
        className="opacity-75"
      />
    </div>
  );
}