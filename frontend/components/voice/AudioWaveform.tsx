'use client'

import { useEffect, useRef } from 'react'
import { useAtomValue } from 'jotai'
import { isRecordingAtom, audioLevelAtom } from '@/atoms/voice.atoms'
import { cn } from '@/lib/utils/cn'

interface AudioWaveformProps {
  className?: string
}

export function AudioWaveform({ className }: AudioWaveformProps) {
  const isRecording = useAtomValue(isRecordingAtom)
  const audioLevel = useAtomValue(audioLevelAtom)
  const canvasRef = useRef<HTMLCanvasElement>(null)
  const animationRef = useRef<number>()
  const waveHistoryRef = useRef<number[]>([])
  
  useEffect(() => {
    if (!canvasRef.current) return
    
    const canvas = canvasRef.current
    const ctx = canvas.getContext('2d')
    if (!ctx) return
    
    // Set canvas size
    const resizeCanvas = () => {
      const rect = canvas.getBoundingClientRect()
      canvas.width = rect.width * window.devicePixelRatio
      canvas.height = rect.height * window.devicePixelRatio
      ctx.scale(window.devicePixelRatio, window.devicePixelRatio)
    }
    
    resizeCanvas()
    window.addEventListener('resize', resizeCanvas)
    
    // Animation loop
    const animate = () => {
      if (!ctx || !canvas) return
      
      const width = canvas.width / window.devicePixelRatio
      const height = canvas.height / window.devicePixelRatio
      
      // Clear canvas
      ctx.clearRect(0, 0, width, height)
      
      // Update wave history
      if (isRecording) {
        waveHistoryRef.current.push(audioLevel)
        if (waveHistoryRef.current.length > width / 2) {
          waveHistoryRef.current.shift()
        }
      } else if (waveHistoryRef.current.length > 0) {
        // Fade out when not recording
        waveHistoryRef.current = waveHistoryRef.current.map(v => v * 0.95)
        if (waveHistoryRef.current[waveHistoryRef.current.length - 1] < 0.01) {
          waveHistoryRef.current = []
        }
      }
      
      // Draw waveform
      if (waveHistoryRef.current.length > 0) {
        ctx.strokeStyle = 'hsl(var(--primary))'
        ctx.lineWidth = 2
        ctx.beginPath()
        
        const step = width / waveHistoryRef.current.length
        waveHistoryRef.current.forEach((level, i) => {
          const x = i * step
          const amplitude = level * height * 0.4
          const y = height / 2 + Math.sin(i * 0.1) * amplitude
          
          if (i === 0) {
            ctx.moveTo(x, y)
          } else {
            ctx.lineTo(x, y)
          }
        })
        
        ctx.stroke()
        
        // Draw mirrored waveform
        ctx.beginPath()
        waveHistoryRef.current.forEach((level, i) => {
          const x = i * step
          const amplitude = level * height * 0.4
          const y = height / 2 - Math.sin(i * 0.1) * amplitude
          
          if (i === 0) {
            ctx.moveTo(x, y)
          } else {
            ctx.lineTo(x, y)
          }
        })
        
        ctx.stroke()
      } else {
        // Draw idle line
        ctx.strokeStyle = 'hsl(var(--muted-foreground))'
        ctx.lineWidth = 1
        ctx.setLineDash([5, 5])
        ctx.beginPath()
        ctx.moveTo(0, height / 2)
        ctx.lineTo(width, height / 2)
        ctx.stroke()
        ctx.setLineDash([])
      }
      
      animationRef.current = requestAnimationFrame(animate)
    }
    
    animate()
    
    return () => {
      window.removeEventListener('resize', resizeCanvas)
      if (animationRef.current) {
        cancelAnimationFrame(animationRef.current)
      }
    }
  }, [isRecording, audioLevel])
  
  return (
    <canvas
      ref={canvasRef}
      className={cn('w-full h-20', className)}
    />
  )
}