import { NextRequest, NextResponse } from 'next/server';

export async function POST(request: NextRequest) {
  try {
    const body = await request.json();
    
    // Forward request to orchestrator
    const orchestratorUrl = process.env.ORCHESTRATOR_URL || 'http://localhost:8000';
    const response = await fetch(`${orchestratorUrl}/ultra-fast`, {
      method: 'POST',
      headers: {
        'Content-Type': 'application/json',
      },
      body: JSON.stringify(body),
    });
    
    if (!response.ok) {
      throw new Error(`Orchestrator responded with ${response.status}`);
    }
    
    const data = await response.json();
    
    // Convert to streaming format expected by frontend
    const stream = new ReadableStream({
      start(controller) {
        // Send text response
        const textData = JSON.stringify({ type: 'text', text: data.response_text });
        controller.enqueue(`data: ${textData}\n\n`);
        
        // Send audio response if available (as WAV data)
        if (data.audio_data) {
          const audioData = JSON.stringify({ type: 'wav_audio', data: data.audio_data });
          controller.enqueue(`data: ${audioData}\n\n`);
        }
        
        // Send completion
        const completeData = JSON.stringify({ 
          type: 'complete', 
          latency_ms: data.latency_ms 
        });
        controller.enqueue(`data: ${completeData}\n\n`);
        
        controller.close();
      }
    });
    
    return new NextResponse(stream, {
      headers: {
        'Content-Type': 'text/event-stream',
        'Cache-Control': 'no-cache',
        'Connection': 'keep-alive',
        'Access-Control-Allow-Origin': '*',
        'Access-Control-Allow-Headers': 'Cache-Control'
      }
    });
    
  } catch (error) {
    console.error('Ultra-fast API error:', error);
    return NextResponse.json(
      { error: 'Failed to process request' },
      { status: 500 }
    );
  }
}