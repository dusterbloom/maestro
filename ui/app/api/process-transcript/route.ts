import { NextRequest, NextResponse } from 'next/server';

export async function POST(request: NextRequest) {
  try {
    const body = await request.json();
    
    // Check if real-time streaming is requested
    const useStreaming = request.headers.get('X-Use-Streaming') === 'true';
    
    const orchestratorUrl = process.env.ORCHESTRATOR_URL || 'http://localhost:8000';
    
    if (useStreaming) {
      // 🚀 REAL-TIME STREAMING MODE
      console.log('🚀 Using real-time streaming endpoint');
      
      const response = await fetch(`${orchestratorUrl}/ultra-fast-stream`, {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify(body),
      });
      
      if (!response.ok) {
        throw new Error(`Orchestrator streaming error: ${response.status}`);
      }
      
      // Return the Server-Sent Events streaming response
      return new Response(response.body, {
        headers: {
          'Content-Type': 'text/event-stream',
          'Cache-Control': 'no-cache',
          'Connection': 'keep-alive',
          'Access-Control-Allow-Origin': '*',
          'Access-Control-Allow-Headers': 'Cache-Control'
        }
      });
    } else {
      // 🔄 BATCH MODE (original behavior)
      console.log('🔄 Using batch mode endpoint');
      
      const response = await fetch(`${orchestratorUrl}/ultra-fast`, {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify(body),
      });
      
      if (!response.ok) {
        throw new Error(`Orchestrator error: ${response.status}`);
      }
      
      const data = await response.json();
      
      // Transform the response to match expected format
      if (data.audio_data) {
        const transformedData = {
          type: 'wav_audio',
          data: data.audio_data,
          response_text: data.response_text,
          latency_ms: data.latency_ms
        };
        return NextResponse.json(transformedData);
      } else if (data.sentence_complete === false) {
        // Sentence not complete - this is normal, not an error
        return NextResponse.json({
          type: 'incomplete',
          message: 'Sentence not complete',
          response_text: data.response_text || '',
          latency_ms: data.latency_ms || 0
        });
      } else {
        // Return error if no audio data and sentence was supposed to be complete
        return NextResponse.json({
          type: 'error',
          message: 'No audio data received from TTS service',
          response_text: data.response_text || 'Error occurred'
        });
      }
    }
    
  } catch (error) {
    console.error('API error:', error);
    return NextResponse.json(
      { error: 'Failed to process transcript' },
      { status: 500 }
    );
  }
}