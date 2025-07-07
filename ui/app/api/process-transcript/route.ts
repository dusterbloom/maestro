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
      // 🔄 FALLBACK MODE - Use streaming endpoint for non-streaming requests too
      console.log('🔄 Using streaming endpoint in non-streaming mode');
      
      const response = await fetch(`${orchestratorUrl}/ultra-fast-stream`, {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify(body),
      });
      
      if (!response.ok) {
        throw new Error(`Orchestrator error: ${response.status}`);
      }
      
      // Parse SSE response to extract audio data
      if (!response.body) {
        throw new Error('No response stream available');
      }
      
      const reader = response.body.getReader();
      const decoder = new TextDecoder();
      let buffer = '';
      let audioData = '';
      let responseText = '';
      
      while (true) {
        const { done, value } = await reader.read();
        
        if (done) break;
        
        if (value) {
          buffer += decoder.decode(value, { stream: true });
          
          // Process complete lines
          const lines = buffer.split('\n');
          buffer = lines.pop() || ''; // Keep incomplete line
          
          for (const line of lines) {
            if (line.startsWith('data: ')) {
              try {
                const data = JSON.parse(line.slice(6));
                
                if (data.type === 'sentence_audio') {
                  audioData = data.audio_data;
                  responseText = data.text;
                } else if (data.type === 'error') {
                  throw new Error(data.message);
                }
              } catch (e) {
                console.error('Failed to parse SSE data:', e);
              }
            }
          }
        }
      }
      
      if (audioData) {
        return NextResponse.json({
          type: 'wav_audio',
          data: audioData,
          response_text: responseText,
          latency_ms: 0
        });
      } else {
        return NextResponse.json({
          type: 'error',
          message: 'No audio data received from TTS service',
          response_text: responseText || 'Error occurred'
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