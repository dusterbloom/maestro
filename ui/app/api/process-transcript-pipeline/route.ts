import { NextRequest, NextResponse } from 'next/server';

export async function POST(request: NextRequest) {
  try {
    const body = await request.json();
    
    // Forward request to orchestrator ultra-low latency pipeline endpoint
    const orchestratorUrl = process.env.ORCHESTRATOR_URL || 'http://localhost:8000';
    const response = await fetch(`${orchestratorUrl}/process-transcript-pipeline`, {
      method: 'POST',
      headers: {
        'Content-Type': 'application/json',
      },
      body: JSON.stringify(body),
    });
    
    if (!response.ok) {
      throw new Error(`Orchestrator error: ${response.status}`);
    }
    
    // Return streaming response
    return new NextResponse(response.body, {
      status: 200,
      headers: {
        'Content-Type': 'text/event-stream',
        'Cache-Control': 'no-cache',
        'Connection': 'keep-alive',
        'Access-Control-Allow-Origin': '*',
        'Access-Control-Allow-Headers': 'Cache-Control'
      }
    });
    
  } catch (error) {
    console.error('Ultra-low latency pipeline API error:', error);
    return NextResponse.json(
      { error: 'Failed to process transcript through ultra-low latency pipeline' },
      { status: 500 }
    );
  }
}