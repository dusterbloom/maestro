import { NextResponse } from 'next/server';

export async function POST(request: Request) {
  try {
    const body = await request.json();
    
    // Reduce audio data size to prevent massive JSON parsing issues
    if (body.audio_data && body.audio_data.length > 100000) {
      console.warn(`Large audio data detected: ${body.audio_data.length} chars, truncating for performance`);
      // Keep only last portion of audio data (roughly 2 seconds instead of 10)
      body.audio_data = body.audio_data.slice(-20000);
    }
    
    const response = await fetch(`${process.env.ORCHESTRATOR_URL}/ultra-fast-stream`, {
      method: 'POST',
      headers: {
        'Content-Type': 'application/json',
      },
      body: JSON.stringify(body),
    });

    if (!response.ok) {
      throw new Error(`HTTP error! status: ${response.status}`);
    }

    // Add robust JSON parsing with fallback
    let data;
    try {
      const responseText = await response.text();
      data = JSON.parse(responseText);
    } catch (parseError) {
      console.error('JSON parsing failed:', parseError);
      // Return a basic success response if JSON parsing fails
      return NextResponse.json({ 
        error: 'Response parsing failed', 
        message: 'Request processed but response was malformed' 
      }, { status: 200 });
    }
    
    return NextResponse.json(data);
  } catch (error) {
    console.error('Failed to process transcript:', error);
    return NextResponse.json({ error: 'Failed to process transcript' }, { status: 500 });
  }
}
