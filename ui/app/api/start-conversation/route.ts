import { NextResponse } from 'next/server';

export async function POST() {
  try {
    const response = await fetch(`${process.env.ORCHESTRATOR_URL}/start-conversation`, {
      method: 'POST',
    });

    if (!response.ok) {
      throw new Error(`HTTP error! status: ${response.status}`);
    }

    const data = await response.json();
    return NextResponse.json(data);
  } catch (error) {
    console.error('Failed to start conversation:', error);
    return NextResponse.json({ error: 'Failed to start conversation' }, { status: 500 });
  }
}
