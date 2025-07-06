# Verified Implementation Plan: "Magical" Speaker Identification

This document outlines the finalized, voice-first strategy for seamless speaker enrollment and recognition. The goal is to create a "magical" user experience where the system learns and remembers users conversationally, while maintaining a robust, stateless, and event-driven architecture.

## 🎯 Core Goal: The "Magic" of Voice-First Identity

1.  **First Visit:** A new user speaks. The system automatically detects them, asks for their name conversationally using its own voice, and remembers it.
2.  **Return Visit:** The user speaks again. The system instantly recognizes them by name without any login or manual steps.

## 🔧 Architecture: Stateless Core with Intelligent Memory

The Orchestrator remains **100% stateless**. State management is delegated to specialized services, ensuring scalability and resilience.

-   **Orchestrator:** The stateless coordinator. It manages the workflow between services but stores no user data itself.
-   **ChromaDB:** The long-term **"Voice Memory."** It stores and indexes all speaker voice embeddings for fast, similarity-based lookups.
-   **Redis:** The **"Profile Store."** It stores user metadata (e.g., `user_id`, `name`, `status`) and maps the voice embeddings in ChromaDB to user profiles.
-   **Ollama (LLM):** The **"Conversational Onboarder."** It drives the voice-first user interaction, asking for and extracting the user's name.
-   **Diglett:** The **"Voiceprint Generator."** Used to create the initial, high-quality voice embedding from a user's first few seconds of speech via its `/embed` endpoint.

---

## ✨ The "Magical" Workflow: A Step-by-Step Guide

### Step 1: A New Voice Appears

1.  The UI streams a user's first audio to the Orchestrator.
2.  The Orchestrator sends this initial audio snippet (the first 3-5 seconds) to **Diglett's `/embed` endpoint** to generate a voiceprint (embedding).

### Step 2: The "Who Is This?" Moment

1.  The Orchestrator queries **ChromaDB** with the new embedding: "Find the closest voice match."
2.  **Scenario A: Match Found (Returning User)**
    -   ChromaDB returns a matching `user_id` with high confidence.
    -   The Orchestrator looks up the `user_id` in **Redis** to retrieve the user's name (e.g., "Alex").
    -   The Orchestrator sends a WebSocket event to the UI: `{ "event": "speaker.identified", "name": "Alex", "status": "active" }`. The magic is complete.
3.  **Scenario B: No Match Found (New User)**
    -   ChromaDB finds no close match.
    -   **Automatic Enrollment:** The Orchestrator generates a new `user_id`, saves the embedding to ChromaDB, and creates a profile in Redis with a temporary status: `{ "name": "Speaker 1", "status": "pending_naming" }`.

### Step 3: The LLM-Powered Conversation

This is where the system becomes truly voice-first.

1.  The Orchestrator detects the `pending_naming` status and **injects a system command** into the prompt for the LLM.
2.  **Orchestrator to LLM:**
    ```
    SYSTEM_PROMPT: "A new user is speaking. Their temporary name is 'Speaker 1'. Your immediate task is to ask them what you should call them. Be friendly and direct."
    USER_TRANSCRIPT: "so what's the weather like today"
    ```
3.  **LLM Generates a Response:** The LLM crafts a natural reply that is sent to the TTS service and played to the user.
    > "The weather is pleasant today. I don't believe we've met, what should I call you?"

### Step 4: Voice-Only Name Capture

1.  The user replies naturally: "You can call me Alex."
2.  The Orchestrator receives the transcript and, knowing the user's status is still `pending_naming`, sends another targeted request to the LLM.
3.  **Orchestrator to LLM:**
    ```
    SYSTEM_PROMPT: "The user is responding to your request for their name. Extract ONLY their name from the transcript. Respond with just the name."
    USER_TRANSCRIPT: "You can call me Alex"
    ```
4.  **LLM Extracts the Name:** The LLM returns a single word: `Alex`.

### Step 5: Finalizing the Voice Profile

1.  The Orchestrator receives "Alex" from the LLM and updates the user's profile in **Redis**, setting the name and changing the status to `active`.
2.  **Confirmation (Optional but Recommended):** The Orchestrator can instruct the LLM to provide a final confirmation to the user:
    > "Great, I'll remember that. Nice to meet you, Alex!"

## 💻 UI Implementation

-   **Primary State:** The UI simply displays the name it receives from the `speaker.identified` event.
-   **Fallback Mechanism:** If the LLM interaction fails, or for accessibility, a small "Set Name" button can appear next to "Speaker 1". Clicking this reveals a text input that fires a `speaker.claim` event, allowing for manual name entry.
