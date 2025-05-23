BeatDebate — Project Scope (AgentX 2025)

⸻

1 . Objective

Build a chat-first recommender that helps music lovers uncover under-the-radar tracks and indie artists. It uses (a) a conversational Preference-Elicitation Agent to learn a listener’s tastes and (b) a Multi-Agent Debate loop that pits several “advocate” agents—and a critic—against each other. A Judge Agent delivers the top-3 songs plus concise “why” explanations within ≤ 6 seconds in Discord and on a Hugging Face Space demo.


⸻

3 . In-Scope Functionality
	1.	Conversational onboarding
	•	4–6 quick questions (genre, mood, era, tolerance for explicit lyrics, etc.).
	•	Produces a JSON taste profile shared with downstream agents.
	2.	Candidate retrieval
	•	Pull 40–60 tracks per request from Million Playlist Dataset + FMA + public Spotify API.
	•	Embed tracks (CLAP audio + sentence-transformer tag vectors) and filter by region availability & explicit-content setting.
	3.	Multi-Agent Debate core
	•	Advocate Agents (k ≈ 4) each champion a different track.
	•	Critic Agent flags duplicates, popularity bias, policy risks.
	•	Judge Agent ranks, diversity-balances, and writes explanations.
	4.	Response packaging
	•	Discord bot and Hugging Face Space both return:

{
  "title": "...",
  "artist": "...",
  "preview_url": "...",
  "why": "Two-sentence rationale"
}


	5.	Feedback loop
	•	👍 / 👎 buttons log to Supabase; next session re-weights advocate priors.
	6.	Safety & licensing guard-rails
	•	OpenAI Moderation or equivalent on all user prompts.
	•	Final Judge pings Spotify track endpoint; drops any unavailable or DMCA-flagged items.

⸻

4 . Explicitly Out-of-Scope (for AgentX demo)
	•	Continuous playlist-generation (>3 tracks).
	•	Mobile native app.
	•	Paid licensing of full-length audio—demo serves 30-second previews only.
	•	Real-time collaborative listening rooms.

⸻

5 . Technical Constraints & Choices
	•	LLM: GPT-4o (primary), Mixtral-8×22B fallback.
	•	Agent framework: LangGraph on top of AutoGen.
	•	Vector store: ChromaDB with 768-d embeddings.
	•	Back-end: FastAPI + Redis latency cache.
	•	Infra: 1× A10 GPU (Lambda credits) + small CPU instance for Discord bot.

⸻

6 . Work-Breakdown & Timeline (today = May 20 → deadline = May 31)


May 31	Submit 7-page paper or 20-slide deck + live demo link	PM



⸻

8 . Deliverables to AgentX
	1.	Working demo: Discord bot invite + HF Space.
	2.	Code repo (MIT/CC-BY-NC) with Docker + instructions.
	4.	Safety checklist & red-team transcript appendix.
	5.	Pitch deck (or 7-page research paper).

⸻

9 . Post-Hackathon Stretch Goals (nice-to-have)
	•	Full 30-song playlist builder.
	•	Spotify “Add to Library” OAuth integration.
	•	Visual “taste radar” graph for explanations.
	•	Twitch extension that chats recommendations live on stream.

⸻

This scope is intentionally tight so you can reach a polished vertical slice by May 31 while showcasing genuine agentic novelty. Let me know if you’d like deeper detail on any subsystem or need to resize the timeline!