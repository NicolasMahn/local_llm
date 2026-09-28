# Agents

## Every crash becomes a protection

When anything here fails (a model, the gateway, the GPU, the desktop), fixing the symptom is only half the job:

1. **Diagnose from evidence.** Read the container logs (`./llm logs NAME`), the kernel log (`journalctl -k -b`), the monitor's snapshot (`~/.local/state/local-llm/stats.json`) and `./llm status`. Name the root cause, not just what broke.
2. **Remove the cause** so it cannot happen again: a setting in `llm`, a limit, a refusal in `./llm up` before anything starts.
3. **Make it visible before it hurts next time.** Add a check to `health.py`, so the problem shows in `./llm status`, the panel and the chat page, and the gateway can answer requests with the reason instead of failing.
4. **Test the check** against the real failure or a faithful stand-in, and add the case to "When things go wrong" in the README.

A failure that is fixed but would go unnoticed if it came back is not done.
