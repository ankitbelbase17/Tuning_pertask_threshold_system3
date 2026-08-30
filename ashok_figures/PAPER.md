# Foresight: Planning Future Perception in Streaming Vision-Language Models

Working draft of the argument. Edit freely — this is a shared scratchpad, not a decision.

---

## The one-sentence idea

> **A config written for the future is a forecast.** When the model writes
> *"check again in 1 s, look densely, and ask: did a goal happen?"* it has predicted that
> something is about to matter and pre-allocated its perception to it — without being
> trained to forecast anything.

Everything in the paper hangs off that sentence.

## The unified mechanism

**The model writes a perception budget for its future self.**

Pruning and densification are the *same knob* pointed in two directions:

| direction | what it means |
|---|---|
| spend more | look here, look often, look closely — something is developing |
| spend less | drop these tokens, skip these frames — nothing here matters |

So "input pruning" and "planning" are not two contributions. They are one contribution seen
from two ends: **allocation of a finite perception budget over a future the model has not
seen yet.**

## Why it can only be a streaming method

Offline video QA has no future — every frame is already available, so there is nothing to
plan and nothing to allocate. Perception planning is *only definable* when perception is
incremental and irreversible.

This is a scoping **strength**, not a limitation. It defines a problem class that exists
only in streaming, and it is why the method has no offline analogue to be compared against.

## Why training-free is not a compromise

The forecast is expressed as a **config** — a short, structured, natural-language-adjacent
artifact. Writing one is a task a frozen instruction-following model can already do. There
is nothing to learn because the capability is already there; what was missing was a
*place to put it* and a *loop that honours it*.

Contrast: the rival trains a binary control token end-to-end. We ask, and read.

## The structure

- **Two roles.** One ingests the stream and remembers. One thinks, and writes the config.
- **The config file** is the interface between them, and between now and later. It is a
  first-class, model-written, persistent object — not a hyperparameter.
- **Time** is the axis everything lives on: the config written at *t* governs the perception
  that happens at *t+1*.

## Draft abstract (v0 — for shredding)

> Streaming vision-language models perceive blindly. They sample frames at a fixed rate and
> encode every region with equal effort, spending the same compute on a static corridor as
> on the one second that matters — and they cannot look harder when it counts, because
> nothing in the system decides how to look.
>
> We argue that perception should be *planned*, and that a frozen model can already plan it.
> We introduce **Foresight**, a training-free architecture in which a streaming
> vision-language model writes a **config for its own future perception**: where to attend,
> how densely to sample, what to prune, when to look again, and what question to ask itself
> when it does. Because the config is written before the relevant content arrives, it is a
> forecast — the model commits perceptual resources to an event it has not yet seen.
>
> Foresight requires no gradients: the planning capability is elicited in context from a
> frozen model, and the config is the only thing that changes over time. The method is
> defined only for streaming input — offline video understanding has no future to plan for —
> and we evaluate it on [BENCHMARK], where [RESULT].

## The contributions, restated

1. **Perception planning as a first-class problem** in streaming video-language models, and
   the observation that it has no offline counterpart.
2. **The config-as-forecast mechanism:** a persistent, model-written control artifact that
   allocates future perception, obtained from a frozen model in context.
3. **Input pruning under a planned budget** — spending less where the plan says nothing
   matters, more where it says something will.
4. *(minor, explicitly not our problem)* KV-cache compaction/summarisation as the memory
   substrate that lets the plan persist over an unbounded stream.

## Open framing decision

`MISSION.md` §2 currently sells a different paper:
*"A frozen VLM already knows when to speak. Proactivity is an architecture problem."*

That is **when to speak**. The title is **what to perceive next**. These are two papers, and
the second is the more distinctive one — *when to speak* is the rival's own framing done
training-free, while *planning perception* is an axis nobody is working on.

Recommendation: make foresight the thesis, and demote "knows when to speak" to a special
case of it — deciding to check now is one entry in the perception plan.

## Planned figures

1. Benchmark performance
2. **Circuit diagram** (the architecture — the one we are designing)
3. State diagram (how the config evolves over the stream)
