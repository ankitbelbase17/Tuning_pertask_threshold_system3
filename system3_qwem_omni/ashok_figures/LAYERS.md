# Figure 1 — layer plan

Build one layer at a time. **Lock it before starting the next. Never go back.**

Two rules that hold across every layer:

- **Roles, not implementations.** *Memory*, not KV cache. *Perception*, not vision encoder.
  *Deliberation*, not LLM. If a label names a technology or a number, it is method and it
  does not belong in this figure.
- **The erasure test.** Turning a layer off must leave a simpler figure that is still true.
  If removing a layer makes the figure wrong rather than just plainer, the layers are
  tangled and the split is in the wrong place.

---

## Layer 0 — Frame

The outer boundary of the system, and nothing else. Page proportions, one shape, the line
between *inside the system* and *the world*.

**Excludes:** everything.
**Locked when:** you can say in one sentence what is inside and what is outside.

## Layer 1 — Inputs and outputs

The interface, and only the interface. Nothing internal.

- **In:** an unbounded stream that arrives whether or not the system is ready, plus a
  standing goal from the user, given once.
- **Out:** discrete utterances, at moments the system chooses.

The asymmetry *is* the problem statement: **input is continuous and not under your control;
output is sparse and entirely under your control.** Every hard question in the paper —
when to speak, what to keep, what to look at — falls out of that one mismatch.

**Excludes:** any internal component, any arrow that does not cross the boundary.
**Locked when:** a reader seeing only this layer can state the problem being solved.

## Layer 2 — Data flow

The forward, irreversible path. Four roles:

| role | what it does |
|---|---|
| **Perception** | turns arriving signal into representation |
| **Memory** | accumulates it; bounded, so it must forget |
| **Deliberation** | reads memory and decides |
| **Expression** | emits to the user |

Solid arrows only, all carrying data, all pointing forward.

**Excludes:** control, feedback, anything dashed, anything that points backwards.
**Locked when:** this layer alone describes *every* streaming omni model, including the
baselines you compare against. It should contain nothing anyone would argue with.

## Layer 3 — Control

**This layer is the contribution.** One new object — the **policy** the system writes for
itself — plus dashed edges from it to the layer-2 roles it governs:

- to Perception — how much to take in, and from where
- to Memory — what to retain, what to let go
- to Deliberation — when to think next, and what to ask itself
- to Expression — when speaking is warranted

And one edge *back*: Deliberation authors the policy. That closed curve is the paper.

**Excludes:** new components other than the policy artifact. Control does not add machinery;
it re-aims machinery that already exists.
**Locked when:** deleting this layer leaves an ordinary streaming model. That deletion is
your contribution statement — if it doesn't degrade to the baseline, something in layer 2
is actually part of the contribution and is in the wrong layer.

## Layer 4 — Time

The axis that makes it *foresight* rather than feedback.

- past · now · not-yet
- the policy is written **now** and governs perception **later**

Without this layer the control loop reads as reaction. With it, it reads as planning — which
is the title of the paper.

**Excludes:** anything that isn't a temporal mark.
**Locked when:** you can point at the part of the figure that hasn't happened yet.

## Layer 5 — Annotation

Labels, legend, numbered call-outs keyed to the caption. Last, and thinnest.

**Locked when:** removing this layer entirely still leaves the structure readable.

---

## Working notes

- Name the Figma layers exactly as above and lock each on completion.
- Test by toggling: each layer alone, then cumulative 0–1, 0–2, 0–3… Every prefix should be
  a coherent figure. If a prefix looks broken, the order is wrong.
- Keep the concrete version (`iter06` — football example, KV cache, fps values) as a
  **second figure**: the instantiation. It is good work and it answers "did you build it?" —
  it just isn't Figure 1.
