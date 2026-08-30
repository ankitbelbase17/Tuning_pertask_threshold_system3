#!/usr/bin/env bash
# login_chain.sh -- the second fleet: a self-chaining sweep on the ln003 login node.
#
# WHY A SECOND FLEET AT ALL. debug-qos allows ONE running job per user, so the
# debug chain cannot be widened. The login node's 4 GH200s are idle capacity that
# needs no allocation. They are also SHARED with every other user on ln003
# (18 logged in at last count, 3 of the 4 GPUs already carrying ~26 GB), and CSCS
# reaps heavy login-node processes without warning -- so this fleet is treated as
# strictly best-effort. Everything it banks is banked per sample; anything it
# loses the debug chain simply re-claims.
#
# WHY IT IS SAFE TO RUN BOTH. Both fleets pull from the same worklist through the
# same atomic claim (lib/worklist.py), and completion is a property of the CELL,
# not of a lane, so neither can duplicate or corrupt the other's work.
#
#   start:  nohup bash bin/login_chain.sh > logs/login_chain.out 2>&1 &
#   stop:   touch STOP_LOGIN     (checked between generations and between units)
#   status: bash bin/status.sh
set -uo pipefail
export THR_ROOT=/iopsstor/scratch/cscs/dthapa/system3_qwem_omni_8_28/omni_thr_fit
source "$THR_ROOT/env.sh"

GEN_MIN=${GEN_MIN:-90}                 # minutes per generation
MIN_FREE_MIB=${MIN_FREE_MIB:-45000}    # a GH200 is 97 GB; the model needs ~21 GB
                                       # of weights plus activations, and we must
                                       # not evict a co-tenant's job
LANE_BASE=${LANE_BASE:-100}            # debug uses 0..15; login uses 100+
MAX_GEN=${MAX_GEN:-200}
STOP=$THR_ROOT/STOP_LOGIN

echo "[login] chain start $(date) on $(hostname), gen=${GEN_MIN}min lane_base=$LANE_BASE"

for (( g = 1; g <= MAX_GEN; g++ )); do
  [ -f "$STOP" ] && { echo "[login] STOP_LOGIN present -- standing down"; break; }

  ST=$(timeout 300 "$PY" "$THR_ROOT/lib/chain_state.py" --nodes 1 2>/dev/null)
  if [ -z "$ST" ]; then
    echo "[login] gen $g: chain_state unreadable, retrying in 60s"; sleep 60; continue
  fi
  COMPLETE=$(sed -n 's/^COMPLETE=//p' <<<"$ST")
  DONE=$(sed -n 's/^DONE=//p' <<<"$ST"); TARGET=$(sed -n 's/^TARGET=//p' <<<"$ST")
  CD=$(sed -n 's/^CELLS_DONE=//p' <<<"$ST"); CT=$(sed -n 's/^CELLS_TOTAL=//p' <<<"$ST")
  if [ "$COMPLETE" = "1" ]; then
    echo "[login] worklist COMPLETE ($DONE/$TARGET) -- standing down"; break
  fi

  # Which GPUs can we actually use right now? Co-tenants come and go, so this is
  # re-asked every generation rather than fixed at start.
  GPUS=$(nvidia-smi --query-gpu=index,memory.free --format=csv,noheader,nounits 2>/dev/null \
         | awk -v m="$MIN_FREE_MIB" -F', *' '$2 >= m {print $1}' | paste -sd, -)

  # GPU ALLOWLIST -- re-read from a FILE every generation, not fixed at launch.
  #
  # This login node is shared, and on 2026-08-29 the owner of this account needed
  # two of its four GH200s back for a training run (which had already taken GPUs
  # 1 and 2 via CUDA_VISIBLE_DEVICES=2,1). Free-memory discovery alone cannot see
  # that: a job that has only just started holds ~556 MiB, so its GPUs still look
  # idle and this fleet would take them straight back at the next generation.
  #
  # It is a file rather than an env var precisely so it can be changed while the
  # chain is running -- editing the script itself is not safe (bash reads a script
  # incrementally, so a running invocation can be corrupted mid-loop), and an env
  # var would need a restart, which costs every lane its in-flight unit.
  #   echo 0,3 > $THR_ROOT/LOGIN_GPUS      # yield 1 and 2 at the next generation
  #   rm       $THR_ROOT/LOGIN_GPUS        # go back to using whatever is free
  ALLOW=$(cat "$THR_ROOT/LOGIN_GPUS" 2>/dev/null | tr -d ' \n')
  ALLOW=${LOGIN_GPUS:-$ALLOW}
  if [ -n "$ALLOW" ]; then
    GPUS=$(tr ',' '\n' <<<"$GPUS" | grep -Fxf <(tr ',' '\n' <<<"$ALLOW") | paste -sd, -)
    echo "[login] gen $g: GPU allowlist [$ALLOW] -> usable [$GPUS]"
  fi
  if [ -z "$GPUS" ]; then
    echo "[login] gen $g: no GPU with ${MIN_FREE_MIB} MiB free -- waiting 5 min"
    sleep 300; continue
  fi
  echo "[login] gen $g: $DONE/$TARGET samples, cells $CD/$CT, using GPUs [$GPUS]"

  DEADLINE=$(( $(date +%s) + GEN_MIN * 60 ))
  PIDS=()
  i=0
  for gpu in ${GPUS//,/ }; do
    LANE=$(( LANE_BASE + i ))
    # WINDOW-FIT CAP ON THE LOGIN FLEET. Default is now 0 = NO CAP.
    #
    # It was 300 s, for the two reasons below, and that was right while the node
    # was shared. It stopped being right the moment the debug chain's 4-node band
    # emptied: from then on every remaining sample was LONGER than 300 s, so each
    # login lane found nothing actionable, exited at once, and login_chain spun a
    # fresh 90-minute generation every ~60 seconds while banking nothing --
    # generations 30-34 all "finished" inside a minute. A cap that silently turns
    # a whole fleet into a no-op is worse than no fleet.
    #
    # The login generation is 90 minutes; the longest video in the benchmark is
    # 594.6 s, which at the measured xRT of ~2.25 needs ~22 minutes. The long tail
    # fits here comfortably -- and the tail is now exactly what needs clearing,
    # since debug can only reach it by giving up GPUs for window length.
    # Override with LOGIN_MAX_DUR if the node becomes contended again.
    #
    # The two original reasons, kept because they still apply if that happens:
    # (a) Throughput: this node is shared, and a contended GH200 was measured at
    #     an effective xRT far above the 2.7 a dedicated one gives -- long videos
    #     are a bad bet here and belong to the debug chain's reshaped generations.
    # (b) Blast radius: worker.sh bounds a unit by the generation deadline, but
    #     heartbeat.sh keeps a claim fresh while a lane is HUNG (not dead), so a
    #     hang holds its unit for the rest of the generation. Capping duration
    #     keeps that loss small and keeps the long tail out of the fleet most
    #     likely to stall.
    CUDA_VISIBLE_DEVICES=$gpu nohup bash "$THR_ROOT/bin/worker.sh" \
        "$LANE" "$DEADLINE" "${LOGIN_MAX_DUR:-0}" \
        >> "$THR_ROOT/logs/login_lane_${LANE}.out" 2>&1 &
    PIDS+=($!)
    i=$(( i + 1 ))
    sleep 20      # stagger the model loads so 4 lanes do not hit the FS at once
  done

  # wait out the generation, but poll for STOP so a stop request is honoured
  # within a minute rather than after 90.
  while :; do
    alive=0
    for p in "${PIDS[@]}"; do kill -0 "$p" 2>/dev/null && alive=1; done
    [ "$alive" = "0" ] && break
    if [ -f "$STOP" ]; then
      echo "[login] STOP_LOGIN -- terminating gen $g lanes"
      for p in "${PIDS[@]}"; do kill "$p" 2>/dev/null; done
      break
    fi
    sleep 60
  done
  echo "[login] gen $g finished $(date +%H:%M:%S)"
done
echo "[login] chain exit $(date)"
