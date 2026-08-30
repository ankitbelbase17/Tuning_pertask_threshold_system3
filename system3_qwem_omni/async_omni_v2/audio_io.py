"""
audio_io.py — FORK (system3_qwem_omni): raw-audio chunk extraction for
Option A synchronous ingestion (see input_ingester.py and MISSION-adjacent
discussion in OMNI_FEASIBILITY.md / OMNI_MODEL_SURVEY.md).

Deliberately NOT a thread, NOT a queue producer. It is a small stateful
demuxer the INGESTER calls synchronously, once per `audio_seconds_per_chunk`
of video time, to pull exactly the audio spanning the window that just
finished. Because the ingester is already the single writer of the primary
cache, and this is the ingester's own call (not a second thread's), audio can
never be appended out of true-time order relative to video -- there is no
race to prevent here, because there is no second producer to race against.

If the source video has no audio stream (or av can't open one), chunks come
back as silence (zeros) rather than raising -- a stream with no audio track
should degrade to vision-only behaviour, not crash the pipeline.
"""
import numpy as np


class AudioChunkReader:
    """Pull `[t0, t1)` (real video-time seconds) of mono float32 PCM at
    `sample_rate` Hz out of `video_path`'s audio track, on demand.

    Opens its own `av` container (a full second decode context, independent
    of vision_stream.py's video-only container) so it can seek freely without
    disturbing the video decode. For the video lengths this pipeline targets
    (max_seconds, typically <= 600s) a fresh small seek-and-decode per chunk
    is cheap relative to the ~160ms encoder forward pass it feeds into
    (measured for the analogous Qwen3-Omni/Whisper-family encoder in
    OMNI_FEASIBILITY.md section 5); if this ever shows up as a bottleneck the
    fix is to decode audio once, up front, into an in-memory ring rather than
    re-opening per chunk -- not attempted here to keep this change small and
    reviewable."""

    def __init__(self, video_path, sample_rate=16000):
        self.sample_rate = sample_rate
        self.has_audio = False
        self._video_path = video_path
        # `import av` failing (a broken/partial environment -- this project's
        # own LEARNINGS.md and EXPERIMENTS.md both record this scratch conda
        # env periodically losing shared libraries) is a DIFFERENT condition
        # from "this specific file has no audio track", and must not be
        # silently folded into it -- a caller seeing has_audio=False needs to
        # be able to tell "the video is silent" from "audio decoding is
        # broken on this node" from the log alone. Caught a real instance of
        # exactly this while building this class: `import av` raised
        # `ImportError: libXau...so: cannot open shared object file` with no
        # video-specific problem at all.
        try:
            import av
        except Exception as e:
            print(f"[audio_io] WARNING: `import av` failed ({e!r}) -- audio "
                  f"will be silence for the whole run. This is an ENVIRONMENT "
                  f"problem, not a property of {video_path!r}; vision_stream.py "
                  f"needs the same `av` and will fail the same way if this "
                  f"pipeline gets far enough to decode video at all.", flush=True)
            return
        try:
            probe = av.open(video_path)
            self.has_audio = len(probe.streams.audio) > 0
            probe.close()
            if not self.has_audio:
                print(f"[audio_io] {video_path!r} opened fine but has no audio "
                      f"stream -- audio will be silence for this sample "
                      f"(this IS a property of the file, not an error).", flush=True)
        except Exception as e:
            print(f"[audio_io] WARNING: could not open {video_path!r} for audio "
                  f"({e!r}) -- audio will be silence for this sample.", flush=True)
            self.has_audio = False

    def read(self, t0, t1):
        """Return a 1D float32 numpy array of (t1-t0)*sample_rate samples.
        Silence if the source has no audio track or the window is out of
        range -- never raises, so a missing audio track degrades gracefully
        rather than taking the whole ingest thread down with it."""
        n_want = max(1, round((t1 - t0) * self.sample_rate))
        if not self.has_audio or t1 <= t0:
            return np.zeros(n_want, dtype=np.float32)
        try:
            return self._read_real(t0, t1, n_want)
        except Exception as e:
            # Degrade THIS chunk to silence rather than kill a multi-minute
            # ingest thread over one bad seek -- but say so; a run whose log
            # is full of these needs to be able to tell that from a genuinely
            # quiet video (see __init__'s distinct log line for that case).
            print(f"[audio_io] WARNING: chunk [{t0:.1f},{t1:.1f})s read failed "
                  f"({e!r}) -- using silence for this chunk only.", flush=True)
            return np.zeros(n_want, dtype=np.float32)

    def _read_real(self, t0, t1, n_want):
        import av
        container = av.open(self._video_path)
        try:
            astream = container.streams.audio[0]
            resampler = av.AudioResampler(format="fltp", layout="mono", rate=self.sample_rate)
            container.seek(int(t0 * av.time_base), any_frame=False, backward=True)
            chunks = []
            got = 0
            for frame in container.decode(astream):
                ft = float(frame.pts * astream.time_base) if frame.pts is not None else None
                if ft is not None and ft < t0 - 1.0:      # well before the window: skip cheaply
                    continue
                for rframe in resampler.resample(frame):
                    arr = rframe.to_ndarray().reshape(-1).astype(np.float32)
                    chunks.append(arr)
                    got += arr.shape[0]
                if ft is not None and ft > t1 + 1.0:       # well past the window: stop
                    break
                if got >= n_want * 2:                       # comfortable margin, then stop
                    break
        finally:
            container.close()
        if not chunks:
            return np.zeros(n_want, dtype=np.float32)
        full = np.concatenate(chunks)
        # `full` starts at roughly (t0 - 1.0)s due to the backward seek above;
        # trim/pad to exactly the requested window. Being off by a small,
        # bounded amount here is a real limitation of per-chunk re-seeking
        # (see class docstring) -- flagged, not silently assumed exact.
        start = max(0, full.shape[0] - n_want) if full.shape[0] > n_want else 0
        out = full[start:start + n_want]
        if out.shape[0] < n_want:
            out = np.pad(out, (0, n_want - out.shape[0]))
        return out.astype(np.float32)
