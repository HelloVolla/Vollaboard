package com.volla.vollaboard.recognition.recognizers.sources

import android.os.Handler
import android.os.Looper
import android.util.Log
import androidx.lifecycle.LiveData
import androidx.lifecycle.MutableLiveData
import androidx.lifecycle.Observer
import com.volla.vollaboard.recognition.recognizers.Recognizer
import com.volla.vollaboard.recognition.recognizers.RecognizerSource
import com.volla.vollaboard.recognition.recognizers.RecognizerState
import com.volla.vollaboard.whisper.WhisperSplitEngine
import java.util.Locale
import java.util.concurrent.CountDownLatch
import java.util.concurrent.Executor
import java.util.concurrent.Executors
import java.util.concurrent.TimeUnit
import kotlin.math.sqrt

class WhisperLocal(
    private val modelDir: String,
    override val locale: Locale
) : RecognizerSource {

    private val stateMLD = MutableLiveData(RecognizerState.NONE)
    override val stateLD: LiveData<RecognizerState> get() = stateMLD

    private val engine = WhisperSplitEngine()
    private val whisperRecognizer = WhisperRecognizer(engine, locale)

    override val recognizer: Recognizer get() = whisperRecognizer
    override val closed: Boolean get() = !engine.isInitialized()
    override val addSpaces: Boolean get() = true
    override val errorMessage: Int get() = 0
    override val name: String get() = locale.displayName

    override fun initialize(executor: Executor, onLoaded: Observer<RecognizerSource?>) {
        stateMLD.postValue(RecognizerState.LOADING)
        executor.execute {
            try {
                engine.initialize(modelDir, locale.language)
                stateMLD.postValue(RecognizerState.READY)
                Handler(Looper.getMainLooper()).post { onLoaded.onChanged(this) }
            } catch (e: Exception) {
                Log.e(TAG, "Failed to initialise Whisper engine at $modelDir", e)
                stateMLD.postValue(RecognizerState.ERROR)
                Handler(Looper.getMainLooper()).post { onLoaded.onChanged(null) }
            }
        }
    }

    override fun close(freeRAM: Boolean) {
        if (freeRAM) engine.deinitialize()
    }

    // -------------------------------------------------------------------------

    /**
     * Buffers 16 kHz PCM audio from MySpeechService using simple energy-based
     * voice-activity detection (VAD): leading silence is dropped, speech is
     * accumulated from its onset, and the utterance is transcribed when the
     * speaker pauses.
     *
     * MySpeechService reads the microphone and calls acceptWaveForm on a
     * single thread in a tight loop; if that call blocks for as long as
     * Whisper inference takes (measured ~1-2.5s on device), the native
     * AudioRecord buffer — sized for only ~0.2s — silently drops everything
     * spoken during the stall. So transcription runs on a dedicated
     * background executor: acceptWaveForm only ever does cheap VAD
     * bookkeeping and returns immediately, keeping the capture loop reading
     * the microphone continuously.
     *
     * The model itself only ever sees a fixed window at a time (the mel
     * extractor's input length, read from the loaded model — [maxSamples]).
     * For utterances longer than that, each window is transcribed as soon as
     * the buffer fills, but the result is held rather than committed
     * immediately: MySpeechService types
     * whatever acceptWaveForm's "true" return causes it to fetch, and that
     * type is irreversible, so we can't fix up text after the fact the way a
     * redrawable streaming display could. Instead we carry the trailing ~0.3s
     * of raw audio into the next window (so a word split by the cut appears
     * whole in at least one window) and stitch each chunk's text onto the
     * held text by matching the overlapping words, only emitting the combined
     * result once a real pause ends the utterance. This is the standard
     * overlap-and-stitch technique used by streaming Whisper implementations.
     */
    private class WhisperRecognizer(
        private val engine: WhisperSplitEngine,
        override val locale: Locale?
    ) : Recognizer {

        override val sampleRate: Float = WhisperSplitEngine.SAMPLE_RATE.toFloat()

        // Runs every engine.transcribeBuffer() call, off the capture thread.
        // Single-threaded so chunks are transcribed and stitched in order.
        private val executor = Executors.newSingleThreadExecutor()

        // The model's actual window size, read from the loaded .tflite file
        // rather than assumed — a re-exported model with a longer window
        // (e.g. 5s) is picked up automatically. Resolved lazily on first use
        // (the mic capture loop that calls acceptWaveForm only ever starts
        // after engine.initialize() has already completed).
        private val maxSamples: Int by lazy { engine.getMelInputSamples() }

        // Speech is accumulated here (float [-1, 1]); capped at the model's window.
        // Owned by the capture thread only.
        private val buffer: FloatArray by lazy { FloatArray(maxSamples) }
        private var bufferPos = 0
        private var inSpeech = false
        private var silenceRun = 0          // consecutive silent samples while in speech
        private var lastResult = ""

        // Text held across a mid-utterance buffer split, not yet committed.
        // Touched only inside executor jobs — never from the capture thread.
        private var pendingText = ""

        // Cross-thread handoff for a finished, ready-to-commit utterance.
        // Single writer (executor), single reader (capture thread via
        // pollReady); a one-tick (~0.2s) delivery delay is harmless.
        @Volatile
        private var readyResult: String? = null

        override fun acceptWaveForm(buffer: ShortArray?, nread: Int): Boolean {
            if (buffer != null && engine.isInitialized()) {
                // Energy (RMS) of this incoming frame decides speech vs silence.
                var sumSq = 0.0
                for (i in 0 until nread) {
                    val v = buffer[i] / 32767f
                    sumSq += v.toDouble() * v
                }
                val frameRms = sqrt(sumSq / maxOf(1, nread)).toFloat()
                val isSpeech = frameRms >= SPEECH_RMS

                if (isSpeech) {
                    inSpeech = true
                    silenceRun = 0
                    appendFrame(buffer, nread)
                } else if (inSpeech) {
                    // Keep trailing silence so word endings aren't clipped, but use
                    // it to detect the end-of-utterance pause.
                    appendFrame(buffer, nread)
                    silenceRun += nread
                    if (silenceRun >= PAUSE_SAMPLES) flush()
                }
                // Otherwise: leading silence before any speech — ignore it.

                // Can't hold more than the model's window, but speech is still
                // ongoing (no pause yet) — hold the transcript, don't commit it.
                if (bufferPos >= maxSamples) splitFlush()
            }
            // Deliver whatever utterance finished transcribing in the
            // background since the last call, if any.
            return pollReady()
        }

        private fun pollReady(): Boolean {
            val r = readyResult ?: return false
            readyResult = null
            lastResult = r
            return true
        }

        /** Copies up to the remaining capacity of [buffer] into the speech buffer. */
        private fun appendFrame(src: ShortArray, nread: Int) {
            var i = 0
            while (i < nread && bufferPos < maxSamples) {
                buffer[bufferPos++] = src[i] / 32767f
                i++
            }
        }

        /**
         * A real pause ended the utterance: snapshot whatever's left and
         * hand it to the background executor, which transcribes it, stitches
         * it onto any text held from earlier splits, and makes the combined
         * result available via [readyResult]. Full VAD reset here on the
         * capture thread — this is a genuine boundary — but the transcript
         * itself isn't ready until the executor job runs.
         */
        private fun flush() {
            val len = bufferPos
            resetState()
            val chunk = if (len < MIN_SPEECH_SAMPLES) null else buffer.copyOf(len)

            executor.execute {
                val text = if (chunk != null) engine.transcribeBuffer(chunk) else ""
                val combined = stitch(pendingText, text)
                pendingText = ""
                if (combined.isNotEmpty()) readyResult = combined
            }
        }

        /**
         * The model's window filled but speech is still ongoing (no pause
         * yet): hand this window to the background executor and hold its
         * text rather than committing it, since MySpeechService types a
         * committed result immediately and it can't be revised afterwards.
         * Carry the trailing overlap of raw audio into the next window so a
         * word cut by this split still appears whole in (at least) one
         * window, and stitch the two chunks' text together once the next
         * one arrives.
         */
        private fun splitFlush() {
            val chunk = buffer.copyOf(bufferPos)
            val tail = chunk.copyOfRange(chunk.size - OVERLAP_SAMPLES, chunk.size)
            tail.copyInto(buffer)
            bufferPos = OVERLAP_SAMPLES
            silenceRun = 0
            // inSpeech stays true — we're mid-utterance, not at a real pause.

            executor.execute {
                val text = engine.transcribeBuffer(chunk)
                if (text.isNotEmpty()) pendingText = stitch(pendingText, text)
            }
        }

        /**
         * Merges consecutive, overlapping transcript chunks into one
         * continuous string. Because [splitFlush] carries shared audio into
         * the next window, [next]'s leading words should re-transcribe
         * roughly the same words [prev] ended with; find the longest such
         * overlap (compared loosely, ignoring case/punctuation) and drop the
         * duplicate. Falls back to plain concatenation if no overlap is
         * found — never worse than the un-stitched result.
         */
        private fun stitch(prev: String, next: String): String {
            if (prev.isEmpty()) return next
            if (next.isEmpty()) return prev

            val prevWords = prev.trim().split(Regex("\\s+"))
            val nextWords = next.trim().split(Regex("\\s+"))

            val maxOverlap = minOf(prevWords.size, nextWords.size)
            for (k in maxOverlap downTo 1) {
                val prevTail = prevWords.subList(prevWords.size - k, prevWords.size)
                val nextHead = nextWords.subList(0, k)
                if (prevTail.indices.all { normalizeWord(prevTail[it]) == normalizeWord(nextHead[it]) }) {
                    return (prevWords + nextWords.subList(k, nextWords.size)).joinToString(" ")
                }
            }
            return "$prev $next"
        }

        private fun normalizeWord(word: String): String =
            word.lowercase(Locale.ROOT).filter { it.isLetterOrDigit() }

        private fun resetState() {
            bufferPos = 0
            inSpeech = false
            silenceRun = 0
        }

        override fun getResult(): String {
            val r = lastResult
            lastResult = ""
            return r
        }

        override fun getPartialResult(): String = ""

        override fun getFinalResult(): String {
            // Flush any buffered speech when recognition ends. Capture has
            // already stopped at this point, so blocking here to drain the
            // executor is safe and bounded by awaitDrain()'s timeout.
            val len = bufferPos
            resetState()
            val chunk = if (len in MIN_SPEECH_SAMPLES..maxSamples) buffer.copyOf(len) else null

            executor.execute {
                val text = if (chunk != null) engine.transcribeBuffer(chunk) else ""
                var combined = stitch(pendingText, text)
                pendingText = ""
                // Fold in a not-yet-polled result too, in case a prior
                // flush()'s job finished after the last acceptWaveForm poll.
                val existingReady = readyResult
                readyResult = null
                if (existingReady != null) combined = stitch(existingReady, combined)
                readyResult = combined.ifEmpty { null }
            }
            awaitDrain()

            val r = readyResult
            readyResult = null
            return r ?: ""
        }

        /** Blocks until every job queued on [executor] so far has completed. */
        private fun awaitDrain() {
            val latch = CountDownLatch(1)
            executor.execute { latch.countDown() }
            try {
                latch.await(DRAIN_TIMEOUT_MS, TimeUnit.MILLISECONDS)
            } catch (e: InterruptedException) {
                Thread.currentThread().interrupt()
            }
        }

        override fun reset() {
            resetState()
            lastResult = ""
            readyResult = null
            executor.execute {
                pendingText = ""
                readyResult = null
            }
        }

        companion object {
            // Ignore utterances shorter than 0.25 s (stray clicks/noise).
            private const val MIN_SPEECH_SAMPLES = WhisperSplitEngine.SAMPLE_RATE / 4
            // ~0.4 s of silence ends an utterance and triggers transcription.
            private const val PAUSE_SAMPLES = WhisperSplitEngine.SAMPLE_RATE * 2 / 5
            // Per-frame RMS above this counts as speech (below ≈ noise/silence).
            // Device measurements: background noise ≈ 0.011, speech ≈ 0.025+.
            private const val SPEECH_RMS = 0.018f
            // Trailing audio (~0.3 s) carried across a mid-utterance window
            // split so a word cut by the split appears whole in the next
            // window too, letting stitch() line the two chunks' text back up.
            // Independent of the model's window size — this only needs to
            // cover one word, not scale with a longer window.
            private const val OVERLAP_SAMPLES = WhisperSplitEngine.SAMPLE_RATE * 3 / 10
            // Safety bound for getFinalResult()'s drain wait. A larger model
            // window means more mel frames/encoder tokens per inference call,
            // so this is generous rather than tuned tightly to the 1s model's
            // observed ~3s/chunk — just a backstop against hanging forever.
            private const val DRAIN_TIMEOUT_MS = 30_000L
        }
    }

    companion object {
        private const val TAG = "WhisperLocal"
    }
}
