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
     * redrawable streaming display could.
     * consecutive windows overlap by OVERLAP_SECONDS of raw audio (5s window → 3s
     * stride), and each chunk's text is merged onto the held text with
     * aggregate(), only emitting the combined result once a real pause
     * ends the utterance.
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

        // Audio carried from one window into the next on a split. Capped at
        // half the window so a smaller model (e.g. 1s) still advances.
        private val overlapSamples: Int by lazy {
            minOf(OVERLAP_SECONDS * WhisperSplitEngine.SAMPLE_RATE, maxSamples / 2)
        }
        private var bufferPos = 0
        private var inSpeech = false
        private var silenceRun = 0          // consecutive silent samples while in speech
        // Buffered samples that were actually speech (excludes the trailing
        // silence appended during a pause). This — not bufferPos — decides
        // whether there's enough real speech for a pause to end the utterance.
        private var speechSamples = 0
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
                    if (!inSpeech) Log.d(TAG, "VAD: speech started (frameRms=$frameRms)")
                    inSpeech = true
                    silenceRun = 0
                    val before = bufferPos
                    appendFrame(buffer, nread)
                    speechSamples += bufferPos - before
                } else if (inSpeech) {
                    // Keep a SHORT tail of silence so word endings aren't clipped —
                    // but stop there, and keep counting the rest only for pause
                    // detection. Appending the whole pause dilutes the buffer's
                    // average level until the engine's silence gate throws real
                    // speech away: measured, a 0.7s utterance followed by 2.2s of
                    // appended silence came to rms 0.0154 against a 0.015 gate, and
                    // the end of the sentence was discarded without ever reaching
                    // the model. It also stops a long pause from inflating the
                    // buffer to the window size and triggering a pointless split.
                    if (silenceRun < TRAILING_SILENCE_SAMPLES) appendFrame(buffer, nread)
                    silenceRun += nread

                    // A pause only ends the utterance once enough actual speech
                    // has accumulated. Natural mid-sentence gaps are frequently
                    // longer than PAUSE_SAMPLES, and committing there chops a
                    // sentence into sub-second fragments that the model cannot
                    // transcribe (it hallucinates on them). A much longer silence
                    // ends the utterance regardless, so a genuinely short one-word
                    // utterance still gets committed.
                    val enoughSpeech = speechSamples >= MIN_UTTERANCE_SAMPLES
                    if (enoughSpeech && silenceRun >= PAUSE_SAMPLES) {
                        flush("pause")
                    } else if (silenceRun >= LONG_PAUSE_SAMPLES) {
                        flush("long-pause")
                    }
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
        private fun flush(reason: String) {
            val len = bufferPos
            val speech = speechSamples
            Log.d(
                TAG, "flush($reason): speechSeconds=${speech.toSeconds()}" +
                    " bufferSeconds=${len.toSeconds()} silenceSeconds=${silenceRun.toSeconds()}"
            )
            resetState()
            // Judge "too short to be a word" on real speech, not on a buffer
            // that may be mostly trailing silence.
            val chunk = if (speech < MIN_SPEECH_SAMPLES) null else buffer.copyOf(len)
            if (chunk == null) Log.d(TAG, "flush($reason): discarded, too little speech")

            val submittedAt = System.currentTimeMillis()
            executor.execute {
                val startedAt = System.currentTimeMillis()
                val text = if (chunk != null) engine.transcribeBuffer(chunk) else ""
                val combined = aggregate(pendingText, text)
                pendingText = ""
                if (combined.isNotEmpty()) readyResult = combined
                Log.d(
                    TAG, "flush($reason) job done: queueWait=${startedAt - submittedAt}ms" +
                        " work=${System.currentTimeMillis() - startedAt}ms" +
                        " sinceSubmit=${System.currentTimeMillis() - submittedAt}ms"
                )
            }
        }

        /**
         * The model's window filled but speech is still ongoing (no pause
         * yet): hand this window to the background executor and hold its
         * text rather than committing it, since MySpeechService types a
         * committed result immediately and it can't be revised afterwards.
         * Carry the trailing overlap of raw audio into the next window so the
         * words at this split sit well inside the next window too, and
         * aggregate the two chunks' text once the next one arrives.
         */
        private fun splitFlush() {
            Log.d(
                TAG, "splitFlush: window full (${maxSamples.toSeconds()}s) with no pause —" +
                    " holding text to aggregate with the next window"
            )
            val chunk = buffer.copyOf(bufferPos)
            val tail = chunk.copyOfRange(chunk.size - overlapSamples, chunk.size)
            tail.copyInto(buffer)
            bufferPos = overlapSamples
            // The carried-over overlap is prior speech, so it counts toward the
            // next window's speech total.
            speechSamples = overlapSamples
            silenceRun = 0
            // inSpeech stays true — we're mid-utterance, not at a real pause.

            val submittedAt = System.currentTimeMillis()
            executor.execute {
                val startedAt = System.currentTimeMillis()
                val text = engine.transcribeBuffer(chunk)
                pendingText = aggregate(pendingText, text)
                Log.d(
                    TAG, "splitFlush job done: queueWait=${startedAt - submittedAt}ms" +
                        " work=${System.currentTimeMillis() - startedAt}ms" +
                        " sinceSubmit=${System.currentTimeMillis() - submittedAt}ms"
                )
            }
        }

        /**
         * Merges the text of two consecutive, overlapping windows
         *
         * The last word of [prev] and the first word of [next] sit right on
         * a window boundary and may be truncated mid-word, so both are left
         * out of the shared-word search and dropped from the result; with a
         * [OVERLAP_SECONDS] overlap, each of them lies well inside the other
         * window and is transcribed whole there. The longest run of words
         * shared between the rest of [prev]'s tail and [next]'s head
         * (compared ignoring case/punctuation) is kept only once.
         */
        private fun aggregate(prev: String, next: String): String {
            val words = splitWords(prev)
            val newWords = splitWords(next)
            if (words.isEmpty() || newWords.isEmpty()) return (words + newWords).joinToString(" ")

            val tail = words.dropLast(1).map { normalizeWord(it) }
            val head = newWords.drop(1).map { normalizeWord(it) }

            var numShared = 0
            for (k in minOf(tail.size, head.size) downTo 1) {
                if (tail.takeLast(k) == head.take(k)) {
                    numShared = k
                    break
                }
            }
            return (words.dropLast(1) + newWords.drop(numShared + 1)).joinToString(" ")
        }

        private fun splitWords(text: String): List<String> =
            text.trim().split(Regex("\\s+")).filter { it.isNotEmpty() }

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
            word.lowercase(Locale.ROOT).filter { it.isLetterOrDigit() || it == '_' }

        private fun resetState() {
            bufferPos = 0
            inSpeech = false
            silenceRun = 0
            speechSamples = 0
        }

        /** Sample count as seconds, for logging. */
        private fun Int.toSeconds(): Float = this.toFloat() / WhisperSplitEngine.SAMPLE_RATE

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
                var combined = aggregate(pendingText, text)
                pendingText = ""
                // Fold in a not-yet-polled result too, in case a prior
                // flush()'s job finished after the last acceptWaveForm poll.
                // That's a separate utterance with no shared audio, so plain
                // concatenation — aggregate() would drop its boundary words.
                val existingReady = readyResult
                readyResult = null
                if (existingReady != null) {
                    combined = listOf(existingReady, combined).filter { it.isNotEmpty() }.joinToString(" ")
                }
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
            // ~0.8 s of silence ends an utterance, but only once
            // MIN_UTTERANCE_SAMPLES of real speech has accumulated. Was 0.4 s,
            // which fired on ordinary mid-sentence gaps and chopped one spoken
            // sentence into seven fragments — four of them 0.6 s, which the
            // model transcribes as gibberish. See the measurements in
            // WHISPER_INTEGRATION_JOURNEY.md.
            private const val PAUSE_SAMPLES = WhisperSplitEngine.SAMPLE_RATE * 4 / 5
            // A pause only ends an utterance once ~1.2 s of actual speech is
            // buffered; below that the model has too little context and
            // hallucinates. Measured: chunks >= 1.0 s transcribed sensibly,
            // 0.6 s chunks produced garbage.
            private const val MIN_UTTERANCE_SAMPLES = WhisperSplitEngine.SAMPLE_RATE * 6 / 5
            // ...but a long enough silence ends the utterance regardless of how
            // little speech there is, so a genuine one-word reply still commits.
            private const val LONG_PAUSE_SAMPLES = WhisperSplitEngine.SAMPLE_RATE * 2
            // How much trailing silence actually gets buffered (the rest is only
            // counted). Enough to avoid clipping a word ending, little enough not
            // to drag the buffer's average level below the engine's silence gate.
            private const val TRAILING_SILENCE_SAMPLES = WhisperSplitEngine.SAMPLE_RATE * 3 / 10
            // Per-frame RMS above this counts as speech (below ≈ noise/silence).
            // Device measurements: background noise ≈ 0.011, speech ≈ 0.025+.
            private const val SPEECH_RMS = 0.018f
            // Trailing audio carried across a mid-utterance window split, as
            // in inference.py (OVERLAP_SECONDS = 2 on a 5 s window → 3 s
            // stride). Long enough that the boundary words aggregate() drops
            // from one window are transcribed whole in the other.
            private const val OVERLAP_SECONDS = 3
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
