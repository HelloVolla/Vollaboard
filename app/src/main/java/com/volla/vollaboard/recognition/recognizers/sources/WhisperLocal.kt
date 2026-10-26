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
import java.util.concurrent.atomic.AtomicBoolean
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
            } catch (e: WhisperSplitEngine.OutdatedModelException) {
                Log.e(TAG, "Model at $modelDir is outdated and must be re-downloaded", e)
                stateMLD.postValue(RecognizerState.MODEL_OUTDATED)
                Handler(Looper.getMainLooper()).post { onLoaded.onChanged(null) }
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
     * voice-activity detection (VAD) and transcribes it with Whisper.
     *
     * MySpeechService reads the microphone and calls acceptWaveForm on a
     * single thread in a tight loop; if that call blocks for as long as
     * Whisper inference takes (measured ~1-2.5s on device), the native
     * AudioRecord buffer — sized for only ~0.2s — silently drops everything
     * spoken during the stall. So transcription runs on a dedicated
     * background executor: acceptWaveForm only ever does cheap VAD
     * Every run encodes the audio from the beginning, not just the newest
     * part, so the model always hears the whole context.
     * Tokens already typed are passed to the engine as a forced prefix,
     * which rebuilds the decoder's KV cache in
     * one prefill call, and only the text after them is generated.
     *
     * Typed text can't be revised, so while the user is still speaking only
     * the words two consecutive runs agree on are committed (LocalAgreement);
     * the rest stays tentative. At a pause everything is committed. The audio
     * is kept as context until the window ([maxSamples]) fills; then only the
     * last [keepSamples] are kept, and a long pause clears everything.
     */
    private class WhisperRecognizer(
        private val engine: WhisperSplitEngine,
        override val locale: Locale?
    ) : Recognizer {

        override val sampleRate: Float = WhisperSplitEngine.SAMPLE_RATE.toFloat()

        // Runs every engine.transcribeBuffer() call, off the capture thread.
        private val executor = Executors.newSingleThreadExecutor()

        // The model's window, read from the loaded mel extractor (30s once
        // re-exported). Resolved lazily: capture only starts after
        // engine.initialize() has completed.
        private val maxSamples: Int by lazy { engine.getMelInputSamples() }

        // All audio since the last reset (float [-1, 1]). Capture thread only.
        private val buffer: FloatArray by lazy { FloatArray(maxSamples) }

        // Audio kept as context when the window fills: 10s on the 30s model,
        // capped at a third of the window so a smaller model still frees room.
        private val keepSamples: Int by lazy {
            minOf(KEEP_SECONDS * WhisperSplitEngine.SAMPLE_RATE, maxSamples / 3)
        }
        private var bufferPos = 0
        // End of the audio already committed at a pause. Noise after it that
        // turns out too short to be speech is dropped back to here.
        private var committedPos = 0
        private var inSpeech = false
        private var silenceRun = 0          // consecutive silent samples while in speech
        // Real speech since the last pause (excludes appended trailing silence).
        private var speechSamples = 0
        // Speech appended since the last in-speech update run.
        private var sinceUpdate = 0
        private var lastResult = ""

        // Set while an update run is queued or running, so updates don't pile
        // up behind each other when a run is slower than UPDATE_SAMPLES.
        private val updateBusy = AtomicBoolean(false)

        // Cancel flag of the newest update run. A commit submitted after it
        // covers the same audio and more, so it sets this flag: the update is
        // skipped if it hasn't started, or its decoding stops if it has.
        // Capture thread only (the job keeps its own reference).
        private var updateCancel = AtomicBoolean(false)

        // Executor-only: tokens already typed for the audio in the buffer, and
        // the tentative tokens of the previous update run.
        private var committedTokens = IntArray(0)
        private var tentative = IntArray(0)

        // Text of [tentative], shown as composing (replaceable) text until
        // it is committed. Written by the executor, read by the capture thread.
        @Volatile
        private var partialText = ""

        // Committed text waiting to be fetched by the capture thread.
        private val readyLock = Any()
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
                    sinceUpdate += bufferPos - before

                    // With committed context, a run with little new speech
                    // just ends right after the forced prefix, so it waits
                    // for more speech before the first update.
                    val minSpeech = if (committedPos > 0) UPDATE_AFTER_COMMIT_SAMPLES else MIN_UTTERANCE_SAMPLES
                    if (sinceUpdate >= UPDATE_SAMPLES && speechSamples >= minSpeech
                        && updateBusy.compareAndSet(false, true)
                    ) {
                        sinceUpdate = 0
                        submit(Mode.UPDATE)
                    }
                } else if (inSpeech) {
                    // Keep a SHORT tail of silence so word endings aren't
                    // clipped; appending the whole pause dilutes the buffer's
                    // level until the engine's silence gate drops real speech.
                    if (silenceRun < TRAILING_SILENCE_SAMPLES) appendFrame(buffer, nread)
                    silenceRun += nread

                    // A pause only ends the utterance once enough actual speech
                    // has accumulated; a much longer silence ends it regardless,
                    // so a genuine one-word reply still commits.
                    val enoughSpeech = speechSamples >= MIN_UTTERANCE_SAMPLES
                    if ((enoughSpeech && silenceRun >= PAUSE_SAMPLES) || silenceRun >= LONG_PAUSE_SAMPLES) {
                        pause()
                    }
                } else if (bufferPos > 0) {
                    // Silence after a pause, with audio still kept as context.
                    // A long pause means a new topic: drop all buffers so the
                    // next speech starts fresh.
                    silenceRun += nread
                    if (silenceRun >= LONG_PAUSE_SAMPLES) {
                        Log.d(TAG, "long pause — resetting all buffers")
                        resetState()
                        executor.execute {
                            committedTokens = IntArray(0)
                            tentative = IntArray(0)
                            partialText = ""
                        }
                    }
                }
                // Otherwise: leading silence before any speech — ignore it.

                // Window full: commit everything, then keep only the last
                // keepSamples as context and free the rest of the buffer.
                if (bufferPos >= maxSamples) {
                    Log.d(
                        TAG, "window full (${maxSamples.toSeconds()}s) — committing," +
                            " keeping the last ${keepSamples.toSeconds()}s"
                    )
                    submit(Mode.TRIM)
                    buffer.copyInto(buffer, 0, bufferPos - keepSamples, bufferPos)
                    bufferPos = keepSamples
                    committedPos = keepSamples
                    speechSamples = 0
                    sinceUpdate = 0
                }
            }
            return pollReady()
        }

        private fun pollReady(): Boolean {
            val r = synchronized(readyLock) {
                readyResult.also { readyResult = null }
            } ?: return false
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

        /** A pause ended the utterance: commit all of it, keep the audio as context. */
        private fun pause() {
            Log.d(
                TAG, "pause: speechSeconds=${speechSamples.toSeconds()}" +
                    " bufferSeconds=${bufferPos.toSeconds()}"
            )
            if (speechSamples < MIN_SPEECH_SAMPLES) {
                Log.d(TAG, "pause: discarded, too little speech")
                bufferPos = committedPos
            } else {
                submit(Mode.COMMIT)
                committedPos = bufferPos
            }
            inSpeech = false
            // silenceRun keeps counting, so a long pause is measured from the
            // end of speech, not from this commit.
            speechSamples = 0
            sinceUpdate = 0
        }

        private enum class Mode {
            UPDATE,  // still speaking: commit only the agreed words
            COMMIT,  // pause: commit everything, keep the audio as context
            TRIM,    // window full: commit everything, keep the last keepSamples as context
            RESET    // end: commit everything, then drop the context
        }

        /** Snapshots the buffer and transcribes it from the start on the executor. */
        private fun submit(mode: Mode) {
            val chunk = buffer.copyOf(bufferPos)
            val cancel = if (mode == Mode.UPDATE) {
                AtomicBoolean(false).also { updateCancel = it }
            } else {
                updateCancel.set(true)
                null
            }
            val submittedAt = System.currentTimeMillis()
            executor.execute {
                val startedAt = System.currentTimeMillis()
                try {
                    if (cancel?.get() == true) {
                        Log.d(TAG, "UPDATE skipped: a commit came after it")
                        return@execute
                    }
                    // Show every token as the decoder produces it. All of it
                    // is after committedTokens, so all of it is still tentative.
                    val result = engine.transcribeBuffer(chunk, committedTokens, cancel) { generated ->
                        partialText = engine.tokensToText(generated)
                    }
                    // Stopped mid-run: its tokens are incomplete, and the
                    // commit queued behind it commits everything anyway.
                    if (cancel?.get() == true) {
                        Log.d(TAG, "UPDATE stopped: a commit came after it")
                        return@execute
                    }
                    val hyp = result.tokens
                    val n = if (mode == Mode.UPDATE) agreedLength(tentative, hyp) else hyp.size
                    if (n > 0) {
                        val newTokens = hyp.copyOf(n)
                        committedTokens += newTokens
                        emit(engine.tokensToText(newTokens))
                    }
                    tentative = if (mode == Mode.UPDATE) hyp.copyOfRange(n, hyp.size) else IntArray(0)
                    partialText = engine.tokensToText(tentative)
                    if (mode == Mode.RESET) committedTokens = IntArray(0)
                    if (mode == Mode.TRIM) {
                        // There are no timestamps to tell which committed
                        // tokens belong to the kept audio, so transcribe it
                        // alone and use that as the new prefix. Not typed —
                        // its text was already committed above.
                        val kept = chunk.copyOfRange(chunk.size - keepSamples, chunk.size)
                        committedTokens = engine.transcribeBuffer(kept, IntArray(0)).tokens
                    }
                } finally {
                    if (mode == Mode.UPDATE) updateBusy.set(false)
                }
                Log.d(
                    TAG, "$mode job done: queueWait=${startedAt - submittedAt}ms" +
                        " work=${System.currentTimeMillis() - startedAt}ms" +
                        " committedTokens=${committedTokens.size} tentative=${tentative.size}"
                )
            }
        }

        /**
         * Length of the shared token prefix of [prev] and [next], cut back to
         * a word boundary so a word is never typed in two halves.
         */
        private fun agreedLength(prev: IntArray, next: IntArray): Int {
            var n = 0
            while (n < prev.size && n < next.size && prev[n] == next[n]) n++
            while (n > 0 && (n >= next.size || !engine.isWordStart(next[n]))) n--
            return n
        }

        /** Appends committed text to [readyResult] for the capture thread. */
        private fun emit(text: String) {
            if (text.isEmpty()) return
            synchronized(readyLock) {
                readyResult = readyResult?.let { "$it $text" } ?: text
            }
        }

        private fun resetState() {
            bufferPos = 0
            committedPos = 0
            inSpeech = false
            silenceRun = 0
            speechSamples = 0
            sinceUpdate = 0
        }

        /** Sample count as seconds, for logging. */
        private fun Int.toSeconds(): Float = this.toFloat() / WhisperSplitEngine.SAMPLE_RATE

        override fun getResult(): String {
            val r = lastResult
            lastResult = ""
            return r
        }

        // Held back while a commit is waiting: the commit replaces the
        // composing text, so showing the newer partial first would flicker.
        override fun getPartialResult(): String =
            if (synchronized(readyLock) { readyResult } != null) "" else partialText

        override fun getFinalResult(): String {
            // Commit any buffered speech when recognition ends. Capture has
            // already stopped at this point, so blocking here to drain the
            // executor is safe and bounded by awaitDrain()'s timeout.
            if (bufferPos - committedPos >= MIN_SPEECH_SAMPLES) submit(Mode.RESET)
            else executor.execute { committedTokens = IntArray(0); tentative = IntArray(0); partialText = "" }
            resetState()
            awaitDrain()

            return synchronized(readyLock) {
                readyResult.also { readyResult = null }
            } ?: ""
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
            synchronized(readyLock) { readyResult = null }
            executor.execute {
                committedTokens = IntArray(0)
                tentative = IntArray(0)
                partialText = ""
                synchronized(readyLock) { readyResult = null }
            }
        }

        companion object {
            // Ignore utterances shorter than 0.25 s (stray clicks/noise).
            private const val MIN_SPEECH_SAMPLES = WhisperSplitEngine.SAMPLE_RATE / 4
            // ~0.8 s of silence ends an utterance, but only once
            // MIN_UTTERANCE_SAMPLES of real speech has accumulated. Shorter
            // gaps fired on ordinary mid-sentence pauses. See the measurements
            // in WHISPER_INTEGRATION_JOURNEY.md.
            private const val PAUSE_SAMPLES = WhisperSplitEngine.SAMPLE_RATE * 4 / 5
            // Below ~1.2 s of speech the model has too little context and
            // hallucinates. Measured: chunks >= 1.0 s transcribed sensibly,
            // 0.6 s chunks produced garbage.
            private const val MIN_UTTERANCE_SAMPLES = WhisperSplitEngine.SAMPLE_RATE * 6 / 5
            // ...but a long enough silence ends the utterance regardless.
            private const val LONG_PAUSE_SAMPLES = WhisperSplitEngine.SAMPLE_RATE * 2
            // How much trailing silence actually gets buffered (the rest is only
            // counted). Enough to avoid clipping a word ending, little enough not
            // to drag the buffer's average level below the engine's silence gate.
            private const val TRAILING_SILENCE_SAMPLES = WhisperSplitEngine.SAMPLE_RATE * 3 / 10
            // Per-frame RMS above this counts as speech (below ≈ noise/silence).
            // Device measurements: background noise ≈ 0.011, speech ≈ 0.025+.
            private const val SPEECH_RMS = 0.018f
            // New speech between in-speech update runs. A run takes ~1.5-2 s,
            // and updateBusy skips any that would queue behind a running one.
            private const val UPDATE_SAMPLES = WhisperSplitEngine.SAMPLE_RATE
            private const val UPDATE_AFTER_COMMIT_SAMPLES = WhisperSplitEngine.SAMPLE_RATE * 1
            // Context kept when the window fills (the rest is freed).
            private const val KEEP_SECONDS = 10
            // Safety bound for getFinalResult()'s drain wait.
            private const val DRAIN_TIMEOUT_MS = 30_000L
        }
    }

    companion object {
        private const val TAG = "WhisperLocal"
    }
}
