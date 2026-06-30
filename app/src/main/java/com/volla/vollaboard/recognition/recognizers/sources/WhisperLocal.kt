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
import java.util.concurrent.Executor
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
     * speaker pauses (or when the buffer reaches the model's 1-second limit).
     *
     * This aligns the model's 1-second window to the actual speech instead of
     * cutting on arbitrary clock boundaries, which previously sliced words in
     * half and produced hallucinated output. MySpeechService calls
     * acceptWaveForm on a background thread, so inference runs inline.
     */
    private class WhisperRecognizer(
        private val engine: WhisperSplitEngine,
        override val locale: Locale?
    ) : Recognizer {

        override val sampleRate: Float = WhisperSplitEngine.SAMPLE_RATE.toFloat()

        // Speech is accumulated here (float [-1, 1]); capped at the model's window.
        private val buffer = FloatArray(MAX_SAMPLES)
        private var bufferPos = 0
        private var inSpeech = false
        private var silenceRun = 0          // consecutive silent samples while in speech
        private var lastResult = ""

        override fun acceptWaveForm(buffer: ShortArray?, nread: Int): Boolean {
            if (buffer == null || !engine.isInitialized()) return false

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
                if (silenceRun >= PAUSE_SAMPLES) return flush()
            }
            // Otherwise: leading silence before any speech — ignore it.

            // Can't hold more than the model's window — flush what we have.
            if (bufferPos >= MAX_SAMPLES) return flush()
            return false
        }

        /** Copies up to the remaining capacity of [buffer] into the speech buffer. */
        private fun appendFrame(src: ShortArray, nread: Int) {
            var i = 0
            while (i < nread && bufferPos < MAX_SAMPLES) {
                buffer[bufferPos++] = src[i] / 32767f
                i++
            }
        }

        /** Transcribes the accumulated utterance and resets the VAD state. */
        private fun flush(): Boolean {
            val len = bufferPos
            resetState()
            if (len < MIN_SPEECH_SAMPLES) return false   // too short to be a word

            val result = engine.transcribeBuffer(buffer.copyOf(len))
            if (result.isNotEmpty()) {
                lastResult = result
                return true
            }
            return false
        }

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
            // Flush any buffered speech when recognition ends.
            if (bufferPos in MIN_SPEECH_SAMPLES until (MAX_SAMPLES + 1)) {
                val len = bufferPos
                resetState()
                val result = engine.transcribeBuffer(buffer.copyOf(len))
                if (result.isNotEmpty()) lastResult = result
            } else {
                resetState()
            }
            return lastResult.also { lastResult = "" }
        }

        override fun reset() {
            resetState()
            lastResult = ""
        }

        companion object {
            // The model processes at most 1 second (mel extractor input = 16000).
            private const val MAX_SAMPLES = WhisperSplitEngine.SAMPLE_RATE * 1
            // Ignore utterances shorter than 0.25 s (stray clicks/noise).
            private const val MIN_SPEECH_SAMPLES = WhisperSplitEngine.SAMPLE_RATE / 4
            // ~0.4 s of silence ends an utterance and triggers transcription.
            private const val PAUSE_SAMPLES = WhisperSplitEngine.SAMPLE_RATE * 2 / 5
            // Per-frame RMS above this counts as speech (below ≈ noise/silence).
            // Device measurements: background noise ≈ 0.011, speech ≈ 0.025+.
            private const val SPEECH_RMS = 0.018f
        }
    }

    companion object {
        private const val TAG = "WhisperLocal"
    }
}
