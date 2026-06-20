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
     * Accumulates 16 kHz PCM audio from MySpeechService and submits it to the
     * Whisper engine in fixed 3-second chunks.  Since MySpeechService already
     * calls acceptWaveForm on a background thread, inference runs inline without
     * blocking the UI.
     */
    private class WhisperRecognizer(
        private val engine: WhisperSplitEngine,
        override val locale: Locale?
    ) : Recognizer {

        override val sampleRate: Float = WhisperSplitEngine.SAMPLE_RATE.toFloat()

        // Buffer accumulates short samples converted to float [-1, 1]
        private val buffer = FloatArray(CHUNK_SAMPLES)
        private var bufferPos = 0
        private var lastResult = ""

        override fun acceptWaveForm(buffer: ShortArray?, nread: Int): Boolean {
            if (buffer == null || !engine.isInitialized()) return false

            var written = 0
            while (written < nread) {
                val space = CHUNK_SAMPLES - bufferPos
                val toCopy = minOf(nread - written, space)
                for (i in 0 until toCopy) {
                    this.buffer[bufferPos++] = buffer[written + i] / 32767f
                }
                written += toCopy

                if (bufferPos == CHUNK_SAMPLES) {
                    val result = engine.transcribeBuffer(this.buffer)
                    bufferPos = 0
                    if (result.isNotEmpty()) {
                        lastResult = result
                        return true
                    }
                }
            }
            return false
        }

        override fun getResult(): String {
            val r = lastResult
            lastResult = ""
            return r
        }

        override fun getPartialResult(): String = ""

        override fun getFinalResult(): String {
            if (bufferPos == 0) return lastResult.also { lastResult = "" }
            // Process whatever audio is still in the buffer
            val remaining = buffer.copyOf(bufferPos)
            bufferPos = 0
            val result = engine.transcribeBuffer(remaining)
            lastResult = ""
            return result ?: ""
        }

        override fun reset() {
            bufferPos = 0
            lastResult = ""
        }

        companion object {
            // 3 seconds at 16 kHz — balances latency against Whisper's minimum context
            private const val CHUNK_SAMPLES = WhisperSplitEngine.SAMPLE_RATE * 3
        }
    }

    companion object {
        private const val TAG = "WhisperLocal"
    }
}
