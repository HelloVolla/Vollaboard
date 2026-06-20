package com.volla.vollaboard.whisper;

import android.util.Log;

import org.tensorflow.lite.Interpreter;
import org.tensorflow.lite.Tensor;

import java.io.File;
import java.io.FileInputStream;
import java.io.IOException;
import java.nio.ByteBuffer;
import java.nio.ByteOrder;
import java.nio.channels.FileChannel;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.HashMap;
import java.util.List;
import java.util.Map;

/**
 * Whisper inference engine for the split-model format:
 *   whisper_mel_extractor.tflite — raw audio → mel spectrogram
 *   whisper_encoder.tflite       — mel spectrogram → encoder hidden states
 *   whisper_decoder.tflite       — (encoder states + tokens) → next-token logits
 *
 * At initialisation the tensor shapes of all three models are logged so that
 * mismatches can be diagnosed quickly during integration.
 */
public class WhisperSplitEngine {
    private static final String TAG = "WhisperSplitEngine";

    public static final int SAMPLE_RATE = 16000;
    private static final int CHUNK_SAMPLES = SAMPLE_RATE * 30; // 30-second window
    private static final int MAX_DECODE_STEPS = 224;

    private Interpreter melExtractor;
    private Interpreter encoder;
    private Interpreter decoder;
    private WhisperVocabJson vocab;

    // Detected at init: whether the decoder takes encoder output as an explicit input.
    // false → tokens only (encoder output baked into model constants or KV-cache)
    // true  → encoder output as first float32 input, tokens as int32 input
    private boolean decoderNeedsEncoderInput;
    private int decoderEncoderInputIdx;
    private int decoderTokenInputIdx;

    private boolean initialized = false;

    public boolean initialize(String modelDir, String langCode) throws IOException {
        vocab = new WhisperVocabJson();
        vocab.load(modelDir, langCode);

        Interpreter.Options opts = new Interpreter.Options();
        opts.setNumThreads(Math.min(4, Runtime.getRuntime().availableProcessors()));
        opts.setUseXNNPACK(true);

        melExtractor = loadInterpreter(new File(modelDir, "whisper_mel_extractor.tflite"), opts);
        encoder      = loadInterpreter(new File(modelDir, "whisper_encoder.tflite"),       opts);
        decoder      = loadInterpreter(new File(modelDir, "whisper_decoder.tflite"),       opts);

        logShapes("mel_extractor", melExtractor);
        logShapes("encoder",       encoder);
        logShapes("decoder",       decoder);

        detectDecoderInputLayout();
        initialized = true;
        return true;
    }

    public boolean isInitialized() { return initialized; }

    public void deinitialize() {
        if (melExtractor != null) { melExtractor.close(); melExtractor = null; }
        if (encoder      != null) { encoder.close();      encoder      = null; }
        if (decoder      != null) { decoder.close();      decoder      = null; }
        initialized = false;
    }

    /**
     * Transcribe a float PCM buffer (values in [-1, 1], 16 kHz mono).
     * The buffer is zero-padded or trimmed to exactly 30 seconds before inference.
     */
    public String transcribeBuffer(float[] samples) {
        if (!initialized) return "";

        float[] padded = new float[CHUNK_SAMPLES];
        System.arraycopy(samples, 0, padded, 0, Math.min(samples.length, CHUNK_SAMPLES));

        float[] mel = runMelExtractor(padded);
        if (mel == null) return "";

        float[] encoderOut = runEncoder(mel);
        if (encoderOut == null) return "";

        return greedyDecode(encoderOut);
    }

    // -------------------------------------------------------------------------
    // Mel extraction
    // -------------------------------------------------------------------------

    private float[] runMelExtractor(float[] samples) {
        ByteBuffer inputBuf = floatArrayToBuffer(samples);

        Tensor outTensor = melExtractor.getOutputTensor(0);
        int outSize = elemCount(outTensor.shape());
        ByteBuffer outputBuf = ByteBuffer.allocateDirect(outSize * Float.BYTES)
                .order(ByteOrder.nativeOrder());

        try {
            melExtractor.run(inputBuf, outputBuf);
        } catch (Exception e) {
            Log.e(TAG, "mel_extractor inference failed", e);
            return null;
        }

        return bufferToFloatArray(outputBuf, outSize);
    }

    // -------------------------------------------------------------------------
    // Encoder
    // -------------------------------------------------------------------------

    private float[] runEncoder(float[] mel) {
        ByteBuffer inputBuf = floatArrayToBuffer(mel);

        Tensor outTensor = encoder.getOutputTensor(0);
        int outSize = elemCount(outTensor.shape());
        ByteBuffer outputBuf = ByteBuffer.allocateDirect(outSize * Float.BYTES)
                .order(ByteOrder.nativeOrder());

        try {
            encoder.run(inputBuf, outputBuf);
        } catch (Exception e) {
            Log.e(TAG, "encoder inference failed", e);
            return null;
        }

        return bufferToFloatArray(outputBuf, outSize);
    }

    // -------------------------------------------------------------------------
    // Greedy decoder
    // -------------------------------------------------------------------------

    private String greedyDecode(float[] encoderOut) {
        List<Integer> tokens = new ArrayList<>();
        tokens.add(vocab.tokenSOT);
        if (vocab.tokenLang != -1) tokens.add(vocab.tokenLang);
        tokens.add(vocab.tokenTranscribe);
        tokens.add(vocab.tokenNoTimestamps);

        StringBuilder result = new StringBuilder();

        for (int step = 0; step < MAX_DECODE_STEPS; step++) {
            int next = decoderStep(encoderOut, tokens);
            if (next < 0 || next == vocab.tokenEOT) break;

            // Append word-piece text; skip control/special tokens above EOT
            if (next < vocab.tokenEOT) {
                String word = vocab.tokenToWord.get(next);
                if (word != null) result.append(word);
            }
            tokens.add(next);
        }

        return result.toString().trim();
    }

    private int decoderStep(float[] encoderOut, List<Integer> tokens) {
        int nTokens = tokens.size();

        // Build token int32 buffer
        ByteBuffer tokenBuf = ByteBuffer.allocateDirect(nTokens * Integer.BYTES)
                .order(ByteOrder.nativeOrder());
        for (int t : tokens) tokenBuf.putInt(t);

        // Resize the token input to current sequence length, then re-allocate tensors
        decoder.resizeInput(decoderTokenInputIdx, new int[]{1, nTokens});
        decoder.allocateTensors();

        // Determine output size after resize (vocab dimension is fixed, sequence dim varies)
        Tensor outTensor = decoder.getOutputTensor(0);
        int outSize = elemCount(outTensor.shape());
        ByteBuffer outputBuf = ByteBuffer.allocateDirect(outSize * Float.BYTES)
                .order(ByteOrder.nativeOrder());

        try {
            if (decoderNeedsEncoderInput) {
                ByteBuffer encoderBuf = floatArrayToBuffer(encoderOut);
                Object[] inputs = new Object[2];
                inputs[decoderEncoderInputIdx] = encoderBuf;
                inputs[decoderTokenInputIdx]   = tokenBuf;
                Map<Integer, Object> outputs = new HashMap<>();
                outputs.put(0, outputBuf);
                decoder.runForMultipleInputsOutputs(inputs, outputs);
            } else {
                decoder.run(tokenBuf, outputBuf);
            }
        } catch (Exception e) {
            Log.e(TAG, "decoder step failed at step tokens=" + nTokens, e);
            return -1;
        }

        // Logits are shaped [1, nTokens, vocabSize] — take last position
        int[] outShape = outTensor.shape();
        int vocabSize = outShape[outShape.length - 1];
        int lastPosOffset = (nTokens - 1) * vocabSize;

        outputBuf.rewind();
        outputBuf.position(lastPosOffset * Float.BYTES);

        int bestToken = -1;
        float bestScore = Float.NEGATIVE_INFINITY;
        for (int v = 0; v < vocabSize; v++) {
            float score = outputBuf.getFloat();
            if (score > bestScore) {
                bestScore = score;
                bestToken = v;
            }
        }
        return bestToken;
    }

    // -------------------------------------------------------------------------
    // Decoder input-layout detection
    // -------------------------------------------------------------------------

    /**
     * Inspect the decoder's input tensors to decide whether it expects
     * (encoder_output, tokens) or just (tokens).  We distinguish by data type:
     * float32 → encoder output, int32 → token IDs.
     */
    private void detectDecoderInputLayout() {
        int inputCount = decoder.getInputTensorCount();
        decoderEncoderInputIdx = -1;
        decoderTokenInputIdx   = 0; // safe default

        for (int i = 0; i < inputCount; i++) {
            Tensor t = decoder.getInputTensor(i);
            if (t.dataType() == org.tensorflow.lite.DataType.INT32) {
                decoderTokenInputIdx = i;
            } else if (t.dataType() == org.tensorflow.lite.DataType.FLOAT32) {
                decoderEncoderInputIdx = i;
            }
        }

        decoderNeedsEncoderInput = (decoderEncoderInputIdx != -1);
        Log.d(TAG, "Decoder layout: needsEncoderInput=" + decoderNeedsEncoderInput
                + " encoderIdx=" + decoderEncoderInputIdx
                + " tokenIdx=" + decoderTokenInputIdx);
    }

    // -------------------------------------------------------------------------
    // Utilities
    // -------------------------------------------------------------------------

    private static Interpreter loadInterpreter(File file, Interpreter.Options opts)
            throws IOException {
        FileInputStream fis = new FileInputStream(file);
        FileChannel channel = fis.getChannel();
        ByteBuffer buf = channel.map(FileChannel.MapMode.READ_ONLY, 0, channel.size());
        fis.close();
        return new Interpreter(buf, opts);
    }

    private static ByteBuffer floatArrayToBuffer(float[] data) {
        ByteBuffer buf = ByteBuffer.allocateDirect(data.length * Float.BYTES)
                .order(ByteOrder.nativeOrder());
        for (float v : data) buf.putFloat(v);
        buf.rewind();
        return buf;
    }

    private static float[] bufferToFloatArray(ByteBuffer buf, int count) {
        buf.rewind();
        float[] out = new float[count];
        for (int i = 0; i < count; i++) out[i] = buf.getFloat();
        return out;
    }

    private static int elemCount(int[] shape) {
        int n = 1;
        for (int d : shape) n *= d;
        return n;
    }

    private static void logShapes(String name, Interpreter interp) {
        for (int i = 0; i < interp.getInputTensorCount(); i++) {
            Tensor t = interp.getInputTensor(i);
            Log.d(TAG, name + " input[" + i + "]: " + t.name()
                    + " shape=" + Arrays.toString(t.shape()) + " type=" + t.dataType());
        }
        for (int i = 0; i < interp.getOutputTensorCount(); i++) {
            Tensor t = interp.getOutputTensor(i);
            Log.d(TAG, name + " output[" + i + "]: " + t.name()
                    + " shape=" + Arrays.toString(t.shape()) + " type=" + t.dataType());
        }
    }
}
