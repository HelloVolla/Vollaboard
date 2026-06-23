package com.volla.vollaboard.whisper;

import android.util.Log;

import org.tensorflow.lite.Interpreter;

import java.io.File;
import java.io.FileInputStream;
import java.io.IOException;
import java.nio.ByteBuffer;
import java.nio.ByteOrder;
import java.nio.channels.FileChannel;
import java.util.ArrayList;
import java.util.HashMap;
import java.util.List;
import java.util.Map;

/**
 * Whisper inference engine for the split-model TFLite format:
 *   whisper_mel_extractor.tflite — raw audio → log-mel spectrogram
 *   whisper_encoder.tflite       — log-mel → encoder hidden states
 *   whisper_decoder.tflite       — (encoder states + token window) → logits
 *
 * The decoder is a fixed-length, non-cached model:
 *   inputs : encoder_hidden_states [1, 1500, 384]
 *            input_ids             [1, SEQ]   (SEQ is fixed, e.g. 32)
 *            attention_mask        [1, SEQ]
 *   output : logits               [1, SEQ, vocab]
 *
 * Decoding re-feeds the whole token sequence (prompt + generated so far) every
 * step, right-padded to SEQ, with the attention mask marking valid positions.
 * The next token is the argmax of the logits row at the last valid position.
 */
public class WhisperSplitEngine {
    private static final String TAG = "WhisperSplitEngine";

    public static final int SAMPLE_RATE = 16000;

    // Audio level handling. Whisper hallucinates non-speech tags ("[Musik]")
    // when fed quiet/noisy audio, so we gate out low-energy chunks and normalise
    // gain so real speech reaches the model at a usable level. Measured on
    // device: background noise sits around rms 0.011, real speech around
    // rms 0.025–0.04, so the gate cleanly separates the two. Gain is capped low
    // enough that marginal chunks aren't blown up into hallucinations.
    private static final float SILENCE_RMS  = 0.015f;  // below this → treat as silence
    private static final float TARGET_PEAK  = 0.40f;   // normalise loudest sample to this
    private static final float MAX_GAIN     = 6.0f;    // cap amplification of quiet input

    private Interpreter melExtractor;
    private Interpreter encoder;
    private Interpreter decoder;
    private WhisperVocabJson vocab;

    // Number of PCM samples the mel extractor consumes per call (read at init).
    private int melInputSamples = SAMPLE_RATE;

    // Decoder input/output indices, discovered by tensor name at init.
    private int encoderStateIdx  = -1;
    private int inputIdsIdx       = -1;
    private int attentionMaskIdx  = -1;
    private int logitsOutputIdx   = -1;

    // Fixed decoder sequence length (input_ids dimension, e.g. 32).
    private int decoderSeqLen = 0;

    private boolean initialized = false;

    public int  getMelInputSamples() { return melInputSamples; }
    public boolean isInitialized()   { return initialized; }

    // =========================================================================
    // Lifecycle
    // =========================================================================

    public boolean initialize(String modelDir, String langCode) throws IOException {
        vocab = new WhisperVocabJson();
        vocab.load(modelDir, langCode);

        Interpreter.Options opts = new Interpreter.Options();
        opts.setNumThreads(Math.min(4, Runtime.getRuntime().availableProcessors()));
        opts.setUseXNNPACK(true);

        melExtractor = loadInterpreter(new File(modelDir, "whisper_mel_extractor.tflite"), opts);
        encoder      = loadInterpreter(new File(modelDir, "whisper_encoder.tflite"),       opts);
        decoder      = loadInterpreter(new File(modelDir, "whisper_decoder.tflite"),       opts);

        melInputSamples = elemCount(melExtractor.getInputTensor(0).shape());

        detectDecoderLayout();
        initialized = true;
        return true;
    }

    public void deinitialize() {
        if (melExtractor != null) { melExtractor.close(); melExtractor = null; }
        if (encoder      != null) { encoder.close();      encoder      = null; }
        if (decoder      != null) { decoder.close();      decoder      = null; }
        initialized = false;
    }

    // =========================================================================
    // Public entry point
    // =========================================================================

    public String transcribeBuffer(float[] samples) {
        if (!initialized) return "";

        int validLen = Math.min(samples.length, melInputSamples);

        // Measure level on the valid (non-padding) portion.
        float peak = 0f;
        double sumSq = 0;
        for (int i = 0; i < validLen; i++) {
            float v = samples[i];
            float a = Math.abs(v);
            if (a > peak) peak = a;
            sumSq += (double) v * v;
        }
        float rms = (float) Math.sqrt(sumSq / Math.max(1, validLen));

        // Gate out near-silent chunks — feeding them produces "[Musik]" etc.
        if (rms < SILENCE_RMS) return "";

        // Normalise gain so quiet speech reaches the model at a usable level.
        float gain = (peak > 1e-6f) ? Math.min(TARGET_PEAK / peak, MAX_GAIN) : 1f;
        float[] padded = new float[melInputSamples];
        for (int i = 0; i < validLen; i++) padded[i] = samples[i] * gain;

        float[] mel = runMelExtractor(padded);
        if (mel == null) return "";

        float[] encoderOut = runEncoder(mel);
        if (encoderOut == null) return "";

        return greedyDecode(encoderOut);
    }

    // =========================================================================
    // Mel extractor
    // =========================================================================

    private float[] runMelExtractor(float[] samples) {
        ByteBuffer inputBuf = floatArrayToBuffer(samples);
        int outSize = elemCount(melExtractor.getOutputTensor(0).shape());
        ByteBuffer outputBuf = ByteBuffer.allocateDirect(outSize * Float.BYTES)
                .order(ByteOrder.nativeOrder());
        try {
            melExtractor.run(inputBuf, outputBuf);
        } catch (Exception e) {
            Log.e(TAG, "mel_extractor failed", e);
            return null;
        }
        return bufferToFloatArray(outputBuf, outSize);
    }

    // =========================================================================
    // Encoder
    // =========================================================================

    private float[] runEncoder(float[] mel) {
        ByteBuffer inputBuf = floatArrayToBuffer(mel);
        int outSize = elemCount(encoder.getOutputTensor(0).shape());
        ByteBuffer outputBuf = ByteBuffer.allocateDirect(outSize * Float.BYTES)
                .order(ByteOrder.nativeOrder());
        try {
            encoder.run(inputBuf, outputBuf);
        } catch (Exception e) {
            Log.e(TAG, "encoder failed", e);
            return null;
        }
        return bufferToFloatArray(outputBuf, outSize);
    }

    // =========================================================================
    // Greedy decoder (fixed-length, re-feed each step)
    // =========================================================================

    private String greedyDecode(float[] encoderOut) {
        List<Integer> tokens = new ArrayList<>();
        for (int t : buildPrompt()) tokens.add(t);

        StringBuilder result = new StringBuilder();

        // We can generate until the fixed window is full.
        while (tokens.size() < decoderSeqLen) {
            int next = decoderStep(encoderOut, tokens);

            if (next < 0 || next == vocab.tokenEOT) break;

            // Skip special tokens (>= EOT) from the visible text.
            if (next < vocab.tokenEOT) {
                String word = vocab.tokenToWord.get(next);
                if (word != null) result.append(word);
            }
            tokens.add(next);
        }

        return cleanNonSpeech(decodeBpe(result.toString()).trim());
    }

    /**
     * Strips Whisper's non-speech annotations — bracketed/parenthesised tags
     * such as "[Musik]", "[Applaus]", "(Gelächter)" and musical-note glyphs —
     * which the model emits for music/noise rather than spoken words.
     */
    private static String cleanNonSpeech(String text) {
        String cleaned = text
                .replaceAll("\\[[^\\]]*\\]", "")   // [Musik]
                .replaceAll("\\([^\\)]*\\)", "")   // (Gelächter)
                .replaceAll("\\*[^*]*\\*", "")     // * Musik *
                .replaceAll("[♪♫*]", "")           // stray notes / asterisks
                .replaceAll("\\s+", " ")
                .trim();
        return cleaned;
    }

    private int[] buildPrompt() {
        List<Integer> p = new ArrayList<>();
        p.add(vocab.tokenSOT);
        if (vocab.tokenLang != -1) p.add(vocab.tokenLang);
        p.add(vocab.tokenTranscribe);
        p.add(vocab.tokenNoTimestamps);
        return toIntArray(p);
    }

    /**
     * One decode step.  Feeds the whole token sequence (right-padded to the
     * decoder's fixed SEQ length) plus the attention mask and encoder states,
     * then returns the argmax token at the last valid position.
     */
    private int decoderStep(float[] encoderOut, List<Integer> tokens) {
        int n = tokens.size();                 // valid tokens (<= decoderSeqLen)
        int inputCount = decoder.getInputTensorCount();

        // input_ids (int32, right-padded with 0 — matches the reference pipeline)
        ByteBuffer idsBuf = ByteBuffer.allocateDirect(decoderSeqLen * Integer.BYTES)
                .order(ByteOrder.nativeOrder());
        for (int i = 0; i < decoderSeqLen; i++) {
            idsBuf.putInt(i < n ? tokens.get(i) : 0);
        }
        idsBuf.rewind();

        // attention_mask (int32, 1 for valid positions, 0 for padding)
        ByteBuffer maskBuf = ByteBuffer.allocateDirect(decoderSeqLen * Integer.BYTES)
                .order(ByteOrder.nativeOrder());
        for (int i = 0; i < decoderSeqLen; i++) {
            maskBuf.putInt(i < n ? 1 : 0);
        }
        maskBuf.rewind();

        Object[] inputs = new Object[inputCount];
        inputs[encoderStateIdx] = floatArrayToBuffer(encoderOut);
        inputs[inputIdsIdx]     = idsBuf;
        if (attentionMaskIdx >= 0) inputs[attentionMaskIdx] = maskBuf;

        int vocabSize = lastDim(decoder.getOutputTensor(logitsOutputIdx).shape());
        ByteBuffer logitsBuf = ByteBuffer.allocateDirect(
                elemCount(decoder.getOutputTensor(logitsOutputIdx).shape()) * Float.BYTES)
                .order(ByteOrder.nativeOrder());
        Map<Integer, Object> outputs = new HashMap<>();
        outputs.put(logitsOutputIdx, logitsBuf);

        try {
            decoder.runForMultipleInputsOutputs(inputs, outputs);
        } catch (Exception e) {
            Log.e(TAG, "decoder step failed (n=" + n + ")", e);
            return -1;
        }

        // Next-token logits are the row at the last valid position (n - 1).
        logitsBuf.rewind();
        logitsBuf.position((n - 1) * vocabSize * Float.BYTES);

        int bestToken  = -1;
        float bestScore = Float.NEGATIVE_INFINITY;
        for (int v = 0; v < vocabSize; v++) {
            float s = logitsBuf.getFloat();
            if (s > bestScore) { bestScore = s; bestToken = v; }
        }
        return bestToken;
    }

    // =========================================================================
    // Decoder layout detection
    // =========================================================================

    private void detectDecoderLayout() {
        int inputCount = decoder.getInputTensorCount();
        for (int i = 0; i < inputCount; i++) {
            String name = decoder.getInputTensor(i).name();
            if (name.contains("encoder_hidden_states")) {
                encoderStateIdx = i;
            } else if (name.contains("input_ids")) {
                inputIdsIdx   = i;
                decoderSeqLen = lastDim(decoder.getInputTensor(i).shape());
            } else if (name.contains("attention_mask")) {
                attentionMaskIdx = i;
            }
        }

        int outputCount = decoder.getOutputTensorCount();
        for (int o = 0; o < outputCount; o++) {
            int[] shape = decoder.getOutputTensor(o).shape();
            if (shape.length >= 2 && shape[shape.length - 1] > 50000) {
                logitsOutputIdx = o;
                break;
            }
        }

        Log.d(TAG, "Decoder layout: encoderStateIdx=" + encoderStateIdx
                + " inputIdsIdx=" + inputIdsIdx
                + " maskIdx=" + attentionMaskIdx
                + " logitsOut=" + logitsOutputIdx
                + " seqLen=" + decoderSeqLen);

        if (encoderStateIdx < 0 || inputIdsIdx < 0 || logitsOutputIdx < 0 || decoderSeqLen <= 0) {
            Log.e(TAG, "FATAL: could not resolve required decoder tensors");
        }
    }

    // =========================================================================
    // BPE detokenisation
    // =========================================================================

    /**
     * GPT-2/Whisper byte-level BPE uses the visible marker 'Ġ' for a leading
     * space and encodes raw UTF-8 bytes through a printable-character map.  The
     * vocabulary words we appended are those visible tokens; convert them back
     * to real UTF-8 text here.
     */
    private static String decodeBpe(String visible) {
        // Map each visible char back to its original byte, then UTF-8 decode.
        java.io.ByteArrayOutputStream bytes = new java.io.ByteArrayOutputStream();
        for (int i = 0; i < visible.length(); i++) {
            char c = visible.charAt(i);
            Integer b = BYTE_DECODER.get(c);
            if (b != null) {
                bytes.write(b);
            } else {
                // Not in the byte map (already plain) — emit its UTF-8 bytes.
                byte[] raw = String.valueOf(c).getBytes(java.nio.charset.StandardCharsets.UTF_8);
                bytes.write(raw, 0, raw.length);
            }
        }
        return new String(bytes.toByteArray(), java.nio.charset.StandardCharsets.UTF_8);
    }

    // GPT-2 byte<->unicode table (reverse: visible char -> byte value).
    private static final Map<Character, Integer> BYTE_DECODER = buildByteDecoder();

    private static Map<Character, Integer> buildByteDecoder() {
        List<Integer> bs = new ArrayList<>();
        for (int i = '!'; i <= '~'; i++) bs.add(i);
        for (int i = 0xA1; i <= 0xAC; i++) bs.add(i);
        for (int i = 0xAE; i <= 0xFF; i++) bs.add(i);
        List<Integer> cs = new ArrayList<>(bs);
        int n = 0;
        for (int b = 0; b < 256; b++) {
            if (!bs.contains(b)) {
                bs.add(b);
                cs.add(256 + n);
                n++;
            }
        }
        Map<Character, Integer> decoder = new HashMap<>();
        for (int i = 0; i < bs.size(); i++) {
            decoder.put((char) (int) cs.get(i), bs.get(i));
        }
        return decoder;
    }

    // =========================================================================
    // Utilities
    // =========================================================================

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

    private static int lastDim(int[] shape) {
        return shape[shape.length - 1];
    }

    private static int[] toIntArray(List<Integer> list) {
        int[] arr = new int[list.size()];
        for (int i = 0; i < list.size(); i++) arr[i] = list.get(i);
        return arr;
    }
}
