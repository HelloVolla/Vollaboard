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
 *   whisper_decoder.tflite       — KV-cached decoder with two signatures
 *
 * The decoder (extract_decoder_cache.py) exposes:
 *   prefill: input_ids [1, 4] + encoder_hidden_states [1, 1500, 384]
 *            → logits [1, vocab], self_k/self_v [4, 1, 6, 4, 64],
 *              cross_k/cross_v [4, 1, 6, 1500, 64]
 *   step:    input_ids [1, 1] + encoder_hidden_states + self_k/self_v
 *            [4, 1, 6, T, 64] + cross_k/cross_v
 *            → logits [1, vocab], self_k/self_v [4, 1, 6, T+1, 64]
 *
 * Decoding runs prefill once on the prompt, then feeds only the latest token
 * to step, carrying the self K/V caches forward (cross K/V stay fixed per
 * chunk). The next token is the argmax of the returned logits.
 */
public class WhisperSplitEngine {
    private static final String TAG = "WhisperSplitEngine";

    public static final int SAMPLE_RATE = 16000;

    // Intra-op thread count for the TFLite interpreters. Measured on an
    // 8-core device, 2 threads matches 4: encoder 1392ms vs 1353ms (within
    // noise) and decode was slightly faster (211 vs 229 ms/token, likely less
    // thread-coordination overhead on the decoder's many small invocations).
    // That means the workload is memory-bandwidth-bound, not compute-bound —
    // so 2 leaves two cores free for the rest of the system at no cost in
    // speed. Thread count does not affect transcription quality, only how the
    // same work is spread across cores.
    private static final int NUM_THREADS = 2;

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

    // Tokens produced by the most recent greedyDecode, for the timing log.
    private int lastGeneratedTokens = 0;

    // Number of PCM samples the mel extractor consumes per call (read at init).
    private int melInputSamples = SAMPLE_RATE;

    private static final String SIG_PREFILL = "prefill";
    private static final String SIG_STEP    = "step";

    // Cap on prompt + generated tokens per chunk, as in inference.py.
    private static final int MAX_TOKENS = 32;

    // Decoder dimensions, read from the prefill signature at init.
    private int kvLayers  = 0;
    private int kvHeads   = 0;
    private int kvHeadDim = 0;
    private int vocabSize = 0;

    // Cross-attention K/V from prefill, reused by every step of the chunk.
    private ByteBuffer crossK;
    private ByteBuffer crossV;

    private boolean initialized = false;

    public int  getMelInputSamples() { return melInputSamples; }
    public boolean isInitialized()   { return initialized; }

    // =========================================================================
    // Lifecycle
    // =========================================================================

    public boolean initialize(String modelDir, String langCode) throws IOException {
        vocab = new WhisperVocabJson();
        vocab.load(modelDir, langCode);

        int cores = Runtime.getRuntime().availableProcessors();
        int threads = Math.min(NUM_THREADS, cores);
        Log.d(TAG, "Interpreter threads=" + threads + " (device cores=" + cores + ")");

        Interpreter.Options opts = new Interpreter.Options();
        opts.setNumThreads(threads);
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
        crossK = null;
        crossV = null;
        initialized = false;
    }

    // =========================================================================
    // Public entry point
    // =========================================================================

    public String transcribeBuffer(float[] samples) {
        if (!initialized) return "";

        long tStart = System.currentTimeMillis();
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

        // Pairs with the greedyDecode log below: tells us whether a chunk that
        // exhausted the token budget was a full window or a short tail chunk.
        Log.d(TAG, "transcribeBuffer: chunkSeconds=" + ((float) validLen / SAMPLE_RATE)
                + " (window=" + ((float) melInputSamples / SAMPLE_RATE) + "s)");

        // Stage timings. mel+encoder is a flat cost (the buffer is padded to the
        // full window regardless of how much real speech it holds), while decode
        // scales with tokens generated — so this shows which half any latency
        // work has to target.
        long t0 = System.currentTimeMillis();
        float[] mel = runMelExtractor(padded);
        if (mel == null) return "";
        long t1 = System.currentTimeMillis();

        float[] encoderOut = runEncoder(mel);
        if (encoderOut == null) return "";
        long t2 = System.currentTimeMillis();

        String text = greedyDecode(encoderOut);
        long t3 = System.currentTimeMillis();

        long decodeMs = t3 - t2;
        String perToken = lastGeneratedTokens > 0
                ? " (" + lastGeneratedTokens + " tokens, " + (decodeMs / lastGeneratedTokens) + "ms/token)"
                : "";
        Log.d(TAG, "timing: prep=" + (t0 - tStart) + "ms"
                + " mel=" + (t1 - t0) + "ms"
                + " encoder=" + (t2 - t1) + "ms"
                + " decode=" + decodeMs + "ms" + perToken
                + " TOTAL=" + (t3 - tStart) + "ms");
        return text;
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
    // Greedy decoder (KV-cached: prefill once, then one token per step)
    // =========================================================================

    private String greedyDecode(float[] encoderOut) {
        int[] prompt = buildPrompt();
        List<Integer> tokens = new ArrayList<>();
        for (int t : prompt) tokens.add(t);

        ByteBuffer encBuf = floatArrayToBuffer(encoderOut);
        float[][] logits = new float[1][vocabSize];
        String stopReason = "token budget exhausted";

        try {
            // prefill: the whole prompt → first logits, the prompt's self K/V
            // and the cross K/V, which stay fixed for the rest of the chunk.
            float[][][][][] selfK = newSelfKv(prompt.length);
            float[][][][][] selfV = newSelfKv(prompt.length);

            Map<String, Object> inputs = new HashMap<>();
            inputs.put("input_ids", new int[][]{prompt});
            inputs.put("encoder_hidden_states", encBuf);

            Map<String, Object> outputs = new HashMap<>();
            outputs.put("logits",  logits);
            outputs.put("self_k",  selfK);
            outputs.put("self_v",  selfV);
            outputs.put("cross_k", rewound(crossK));
            outputs.put("cross_v", rewound(crossV));

            decoder.runSignature(inputs, outputs, SIG_PREFILL);
            tokens.add(argmax(logits[0]));

            // step: feed the last token plus the caches; the self K/V come
            // back one position longer.
            while (tokens.get(tokens.size() - 1) != vocab.tokenEOT
                    && tokens.size() < MAX_TOKENS) {
                int cacheLen = selfK[0][0][0].length;
                float[][][][][] nextK = newSelfKv(cacheLen + 1);
                float[][][][][] nextV = newSelfKv(cacheLen + 1);

                inputs.clear();
                inputs.put("input_ids", new int[][]{{tokens.get(tokens.size() - 1)}});
                inputs.put("encoder_hidden_states", rewound(encBuf));
                inputs.put("self_k",  selfK);
                inputs.put("self_v",  selfV);
                inputs.put("cross_k", rewound(crossK));
                inputs.put("cross_v", rewound(crossV));

                outputs.clear();
                outputs.put("logits", logits);
                outputs.put("self_k", nextK);
                outputs.put("self_v", nextV);

                decoder.runSignature(inputs, outputs, SIG_STEP);
                selfK = nextK;
                selfV = nextV;
                tokens.add(argmax(logits[0]));
            }
            if (tokens.get(tokens.size() - 1) == vocab.tokenEOT) stopReason = "EOT";
        } catch (Exception e) {
            Log.e(TAG, "decoder failed (tokens=" + tokens.size() + ")", e);
            stopReason = "decoder failed";
        }

        // Skip special tokens (>= EOT) from the visible text.
        StringBuilder result = new StringBuilder();
        for (int i = prompt.length; i < tokens.size(); i++) {
            int t = tokens.get(i);
            if (t < vocab.tokenEOT) {
                String word = vocab.tokenToWord.get(t);
                if (word != null) result.append(word);
            }
        }

        lastGeneratedTokens = tokens.size() - prompt.length;

        // "token budget exhausted" here means the chunk's audio was NOT fully
        // transcribed — the rest of that audio is silently lost. Counts only,
        // no text.
        Log.d(TAG, "greedyDecode: generated " + lastGeneratedTokens
                + "/" + (MAX_TOKENS - prompt.length)
                + " tokens, stopped because: " + stopReason);

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
     * Self-attention K/V cache holding [len] positions. A Java array rather
     * than a ByteBuffer: the sequence dimension is dynamic, and runSignature
     * only resizes an input to match when it can read the shape off an array.
     */
    private float[][][][][] newSelfKv(int len) {
        return new float[kvLayers][1][kvHeads][len][kvHeadDim];
    }

    private static int argmax(float[] row) {
        int best = 0;
        for (int v = 1; v < row.length; v++) {
            if (row[v] > row[best]) best = v;
        }
        return best;
    }

    private static ByteBuffer rewound(ByteBuffer buf) {
        buf.rewind();
        return buf;
    }

    // =========================================================================
    // Decoder layout detection
    // =========================================================================

    private void detectDecoderLayout() {
        List<String> sigs = java.util.Arrays.asList(decoder.getSignatureKeys());
        if (!sigs.contains(SIG_PREFILL) || !sigs.contains(SIG_STEP)) {
            throw new IllegalStateException("whisper_decoder.tflite has signatures " + sigs
                    + "; expected the KV-cache decoder with '" + SIG_PREFILL
                    + "' and '" + SIG_STEP + "' (see extract_decoder_cache.py)");
        }

        // cross_k: [layers, 1, heads, encLen, headDim]
        int[] cross = decoder.getOutputTensorFromSignature("cross_k", SIG_PREFILL).shape();
        kvLayers  = cross[0];
        kvHeads   = cross[2];
        kvHeadDim = cross[4];
        vocabSize = lastDim(decoder.getOutputTensorFromSignature("logits", SIG_PREFILL).shape());

        // Allocated once and reused: ~9 MB each for whisper-tiny.
        int crossBytes = elemCount(cross) * Float.BYTES;
        crossK = ByteBuffer.allocateDirect(crossBytes).order(ByteOrder.nativeOrder());
        crossV = ByteBuffer.allocateDirect(crossBytes).order(ByteOrder.nativeOrder());

        int prefillLen = lastDim(decoder.getInputTensorFromSignature("input_ids", SIG_PREFILL).shape());
        int promptLen = buildPrompt().length;

        Log.d(TAG, "Decoder layout: layers=" + kvLayers + " heads=" + kvHeads
                + " headDim=" + kvHeadDim + " encLen=" + cross[3]
                + " vocab=" + vocabSize + " prefillLen=" + prefillLen);

        if (prefillLen != promptLen) {
            throw new IllegalStateException("decoder prefill takes " + prefillLen
                    + " prompt tokens but the prompt has " + promptLen
                    + " (missing language token?)");
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
