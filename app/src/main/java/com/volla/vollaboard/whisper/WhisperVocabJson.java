package com.volla.vollaboard.whisper;

import android.util.Log;

import org.json.JSONArray;
import org.json.JSONObject;

import java.io.File;
import java.io.FileInputStream;
import java.io.IOException;
import java.util.HashMap;
import java.util.Iterator;
import java.util.Map;

public class WhisperVocabJson {
    private static final String TAG = "WhisperVocabJson";

    public final Map<Integer, String> tokenToWord = new HashMap<>();
    public int tokenEOT = 50256;
    public int tokenSOT = 50257;
    public int tokenTranscribe = 50359;
    public int tokenNoTimestamps = 50363;
    public int tokenLang = -1;

    public void load(String modelDir, String langCode) throws IOException {
        loadVocab(modelDir);
        loadAddedTokens(modelDir, langCode);
        Log.d(TAG, "Vocab loaded: " + tokenToWord.size() + " tokens"
                + ", EOT=" + tokenEOT + ", SOT=" + tokenSOT
                + ", transcribe=" + tokenTranscribe + ", lang=" + tokenLang);
        if (tokenLang == -1) {
            Log.w(TAG, "No language token found for '" + langCode + "', proceeding without it");
        }
    }

    private void loadVocab(String modelDir) throws IOException {
        String json = readFile(new File(modelDir, "vocab.json"));
        try {
            JSONObject obj = new JSONObject(json);
            Iterator<String> keys = obj.keys();
            while (keys.hasNext()) {
                String word = keys.next();
                int id = obj.getInt(word);
                tokenToWord.put(id, word);
            }
        } catch (Exception e) {
            throw new IOException("Failed to parse vocab.json: " + e.getMessage(), e);
        }
    }

    private void loadAddedTokens(String modelDir, String langCode) throws IOException {
        String json = readFile(new File(modelDir, "added_tokens.json"));
        String langToken = "<|" + langCode + "|>";
        try {
            JSONArray arr = new JSONArray(json);
            for (int i = 0; i < arr.length(); i++) {
                JSONObject tok = arr.getJSONObject(i);
                int id = tok.getInt("id");
                String content = tok.getString("content");
                tokenToWord.put(id, content);
                switch (content) {
                    case "<|endoftext|>":         tokenEOT = id;           break;
                    case "<|startoftranscript|>": tokenSOT = id;           break;
                    case "<|transcribe|>":        tokenTranscribe = id;    break;
                    case "<|notimestamps|>":      tokenNoTimestamps = id;  break;
                }
                if (content.equals(langToken)) {
                    tokenLang = id;
                }
            }
        } catch (Exception e) {
            throw new IOException("Failed to parse added_tokens.json: " + e.getMessage(), e);
        }
    }

    private static String readFile(File file) throws IOException {
        FileInputStream fis = new FileInputStream(file);
        byte[] bytes = new byte[(int) file.length()];
        fis.read(bytes);
        fis.close();
        return new String(bytes, "UTF-8");
    }
}
