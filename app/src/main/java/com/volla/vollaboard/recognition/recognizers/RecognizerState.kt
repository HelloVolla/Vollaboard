package com.volla.vollaboard.recognition.recognizers

enum class RecognizerState {
    NONE, LOADING, READY, IN_RAM, CLOSED, ERROR,

    /** The installed model is too old for this build and must be re-downloaded. */
    MODEL_OUTDATED
}