package com.volla.vollaboard.recognition.recognizers.providers

import android.content.Context
import com.volla.vollaboard.Tools
import com.volla.vollaboard.data.InstalledModelReference
import com.volla.vollaboard.data.ModelType
import com.volla.vollaboard.recognition.recognizers.RecognizerSource

class Providers(context: Context) {
    private val voskLocalProvider: VoskLocalProvider
    private val whisperLocalProvider: WhisperLocalProvider
    private val providers: List<RecognizerSourceProvider>

    init {
        val providersM = mutableListOf<RecognizerSourceProvider>()
        voskLocalProvider = VoskLocalProvider(context)
        whisperLocalProvider = WhisperLocalProvider(context)
        providersM.add(voskLocalProvider)
        providersM.add(whisperLocalProvider)
        if (Tools.VOSK_SERVER_ENABLED) {
            providersM.add(VoskServerProvider())
        }
        providers = providersM
    }

    fun recognizerSourceForModel(localModel: InstalledModelReference): RecognizerSource? {
        return when (localModel.type) {
            ModelType.VoskLocal -> voskLocalProvider.recognizerSourceForModel(localModel)
            ModelType.WHISPER   -> whisperLocalProvider.recognizerSourceForModel(localModel)
        }
    }

    fun installedModels(): Collection<InstalledModelReference> {
        return providers.map { it.getInstalledModels() }.flatten()
    }
}