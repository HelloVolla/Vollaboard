package com.volla.vollaboard.recognition.recognizers.providers

import com.volla.vollaboard.Constants
import com.volla.vollaboard.data.InstalledModelReference
import com.volla.vollaboard.data.ModelType
import com.volla.vollaboard.recognition.recognizers.RecognizerSource
import com.volla.vollaboard.recognition.recognizers.sources.WhisperLocal
import android.content.Context
import java.io.File
import java.util.Locale

class WhisperLocalProvider(private val context: Context) : RecognizerSourceProvider {

    override fun getInstalledModels(): List<InstalledModelReference> {
        val modelsDir = Constants.getModelsDirectory(context)
        if (!modelsDir.exists()) return emptyList()

        return modelsDir.listFiles()
            .orEmpty()
            .filter { it.isDirectory && File(it, "whisper_encoder.tflite").exists() }
            .map { localeFolder ->
                val locale = Locale.forLanguageTag(localeFolder.name)
                InstalledModelReference(
                    localeFolder.absolutePath,
                    locale.displayName,
                    ModelType.WHISPER
                )
            }
    }

    override fun recognizerSourceForModel(localModel: InstalledModelReference): RecognizerSource {
        val locale = Locale.forLanguageTag(File(localModel.path).name)
        return WhisperLocal(localModel.path, locale)
    }
}
