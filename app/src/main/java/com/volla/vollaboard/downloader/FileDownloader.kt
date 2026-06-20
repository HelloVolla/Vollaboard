package com.volla.vollaboard.downloader

import android.content.Context
import android.content.Intent
import android.net.Uri
import android.os.Build
import com.volla.vollaboard.data.ModelLink
import com.volla.vollaboard.data.ModelType
import com.volla.vollaboard.downloader.messages.ModelInfo
import java.util.*

object FileDownloader {
    const val ACTION = "action"
    const val ACTION_DOWNLOAD = "action_download"
    const val ACTION_UNZIP = "action_unzip"

    const val DOWNLOAD_URL = "download_url"
    const val DOWNLOAD_FILENAME = "download_filename"
    const val DOWNLOAD_LOCALE = "download_locale"
    const val DOWNLOAD_MODEL_TYPE = "download_model_type"

    const val UNZIP_URI = "unzip_uri"
    const val UNZIP_LOCALE = "unzip_locale"
    fun getInfoForIntent(intent: Intent): ModelInfo? {
        val url = intent.getStringExtra(DOWNLOAD_URL)
        val filename = intent.getStringExtra(DOWNLOAD_FILENAME)
        val locale = if (Build.VERSION.SDK_INT >= Build.VERSION_CODES.TIRAMISU) {
            intent.getSerializableExtra(DOWNLOAD_LOCALE, Locale::class.java)
        } else {
            intent.getSerializableExtra(DOWNLOAD_LOCALE) as Locale?
        }
        val modelType = intent.getStringExtra(DOWNLOAD_MODEL_TYPE)
            ?.let { runCatching { ModelType.valueOf(it) }.getOrNull() }
            ?: ModelType.VoskLocal
        return if (url == null || filename == null || locale == null) null else ModelInfo(
            url,
            filename,
            locale,
            modelType
        )
    }

    fun downloadModel(model: ModelLink, context: Context) {
        var context = context
        context = context.applicationContext
        val serviceIntent = Intent(context, FileDownloadService::class.java)
        serviceIntent.putExtra(ACTION, ACTION_DOWNLOAD)
        serviceIntent.putExtra(DOWNLOAD_URL, model.link)
        serviceIntent.putExtra(DOWNLOAD_FILENAME, model.filename)
        serviceIntent.putExtra(DOWNLOAD_LOCALE, model.locale)
        serviceIntent.putExtra(DOWNLOAD_MODEL_TYPE, model.modelType.name)
        if (Build.VERSION.SDK_INT >= Build.VERSION_CODES.O) {
            context.startForegroundService(serviceIntent)
        } else {
            context.startService(serviceIntent)
        }
    }

    fun importModel(uri: Uri, context: Context) {
        var context = context
        context = context.applicationContext
        val serviceIntent = Intent(context, FileDownloadService::class.java)
        serviceIntent.putExtra(ACTION, ACTION_UNZIP)
        serviceIntent.putExtra(UNZIP_URI, uri)
//        serviceIntent.putExtra(UNZIP_LOCALE, locale)
        if (Build.VERSION.SDK_INT >= Build.VERSION_CODES.O) {
            context.startForegroundService(serviceIntent)
        } else {
            context.startService(serviceIntent)
        }
    }
}