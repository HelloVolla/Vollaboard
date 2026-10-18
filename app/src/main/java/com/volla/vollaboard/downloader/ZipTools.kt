package com.volla.vollaboard.downloader

import android.content.Context
import android.util.Log
import androidx.lifecycle.Observer
import com.volla.vollaboard.BuildConfig
import com.volla.vollaboard.Constants
import com.volla.vollaboard.Tools.deleteRecursive
import java.io.*
import java.util.Locale
import java.util.regex.Pattern
import java.util.zip.ZipEntry
import java.util.zip.ZipFile

object ZipTools {
    private const val TAG = "ZipTools"

    /**
     * Written into every model folder at install time. Its absence marks a model
     * installed by an older build, which for Whisper means it predates the
     * KV-cache decoder and has to be replaced. Safe to put here: the providers
     * only ever look at directories, so this file is never mistaken for a model.
     */
    const val MODEL_META_FILE = ".model-meta.json"

    /** Bumped when a model layout change makes previously installed models unusable. */
    private const val MODEL_META_SCHEMA = 1
    private val localePattern: Pattern =
        Pattern.compile("vosk-model-(small-)?(\\w\\w(-\\w\\w)?)-(\\w+-)?v?\\d\\.?\\d*.*")

    @Throws(IOException::class)
    fun unzip(
        archive: File,
//        tempUnzipLocation: File,
//        unzipFinalDestination: File,
        definedLocale: Locale = Constants.UndefinedLocale,
        context: Context,
        errorObserver: Observer<String>? = null,
        progressObserver: Observer<Double>
    ) {
        var locale = definedLocale
        val tempUnzipLocation = Constants.getTemporaryUnzipLocation(context)

        if (!tempUnzipLocation.parentFile!!.exists()) {
            tempUnzipLocation.parentFile!!.mkdirs()
        }

        if (tempUnzipLocation.exists()) {
            deleteRecursive(tempUnzipLocation)
        }
        val zipfile = ZipFile(archive)
        val e = zipfile.entries()
        val size = zipfile.size().toDouble()

        var foundAmFinalMDL = false
        var foundWhisperModel = false
        var i = 0
        while (e.hasMoreElements()) {
            progressObserver.onChanged(i / size)
            val entry = e.nextElement() as ZipEntry
            if (locale == Constants.UndefinedLocale) {
                Log.d(TAG, "Trying to detect locale: ${entry.name}")
                val matcher = localePattern.matcher(entry.name)
                if (matcher.matches()) {
                    locale = Locale.forLanguageTag(matcher.group(2)!!)
                    Log.d(TAG, "Locale detected: ${locale.toLanguageTag()}")
                }
            }

            // Vosk model signature
            if (!foundAmFinalMDL && entry.name.endsWith("/am/final.mdl")) {
                foundAmFinalMDL = true
            }
            if (!foundAmFinalMDL && entry.name.endsWith("/final.mdl")) {
                foundAmFinalMDL = true
            }

            // Whisper split model signature
            if (!foundWhisperModel && entry.name.endsWith("whisper_encoder.tflite")) {
                foundWhisperModel = true
            }

            unzipEntry(zipfile, entry, tempUnzipLocation.absolutePath)
            i++
        }

        if (!foundAmFinalMDL && !foundWhisperModel) {
            Log.e(TAG, "Not a recognised model zip: ${archive.absolutePath}")
            errorObserver?.onChanged("Zip is not a Vosk model!")
            deleteRecursive(tempUnzipLocation)
            return
        }

        val unzipFinalDestination = Constants.getDirectoryForModel(
            context, locale
        )
        unzipFinalDestination.parentFile!!.mkdirs()

        Log.d(TAG, "Unzipping finished. Moving to ${unzipFinalDestination.absolutePath}")

        // Re-downloading a model must replace the installed one. renameTo() only
        // succeeds onto a missing or empty directory, so move any existing model out
        // of the way first - and keep it until the new one is in place, so a failure
        // never leaves the user with no model at all. The holding folder lives outside
        // Models/ because the providers treat every directory in there as installed.
        val replacedLocation = File(tempUnzipLocation.parentFile, "Replaced")
        if (replacedLocation.exists()) deleteRecursive(replacedLocation)

        val hadPreviousModel = unzipFinalDestination.exists()
        if (hadPreviousModel && !unzipFinalDestination.renameTo(replacedLocation)) {
            Log.e(TAG, "Could not move aside existing model at ${unzipFinalDestination.absolutePath}")
            errorObserver?.onChanged("Could not replace the installed model")
            deleteRecursive(tempUnzipLocation)
            return
        }

        val moveSuccess = tempUnzipLocation.renameTo(unzipFinalDestination)
        if (!moveSuccess) {
            Log.e(TAG, "Could not install model to ${unzipFinalDestination.absolutePath}")
            errorObserver?.onChanged("Could not install the model")
            deleteRecursive(tempUnzipLocation)
            if (hadPreviousModel) replacedLocation.renameTo(unzipFinalDestination)
            return
        }

        if (hadPreviousModel) {
            Log.d(TAG, "Replaced the previously installed model")
            deleteRecursive(replacedLocation)
        }

        writeModelMeta(unzipFinalDestination)
    }

    private fun writeModelMeta(modelDir: File) {
        try {
            File(modelDir, MODEL_META_FILE).writeText(
                """{"schema":$MODEL_META_SCHEMA,""" +
                        """"installedAt":${System.currentTimeMillis()},""" +
                        """"appVersionCode":${BuildConfig.VERSION_CODE}}"""
            )
        } catch (e: IOException) {
            // Not fatal: the model itself installed fine. The worst case is that
            // the model is reported as needing an update when it does not.
            Log.w(TAG, "Could not write $MODEL_META_FILE to ${modelDir.absolutePath}", e)
        }
    }

    @Throws(IOException::class)
    private fun unzipEntry(zipfile: ZipFile, entry: ZipEntry, outputDir: String) {
        if (entry.isDirectory) {
            createDir(File(outputDir, entry.name))
            return
        }
        val outputFile = File(outputDir, entry.name)
        if (!outputFile.parentFile!!.exists()) {
            createDir(outputFile.parentFile!!)
        }
        Log.v("ZIP E", "Extracting: $entry")
        zipfile.getInputStream(entry).use { zin ->
            BufferedInputStream(zin).use { inputStream ->
                BufferedOutputStream(
                    FileOutputStream(outputFile)
                ).use { outputStream ->
                    val b = ByteArray(1024)
                    var n: Int
                    while (inputStream.read(b, 0, 1024).also { n = it } >= 0) {
                        outputStream.write(b, 0, n)
                    }
                }
            }
        }
    }

    private fun createDir(dir: File) {
        if (dir.exists()) {
            return
        }
        Log.v("ZIP E", "Creating dir " + dir.name)
        if (!dir.mkdirs()) {
            throw RuntimeException("Can not create dir $dir")
        }
    }
}