package com.volla.vollaboard.downloader.messages

import com.volla.vollaboard.Constants
import java.util.*

data class ModelInfo(val url: String, val filename: String, val locale: Locale = Constants.UndefinedLocale) {
//    override fun toString(): String {
//        return "ModelInfo{" +
//                "url='" + url + '\'' +
//                ", filename='" + filename + '\'' +
//                ", locale=" + locale +
//                '}'
//    }
}