package com.volla.vollaboard.downloader.messages

import com.volla.vollaboard.Constants
import com.volla.vollaboard.data.ModelType
import java.util.*

data class ModelInfo(val url: String, val filename: String, val locale: Locale = Constants.UndefinedLocale, val type : ModelType) {
//    override fun toString(): String {
//        return "ModelInfo{" +
//                "url='" + url + '\'' +
//                ", filename='" + filename + '\'' +
//                ", locale=" + locale +
//                '}'
//    }
}