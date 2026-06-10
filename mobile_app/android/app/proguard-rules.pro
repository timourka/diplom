# google_mlkit_text_recognition contains optional references to non-Latin
# recognizers. The app uses the default/Latin recognizer for dates, so these
# optional script recognizers may be absent from the APK.
-dontwarn com.google.mlkit.vision.text.chinese.**
-dontwarn com.google.mlkit.vision.text.devanagari.**
-dontwarn com.google.mlkit.vision.text.japanese.**
-dontwarn com.google.mlkit.vision.text.korean.**
