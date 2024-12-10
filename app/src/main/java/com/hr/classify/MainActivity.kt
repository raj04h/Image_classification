package com.hr.classify

import android.Manifest
import android.content.Intent
import android.content.pm.PackageManager
import android.graphics.Bitmap
import android.os.Build
import android.os.Bundle
import android.provider.MediaStore
import android.widget.Button
import android.widget.ImageView
import android.widget.TextView
import androidx.appcompat.app.AppCompatActivity
import androidx.core.app.ActivityCompat
import androidx.core.view.ViewCompat
import androidx.core.view.WindowInsetsCompat
import com.hr.classify.ml.FlowerClassifierModel
import com.hr.classify.ml.MyFrut
import org.tensorflow.lite.DataType
import org.tensorflow.lite.support.image.TensorImage
import org.tensorflow.lite.support.tensorbuffer.TensorBuffer
import java.io.IOException
import java.nio.ByteBuffer

class MainActivity : AppCompatActivity() {

    private lateinit var imgIV: ImageView
    private lateinit var resTV: TextView
    private lateinit var scnBTN: Button
    private lateinit var camBTN: Button
    private lateinit var glryBTN: Button

    private var bitmap: Bitmap? = null
    private val imageSize = 224
    private val labels = arrayOf("HRaj","bougainvillea","daisies","garden_rose","gardenias","hibiscus","hydrangeas","lilies", "orchids", "peonies","tulip") // Replace with your actual labels

    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        setContentView(R.layout.activity_main)

        // Handle edge-to-edge layout
        ViewCompat.setOnApplyWindowInsetsListener(findViewById(R.id.main)) { v, insets ->
            val systemBars = insets.getInsets(WindowInsetsCompat.Type.systemBars())
            v.setPadding(systemBars.left, systemBars.top, systemBars.right, systemBars.bottom)
            insets
        }

        imgIV = findViewById(R.id.IV_pic)
        resTV = findViewById(R.id.TV_res)
        scnBTN = findViewById(R.id.BTN_enter)
        camBTN = findViewById(R.id.BTN_cam)
        glryBTN = findViewById(R.id.BTN_gallery)

        camBTN.setOnClickListener { getCamPermission() }
        glryBTN.setOnClickListener { openGallery() }
        scnBTN.setOnClickListener { classifyImage() }
    }

    private fun classifyImage() {
        bitmap?.let {
            try {
                val model = FlowerClassifierModel.newInstance(this)

                // Preprocess the image
                val scaledBitmap = Bitmap.createScaledBitmap(it, imageSize, imageSize, true)
                val tensorImage = TensorImage(DataType.FLOAT32)
                tensorImage.load(scaledBitmap)

                // Normalize the image
                val normalizedImage = tensorImage.buffer as ByteBuffer// Ensures proper reading of the buffer

                val inputFeature0 = TensorBuffer.createFixedSize(intArrayOf(1, 224, 224, 3), DataType.FLOAT32)
                inputFeature0.loadBuffer(normalizedImage)


                // Run inference
                val outputs = model.process(inputFeature0)
                val outputFeature0 = outputs.outputFeature0AsTensorBuffer

                val confidences = outputFeature0.floatArray
                val maxIdx = confidences.indices.maxByOrNull { confidences[it] } ?: -1

                if (maxIdx >= 0) {
                    val predictedLabel = labels[maxIdx]
                    val confidence = confidences[maxIdx] * 100 // Convert to percentage

                    // Display predicted name and accuracy
                    resTV.text = "Prediction: $predictedLabel\nConfidence: ${"%.2f".format(confidence)}%"
                } else {
                    resTV.text = "Unknown"
                }
                model.close()
            } catch (e: IOException) {
                e.printStackTrace()
            }
        } ?: run {
            resTV.text = "No image selected!"
        }
    }

    override fun onActivityResult(requestCode: Int, resultCode: Int, data: Intent?) {
        super.onActivityResult(requestCode, resultCode, data)

        if (resultCode == RESULT_OK) {
            when (requestCode) {
                GALLERY_REQUEST_CODE -> {
                    data?.data?.let { uri ->
                        try {
                            bitmap = MediaStore.Images.Media.getBitmap(contentResolver, uri)
                            imgIV.setImageBitmap(bitmap)
                        } catch (e: IOException) {
                            e.printStackTrace()
                        }
                    }
                }
                CAMERA_REQUEST_CODE -> {
                    bitmap = data?.extras?.get("data") as? Bitmap
                    imgIV.setImageBitmap(bitmap)
                }
            }
        }
    }

    private fun getCamPermission() {
        if (Build.VERSION.SDK_INT >= Build.VERSION_CODES.M &&
            checkSelfPermission(Manifest.permission.CAMERA) != PackageManager.PERMISSION_GRANTED
        ) {
            ActivityCompat.requestPermissions(this, arrayOf(Manifest.permission.CAMERA), CAMERA_PERMISSION_CODE)
        } else {
            openCamera()
        }
    }

    private fun openGallery() {
        val intent = Intent(Intent.ACTION_PICK, MediaStore.Images.Media.EXTERNAL_CONTENT_URI)
        startActivityForResult(intent, GALLERY_REQUEST_CODE)
    }

    private fun openCamera() {
        val intent = Intent(MediaStore.ACTION_IMAGE_CAPTURE)
        startActivityForResult(intent, CAMERA_REQUEST_CODE)
    }

    companion object {
        private const val CAMERA_REQUEST_CODE = 101
        private const val GALLERY_REQUEST_CODE = 102
        private const val CAMERA_PERMISSION_CODE = 100
    }
}
