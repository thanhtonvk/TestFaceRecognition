package com.example.testfacerecognition;

import android.app.AlertDialog;
import android.content.Intent;
import android.graphics.Bitmap;
import android.graphics.BitmapFactory;
import android.graphics.Canvas;
import android.graphics.Color;
import android.graphics.Paint;
import android.graphics.RectF;
import android.graphics.drawable.BitmapDrawable;
import android.os.Bundle;
import android.provider.MediaStore;
import android.util.SparseArray;
import android.view.View;
import android.widget.Button;
import android.widget.ImageView;
import android.widget.TextView;

import androidx.annotation.Nullable;
import androidx.appcompat.app.AppCompatActivity;

import com.example.testfacerecognition.ml.Model;
import com.google.android.gms.vision.Frame;
import com.google.android.gms.vision.face.Face;
import com.google.android.gms.vision.face.FaceDetector;

import org.tensorflow.lite.DataType;
import org.tensorflow.lite.support.tensorbuffer.TensorBuffer;

import java.io.BufferedReader;
import java.io.IOException;
import java.io.InputStreamReader;
import java.nio.ByteBuffer;
import java.nio.ByteOrder;
import java.util.ArrayList;
import java.util.List;

public class MainActivity extends AppCompatActivity {


    Button btn_open, btn_predict;
    ImageView img;
    TextView tv_predict;
    int request = 123;
    Bitmap bitmap;
    ByteBuffer imgData;

    @Override
    protected void onCreate(Bundle savedInstanceState) {
        super.onCreate(savedInstanceState);
        setContentView(R.layout.activity_main);
        initView();
        onClick();
        tv_predict.setText(GetLabels().size() + "");
    }

    private List<String> GetLabels() {
        List<String> labels = new ArrayList<>();
        BufferedReader reader = null;
        try {
            reader = new BufferedReader(
                    new InputStreamReader(getAssets().open("label.txt")));

            // do reading, usually loop until end of file reading
            String mLine;
            while ((mLine = reader.readLine()) != null) {
                labels.add(mLine);
            }
        } catch (IOException e) {
            //log the exception
        } finally {
            if (reader != null) {
                try {
                    reader.close();
                } catch (IOException e) {
                    //log the exception
                }
            }
        }
        return labels;
    }

    int DIM_BATCH_SIZE = 1;
    int DIM_IMG_SIZE_X = 224;
    int DIM_IMG_SIZE_Y = 224;
    int DIM_PIXEL_SIZE = 3;

    private void onClick() {
        btn_open.setOnClickListener(new View.OnClickListener() {
            @Override
            public void onClick(View v) {
                Intent intent = new Intent(Intent.ACTION_PICK, MediaStore.Images.Media.EXTERNAL_CONTENT_URI);

                startActivityForResult(intent, request);
            }
        });
        imgData = ByteBuffer.allocateDirect(DIM_BATCH_SIZE * DIM_IMG_SIZE_X * DIM_IMG_SIZE_Y * DIM_PIXEL_SIZE); // now buffer size and input size match
        imgData.order(ByteOrder.nativeOrder());
        btn_predict.setOnClickListener(new View.OnClickListener() {
            @Override
            public void onClick(View v) {
                try {

                    Model model = Model.newInstance(getApplicationContext());

                    // Creates inputs for reference.
                    TensorBuffer inputFeature0 = TensorBuffer.createFixedSize(new int[]{1, 224, 224, 3}, DataType.FLOAT32);

                    ByteBuffer bytebuff = comvertBitmapToByteBuffer(detectFace(bitmap));
                    inputFeature0.loadBuffer(bytebuff);

                    // Runs model inference and gets result.
                    Model.Outputs outputs = model.process(inputFeature0);
                    TensorBuffer outputFeature0 = outputs.getOutputFeature0AsTensorBuffer();
                    int max = getMax(outputFeature0.getFloatArray());
                    tv_predict.setText(GetLabels().get(max) + "acc: " + outputFeature0.getFloatArray()[max]);
                    // Releases model resources if no longer used.
                    model.close();
                } catch (IOException e) {
                    // TODO Handle the exception
                }
            }
        });
    }

    private Bitmap detectFace(Bitmap bitmap) {
        BitmapFactory.Options bitmapOptions = new BitmapFactory.Options();
        bitmapOptions.inMutable = true;
        Bitmap defaultBitmap = bitmap;
        Paint rectPaint = new Paint();
        rectPaint.setStrokeWidth(5);
        rectPaint.setColor(Color.CYAN);
        rectPaint.setStyle(Paint.Style.STROKE);

        Bitmap temporaryBitmap = Bitmap.createBitmap(defaultBitmap.getWidth(), defaultBitmap
                .getHeight(), Bitmap.Config.RGB_565);
        Canvas canvas = new Canvas(temporaryBitmap);
        canvas.drawBitmap(defaultBitmap, 0, 0, null);

        FaceDetector faceDetector = new FaceDetector.Builder(MainActivity.this)
                .setTrackingEnabled(false)
                .setLandmarkType(FaceDetector.ALL_LANDMARKS)
                .build();
        if (!faceDetector.isOperational()) {
            new AlertDialog.Builder(MainActivity.this)
                    .setMessage("Face Detector could not be set up on your device :(")
                    .show();
        }

        Frame frame = new Frame.Builder().setBitmap(defaultBitmap).build();
        SparseArray<Face> sparseArray = faceDetector.detect(frame);
        for (int i = 0; i < sparseArray.size(); i++) {
            Face face = sparseArray.valueAt(i);
            float left = face.getPosition().x;
            float top = face.getPosition().y;
            float right = left + face.getWidth();
            float bottom = top + face.getHeight();
            float cornerRadius = 2.0f;
            RectF rectF = new RectF(left, top, right, bottom);
            canvas.drawRoundRect(rectF, cornerRadius, cornerRadius, rectPaint);
        }
        faceDetector.release();
        img.setImageDrawable(new BitmapDrawable(getResources(), temporaryBitmap));
        return temporaryBitmap;
    }

    private static final float IMAGE_MEAN = 127.5f;
    private static final float IMAGE_STD = 127.5f;

    private ByteBuffer comvertBitmapToByteBuffer(Bitmap bitmap) {

        bitmap = Bitmap.createScaledBitmap(bitmap, 224, 224, false);


        ByteBuffer byteBuffer = ByteBuffer.allocateDirect(4 * 224 * 224 * 3);
        byteBuffer.order(ByteOrder.nativeOrder());
        int[] intValues = new int[224 * 224];


        bitmap.getPixels(intValues, 0, bitmap.getWidth(), 0, 0, bitmap.getWidth(), bitmap.getHeight());
        int pixel = 0;
        for (int i = 0; i < 224; i++) {
            for (int j = 0; j < 224; j++) {
                int input = intValues[pixel++];

                byteBuffer.putFloat((((input >> 16 & 0xFF) - IMAGE_MEAN) / IMAGE_STD));
                byteBuffer.putFloat((((input >> 8 & 0xFF) - IMAGE_MEAN) / IMAGE_STD));
                byteBuffer.putFloat((((input & 0xFF) - IMAGE_MEAN) / IMAGE_STD));
            }
        }
        return byteBuffer;
    }


    @Override
    protected void onActivityResult(int requestCode, int resultCode, @Nullable Intent data) {
        super.onActivityResult(requestCode, resultCode, data);
        if (data != null) {
            img.setImageURI(data.getData());
            try {
                bitmap = MediaStore.Images.Media.getBitmap(getContentResolver(), data.getData());

            } catch (IOException e) {
                e.printStackTrace();
            }
        }
    }

    private void initView() {
        btn_open = findViewById(R.id.btn_open);
        btn_predict = findViewById(R.id.btn_predict);
        img = findViewById(R.id.img);
        tv_predict = findViewById(R.id.tv_predict);
    }

    private int getMax(float[] arr) {
        int index = 0;
        float min = 0.0f;
        for (int i = 0; i < 14; i++) {
            if (arr[i] > min) {
                index = i;
                min = arr[i];
            }
        }
        return index;
    }


}