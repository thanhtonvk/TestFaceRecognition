package com.example.testfacerecognition;

import android.app.AlertDialog;
import android.graphics.Bitmap;
import android.graphics.BitmapFactory;
import android.graphics.Canvas;
import android.graphics.Color;
import android.graphics.Paint;
import android.graphics.RectF;
import android.graphics.drawable.BitmapDrawable;
import android.os.Bundle;
import android.util.SparseArray;
import android.view.View;
import android.widget.Button;
import android.widget.ImageView;

import androidx.appcompat.app.AppCompatActivity;

import com.google.android.gms.vision.Frame;
import com.google.android.gms.vision.face.Face;
import com.google.android.gms.vision.face.FaceDetector;
import com.google.android.gms.vision.face.Landmark;

public class FaceDetectActivity extends AppCompatActivity {

    Button btDetection;
    ImageView imageFace;
    @Override
    protected void onCreate(Bundle savedInstanceState) {
        super.onCreate(savedInstanceState);
        setContentView(R.layout.activity_face_detect);
        imageFace = findViewById(R.id.face);
        btDetection = findViewById(R.id.detection);
        btDetection.setOnClickListener(new View.OnClickListener() {
            @Override
            public void onClick(View v) {
                BitmapFactory.Options bitmapOptions = new BitmapFactory.Options();
                bitmapOptions.inMutable = true;
                Bitmap defaultBitmap = BitmapFactory.decodeResource(getResources(), R.drawable.download, bitmapOptions);
                Paint rectPaint = new Paint();
                rectPaint.setStrokeWidth(5);
                rectPaint.setColor(Color.CYAN);
                rectPaint.setStyle(Paint.Style.STROKE);

                Bitmap temporaryBitmap = Bitmap.createBitmap(defaultBitmap.getWidth(), defaultBitmap
                        .getHeight(), Bitmap.Config.RGB_565);
                Canvas canvas = new Canvas(temporaryBitmap);
                canvas.drawBitmap(defaultBitmap, 0, 0, null);

                FaceDetector faceDetector = new FaceDetector.Builder(FaceDetectActivity.this)
                        .setTrackingEnabled(false)
                        .setLandmarkType(FaceDetector.ALL_LANDMARKS)
                        .build();
                if (!faceDetector.isOperational()) {
                    new AlertDialog.Builder(FaceDetectActivity.this)
                            .setMessage("Face Detector could not be set up on your device :(")
                            .show();
                    return;
                }

                Frame frame = new Frame.Builder().setBitmap(defaultBitmap).build();
                SparseArray<Face> sparseArray = faceDetector.detect(frame);
                for (int i = 0; i < sparseArray.size(); i++) {
                    Face face = sparseArray.valueAt(i);
                    float left = face.getPosition().x;
                    float top = face.getPosition().y;
                    float right = left + face.getWidth();
                    float bottom = right + face.getHeight()-500;
                    float cornerRadius = 2.0f;
                    RectF rectF = new RectF(left, top, right, bottom);
                    canvas.drawRoundRect(rectF, cornerRadius, cornerRadius, rectPaint);
                }
                imageFace.setImageDrawable(new BitmapDrawable(getResources(), temporaryBitmap));
                faceDetector.release();
            }
        });
    }
}