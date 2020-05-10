package org.pytorch.helloworld;

import android.Manifest;
import android.annotation.SuppressLint;
import android.content.Context;
import android.content.pm.PackageManager;
import android.graphics.Bitmap;
import android.graphics.BitmapFactory;
import android.os.Bundle;
import android.os.Environment;
import android.os.Handler;
import android.os.HandlerThread;
import android.app.Activity;
import android.os.PowerManager;
import android.os.Trace;
import android.util.Log;
import android.widget.ImageView;
import android.widget.TextView;
import android.os.SystemClock;

import org.pytorch.IValue;
import org.pytorch.Module;
import org.pytorch.Tensor;
import org.pytorch.torchvision.TensorImageUtils;
import org.tensorflow.lite.DataType;
import org.tensorflow.lite.Interpreter;
import org.tensorflow.lite.support.common.FileUtil;
import org.tensorflow.lite.support.common.TensorOperator;
import org.tensorflow.lite.support.common.TensorProcessor;
import org.tensorflow.lite.support.common.ops.NormalizeOp;
import org.tensorflow.lite.support.image.ImageProcessor;
import org.tensorflow.lite.support.image.TensorImage;
import org.tensorflow.lite.support.image.ops.ResizeOp;
import org.tensorflow.lite.support.image.ops.ResizeWithCropOrPadOp;
import org.tensorflow.lite.support.image.ops.Rot90Op;
import org.tensorflow.lite.support.label.TensorLabel;
import org.tensorflow.lite.support.tensorbuffer.TensorBuffer;

import java.io.BufferedInputStream;
import java.io.BufferedReader;
import java.io.BufferedWriter;
import java.io.DataOutputStream;
import java.io.File;
import java.io.FileInputStream;
import java.io.FileNotFoundException;
import java.io.FileOutputStream;
import java.io.FileWriter;
import java.io.IOException;
import java.io.InputStream;
import java.io.InputStreamReader;
import java.io.OutputStream;
import java.net.HttpURLConnection;
import java.net.MalformedURLException;
import java.net.ProtocolException;
import java.net.URL;
import java.nio.MappedByteBuffer;
import java.text.SimpleDateFormat;
import java.util.Date;
import java.util.List;
import java.util.Map;

import androidx.annotation.Nullable;
import androidx.annotation.UiThread;
import androidx.annotation.WorkerThread;
import androidx.appcompat.app.AppCompatActivity;
import androidx.core.app.ActivityCompat;
import androidx.core.content.ContextCompat;
import android.app.Activity;

public class MainActivity extends AppCompatActivity {
  TextView textView;
  TextView msView;
  TextView fileView;
  TextView finView;
  Tensor outputTensor = null;
  String className = "";
  float moduleForwardDuration = 0.0f;
  float[] scores;
  int j = 0;
  int val = 0;
  Bitmap bitmap = null;
  Module module = null;
  File file = null;
  ImageView imageView;
  //FileOutputStream fout = null;
  protected HandlerThread mBackgroundThread;
  protected Handler mBackgroundHandler;
  protected Handler mUIHandler;

  SimpleDateFormat simpleDateFormat = new SimpleDateFormat("dd-MM-yyyy-hh-mm-ss");
  String format = simpleDateFormat.format(new Date());

  //Tensorflow code
  MappedByteBuffer tfliteModel;
  MainActivity activity;
  /** An instance of the driver class to run model inference with Tensorflow Lite. */
  protected Interpreter tflite;
  /** Options for configuring the Interpreter. */
  private final Interpreter.Options tfliteOptions = new Interpreter.Options();
  TensorImage inputImageBuffer;
  /** Image size along the x axis. */
  int imageSizeX = 0;
  TensorBuffer outputProbabilityBuffer;
  TensorProcessor probabilityProcessor;
  long runTime=0;

  /** Image size along the y axis. */
  int imageSizeY = 0;
  List<String> labels;

  @Override
  protected void onCreate(Bundle savedInstanceState) {
    super.onCreate(savedInstanceState);
    setContentView(R.layout.activity_main);

    startBackgroundThread();
    setupImageX();

  }

  /*@SuppressLint({"DefaultLocale", "SetTextI18n"})
  @Override
  protected void disPlay(Bitmap bitmap2, String classes, float forwardTime, int iter){
    imageView = findViewById(R.id.image);
    imageView.setImageBitmap(bitmap2);

    textView = findViewById(R.id.text);
    textView.setText(classes);

    msView = findViewById(R.id.text2);
    msView.setText(String.format("%f ms", forwardTime));

    fileView = findViewById(R.id.text3);
    fileView.setText(Integer.toString(iter));
  }*/

  /**
   * Copies specified asset to the file in /files app directory and returns this file absolute path.
   *
   * @return absolute file path
   */

  private void setupImageX(){
   // imageView = findViewById(R.id.image);
    imageView = (ImageView) findViewById(R.id.image);
    textView =  (TextView) findViewById(R.id.text);
    msView = (TextView) findViewById(R.id.text2);
    fileView = (TextView) findViewById(R.id.text3);
    finView = (TextView) findViewById(R.id.text4);
    finView.setText(String.format("On Progress"));


   // imageView.setImageBitmap(bitmap);
    try {
      // creating bitmap from packaged into app android asset 'image.jpg',
      // app/src/main/assets/image.jpg
      bitmap = BitmapFactory.decodeStream(getAssets().open("image.jpg"));
    } catch (IOException e) {
      Log.e("PytorchHelloWorld", "Error reading assets", e);
      finish();
    }
    imageView.setImageBitmap(bitmap);





    Runnable runnable = new Runnable() {
      @Override
      public void run() {
        //FileInputStream fout2 = null;
        //PowerManager pm = (PowerManager) getSystemService(Context.POWER_SERVICE);
        //@SuppressLint("InvalidWakeLockTag") PowerManager.WakeLock wl = pm.newWakeLock(PowerManager.SCREEN_DIM_WAKE_LOCK, "My Tag");
        //wl.acquire();

        int requestCode=0;
        if (ContextCompat.checkSelfPermission(
                MainActivity.this,
                Manifest.permission.WRITE_EXTERNAL_STORAGE)
                == PackageManager.PERMISSION_DENIED){

          ActivityCompat
                  .requestPermissions(
                          MainActivity.this,
                          new String[] { Manifest.permission.WRITE_EXTERNAL_STORAGE},
                          requestCode);
        }

        try {
          //val = forWardPyTorch(bitmap, module);
          //TensorFlow Lite code
          val = forWardTFLite(inputImageBuffer, module);

        } catch (IOException e) {
          e.printStackTrace();
          finish();
        }

        //HTTP client
        File path = Environment.getExternalStoragePublicDirectory(Environment.DIRECTORY_DOCUMENTS);
        int i = uploadFile(path + "/output_" + format+ ".txt");



        if(val == 1) {
          finView.setText(String.format("Completed"));
          //finish(); //Without this The thread keeps running forever
        }
       // wl.release();
      }
    };
    new Thread(runnable).start();
  }


  public static String assetFilePath(Context context, String assetName) throws IOException {
    File file = new File(context.getFilesDir(), assetName);
    if (file.exists() && file.length() > 0) {
      return file.getAbsolutePath();
    }

    try (InputStream is = context.getAssets().open(assetName)) {
      try (OutputStream os = new FileOutputStream(file)) {
        byte[] buffer = new byte[4 * 1024];
        int read;
        while ((read = is.read(buffer)) != -1) {
          os.write(buffer, 0, read);
        }
        os.flush();
      }
      return file.getAbsolutePath();
    }
  }

  protected void startBackgroundThread() {
    mBackgroundThread = new HandlerThread("ModuleActivity");
    mBackgroundThread.start();
    mBackgroundHandler = new Handler(mBackgroundThread.getLooper());
  }

  protected void stopBackgroundThread() {
    mBackgroundThread.quitSafely();
    try {
      mBackgroundThread.join();
      mBackgroundThread = null;
      mBackgroundHandler = null;
    } catch (InterruptedException e) {
      Log.e("", "Error on stopping background thread", e);
    }
  }

  //TensorFlow Lite Code
  protected int forWardTFLite(TensorImage inputImage, Module module) throws IOException {
    File path = Environment.getExternalStoragePublicDirectory(Environment.DIRECTORY_DOCUMENTS);
    file = new File(path,"/output_" + format+ ".txt");
    BufferedWriter writer = new BufferedWriter(new FileWriter(file, true));
    int probabilityTensorIndex = 0;
    //activity = null; // What is this value?

    for(j = 0; j < 1; j++) {

      try {
        // loading serialized torchscript module from packaged into app android asset model.pt,
        // app/src/model/assets/model.pt
        tfliteModel = FileUtil.loadMappedFile(MainActivity.this, String.format("model_%s.tflite", Integer.toString(j)));
        //Need to Add NNDelegates later
        tfliteOptions.setNumThreads(1);
        tflite = new Interpreter(tfliteModel, tfliteOptions);
        labels = FileUtil.loadLabels(MainActivity.this, "labels.txt");
      } catch (IOException e) {
        Log.e("TFLite World", "Error reading assets", e);
        finish();
      }

      //TensorFlow Lite code
      // Reads type and shape of input and output tensors, respectively.
      int imageTensorIndex = 0;
      int[] imageShape = tflite.getInputTensor(imageTensorIndex).shape(); // {1, height, width, 3}
      imageSizeY = imageShape[1];
      imageSizeX = imageShape[2];
      DataType imageDataType = tflite.getInputTensor(imageTensorIndex).dataType();
      inputImageBuffer = new TensorImage(imageDataType);
      inputImageBuffer = loadImage(bitmap, 0);

      int[] probabilityShape =
              tflite.getOutputTensor(probabilityTensorIndex).shape(); // {1, NUM_CLASSES}
      DataType probabilityDataType = tflite.getOutputTensor(probabilityTensorIndex).dataType();

      outputProbabilityBuffer = TensorBuffer.createFixedSize(probabilityShape, probabilityDataType);
      probabilityProcessor = new TensorProcessor.Builder().add(getPostprocessNormalizeOp()).build();


      long startTimeForReference = 0, endTimeForReference=0;

      for(int k = 0; k < 30; k++) {
        startTimeForReference = SystemClock.uptimeMillis();
        tflite.run(inputImageBuffer.getBuffer(), outputProbabilityBuffer.getBuffer().rewind());
        endTimeForReference = SystemClock.uptimeMillis();
        runTime = endTimeForReference - startTimeForReference;
        writer.write(String.format("%s", Float.toString(runTime)));
        if(k!=29)
          writer.write(String.format(","));
      }

      Map<String, Float> labeledProbability =
              new TensorLabel(labels, probabilityProcessor.process(outputProbabilityBuffer))
                      .getMapWithFloatValue();


      writer.newLine();
      writer.flush();


    }


    return 1;
  }

  //@WorkerThread
  //@Nullable
  protected int forWardPyTorch(final Bitmap bitmap, Module module) throws IOException {
    File path = Environment.getExternalStoragePublicDirectory(Environment.DIRECTORY_DOCUMENTS);
    file = new File(path,"/output_" + format+ ".txt");
    BufferedWriter writer = new BufferedWriter(new FileWriter(file, true));

    for(j = 0; j < 8; j++) {
      try {
        // loading serialized torchscript module from packaged into app android asset model.pt,
        // app/src/model/assets/model.pt
        module = Module.load(assetFilePath(this, String.format("model_%s.pt", Integer.toString(j))));
      } catch (IOException e) {
        Log.e("PytorchHelloWorld", "Error reading assets", e);
        finish();
      }

      // preparing input tensor
      final Tensor inputTensor = TensorImageUtils.bitmapToFloat32Tensor(bitmap,
              TensorImageUtils.TORCHVISION_NORM_MEAN_RGB, TensorImageUtils.TORCHVISION_NORM_STD_RGB);

      // running the model
      float moduleForwardStartTime;
      moduleForwardDuration = 0.0f;
      //writer.write(String.format("---------------- Model_%s----------------", Integer.toString(j)));
      //writer.newLine();
      //float[] mean_buf = new float[30];

      for(int k = 0; k < 30; ++k) {
        moduleForwardStartTime = SystemClock.uptimeMillis(); //SystemClock.elapsedRealtime();
        outputTensor = module.forward(IValue.from(inputTensor)).toTensor();
        //mean_buf[k] = SystemClock.elapsedRealtime() - moduleForwardStartTime;
        moduleForwardDuration = SystemClock.uptimeMillis() - moduleForwardStartTime; //SystemClock.elapsedRealtime() - moduleForwardStartTime;;
        writer.write(String.format("%s", Float.toString(moduleForwardDuration)));
        if(k!=29)
          writer.write(String.format(","));
     /*   if(k%5 == 4){
          float lv_mean = 0.0f, lv_var=0.0f;
          for(int m = 0; m <= k; ++m) {
            lv_mean += mean_buf[m];
          }
          lv_mean = lv_mean / k;
          for(int m = 0; m <= k ; ++m){
            lv_var += (mean_buf[m]-lv_mean)*(mean_buf[m]-lv_mean);
          }
          lv_var = lv_var/k;
          float lv_std = (float) Math.sqrt(lv_var);
          writer.write(String.format("%s %s", Float.toString(lv_mean), Float.toString(lv_std)));
        }*/
      }

      // getting tensor content as java array of floats
      scores = outputTensor.getDataAsFloatArray();

      // searching for the index with maximum score
      float maxScore = -Float.MAX_VALUE;
      int maxScoreIdx = -1;
      for (int i = 0; i < scores.length; i++) {
        if (scores[i] > maxScore) {
          maxScore = scores[i];
          maxScoreIdx = i;
        }
      }

      className = ImageNetClasses.IMAGENET_CLASSES[maxScoreIdx];

     /* if(j==0) {
        writer.write(System.getProperty("os.version"));
        writer.newLine();
        writer.write(android.os.Build.VERSION.RELEASE);
        writer.newLine();
        writer.write(android.os.Build.DEVICE);
        writer.newLine();
        writer.write(android.os.Build.MODEL);
        writer.newLine();
        writer.write("Number of Cores: " + Runtime.getRuntime().availableProcessors());
      }*/

      //writer.write(Float.toString(moduleForwardDuration));
      writer.newLine();
      writer.flush();
        //fout = openFileOutput("output.txt", MODE_APPEND);
        //fout.write(Float.toString(moduleForwardDuration).getBytes());

      runOnUiThread(new Runnable() {
        public void run() {
          // showing image on UI
          textView.setText(className);
          msView.setText(String.format("%f ms", moduleForwardDuration));
          fileView.setText(Integer.toString(j));
        }
      });


      // runOnUiThread(() -> disPlay(bitmap, className, moduleForwardDuration, j));


    }
    return 1;
    // showing className on UI
    /*TextView textView = findViewById(R.id.text);
    textView.setText(className);

    TextView msView = findViewById(R.id.text2);
    msView.setText(String.format("%f ms",moduleForwardDuration));

    TextView fileView = findViewById(R.id.text3);
    fileView.setText(Integer.toString(j));*/
  }

  //@WorkerThread
  //@Nullable
  //protected abstract void forWard(Bitmap bitmap, Module module);

  public int uploadFile(String sourceFileUri) {
    String fileName = sourceFileUri;
    int serverResponseCode = 0;
    HttpURLConnection conn = null;
    DataOutputStream dos = null;
    String lineEnd = "\r\n";
    String twoHyphens = "--";
    String boundary = "*****";
    int bytesRead, bytesAvailable, bufferSize;
    byte[] buffer;
    int maxBufferSize = 1 * 1024 * 1024;
    File sourceFile = new File(sourceFileUri);
    try{
      // open a URL connection to the Servlet
      FileInputStream fileInputStream = new FileInputStream(sourceFile);
      URL url = new URL("http://0d22a1ec.ngrok.io");
      //URL url = new URL("https://arctic-thunder.herokuapp.com/");

      // Open a HTTP  connection to  the URL
      conn = (HttpURLConnection) url.openConnection();
      conn.setDoInput(true); // Allow Inputs
      conn.setDoOutput(true); // Allow Outputs
      conn.setUseCaches(false); // Don't use a Cached Copy
      conn.setRequestMethod("POST");
      conn.setRequestProperty("Connection", "Keep-Alive");
      conn.setRequestProperty("ENCTYPE", "multipart/form-data");
      conn.setRequestProperty("Content-Type", "multipart/form-data;boundary=" + boundary);
      conn.setRequestProperty("file", fileName);

      dos = new DataOutputStream(conn.getOutputStream());

      dos.writeBytes(twoHyphens + boundary + lineEnd);
      dos.writeBytes("Content-Disposition: form-data; name=\"" +
              "file" + "\";filename=\"" +
              fileName + "\"" + lineEnd); //Seems like a point of concern

      dos.writeBytes(lineEnd);

      // create a buffer of  maximum size
      bytesAvailable = fileInputStream.available();

      bufferSize = Math.min(bytesAvailable, maxBufferSize);
      buffer = new byte[bufferSize];

      // read file and write it into form...
      bytesRead = fileInputStream.read(buffer, 0, bufferSize);

      while (bytesRead > 0) {

        dos.write(buffer, 0, bufferSize);
        bytesAvailable = fileInputStream.available();
        bufferSize = Math.min(bytesAvailable, maxBufferSize);
        bytesRead = fileInputStream.read(buffer, 0, bufferSize);

      }

      // send multipart form data necesssary after file data...
      dos.writeBytes(lineEnd);
      dos.writeBytes(twoHyphens + boundary + twoHyphens + lineEnd);

      // Responses from the server (code and message)
      serverResponseCode = conn.getResponseCode();
      String serverResponseMessage = conn.getResponseMessage();

      Log.i("uploadFile", "HTTP Response is : "
              + serverResponseMessage + ": " + serverResponseCode);

      fileInputStream.close();
      dos.flush();
      dos.close();

    } catch (FileNotFoundException e) {
      e.printStackTrace();
    } catch (MalformedURLException e) {
      e.printStackTrace();
    } catch (ProtocolException e) {
      e.printStackTrace();
    } catch (IOException e) {
      e.printStackTrace();
    }


    return serverResponseCode;
  }


  //TensorFlow Lite Code

  /** Loads input image, and applies preprocessing. */
  private TensorImage loadImage(final Bitmap bitmap, int sensorOrientation) {
    // Loads bitmap into a TensorImage.
    inputImageBuffer.load(bitmap);

    // Creates processor for the TensorImage.
    int cropSize = Math.min(bitmap.getWidth(), bitmap.getHeight());
    int numRotation = sensorOrientation / 90;
    // TODO(b/143564309): Fuse ops inside ImageProcessor.
    ImageProcessor imageProcessor =
            new ImageProcessor.Builder()
                    .add(new ResizeWithCropOrPadOp(cropSize, cropSize))
                    .add(new ResizeOp(imageSizeX, imageSizeY, ResizeOp.ResizeMethod.NEAREST_NEIGHBOR))
                    .add(new Rot90Op(numRotation))
                    .add(getPreprocessNormalizeOp())
                    .build();
    return imageProcessor.process(inputImageBuffer);
  }

  protected TensorOperator getPreprocessNormalizeOp() {
    return new NormalizeOp(127.5f, 127.5f); //Dummy values -- No significance
  }

  protected TensorOperator getPostprocessNormalizeOp() {
    return new NormalizeOp(0.0f, 1.0f);
  }


}


//@UiThread
//protected abstract void disPlay(ImageView imageView, String className, TextView textView, TextView msView, TextView fileView);
