package org.pytorch.helloworld;

import android.annotation.SuppressLint;
import android.content.Context;
import android.graphics.Bitmap;
import android.graphics.BitmapFactory;
import android.os.Bundle;
import android.os.Handler;
import android.os.HandlerThread;
import android.util.Log;
import android.widget.ImageView;
import android.widget.TextView;
import android.os.SystemClock;

import org.pytorch.IValue;
import org.pytorch.Module;
import org.pytorch.Tensor;
import org.pytorch.torchvision.TensorImageUtils;

import java.io.DataOutputStream;
import java.io.File;
import java.io.FileInputStream;
import java.io.FileNotFoundException;
import java.io.FileOutputStream;
import java.io.IOException;
import java.io.InputStream;
import java.io.OutputStream;
import java.net.HttpURLConnection;
import java.net.MalformedURLException;
import java.net.URL;

import androidx.annotation.Nullable;
import androidx.annotation.UiThread;
import androidx.annotation.WorkerThread;
import androidx.appcompat.app.AppCompatActivity;

public class MainActivity extends AppCompatActivity {
  TextView textView;
  TextView msView;
  TextView fileView;
  TextView finView;
  String className = "";
  float moduleForwardDuration = 0.0f;
  float[] scores;
  int j = 0;
  int val = 0;
  Bitmap bitmap = null;
  Module module = null;
  ImageView imageView;
  FileOutputStream fout = null;
  protected HandlerThread mBackgroundThread;
  protected Handler mBackgroundHandler;
  protected Handler mUIHandler;

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
        FileInputStream fout2 = null;
          val = forWard(bitmap, module);
        try {
           fout2 = openFileInput("output.txt");
          finView.setText(String.format("File Pass"));

        } catch (FileNotFoundException e) {
          finView.setText(String.format("File Fail"));
          e.printStackTrace();
        }
        try {
          fout2.close();
        } catch (IOException e) {
          e.printStackTrace();
        }

        //HTTP client
          String attachmentName = "output";
          String attachmentFileName = "output.txt";
          String crlf = "\r\n";
          String twoHyphens = "--";
          String boundary =  "*****";

          try {
            URL url = new URL("http://4126ea81.ngrok.io");
            HttpURLConnection httpUrlConnection = (HttpURLConnection) url.openConnection();;
            httpUrlConnection.setUseCaches(false);
            httpUrlConnection.setDoOutput(true);
            httpUrlConnection.setDoInput(true);
            httpUrlConnection.setRequestMethod("POST");
            httpUrlConnection.setRequestProperty("Connection", "Keep-Alive");
            httpUrlConnection.setRequestProperty("Cache-Control", "no-cache");
            httpUrlConnection.setRequestProperty(
                    "Content-Type", "multipart/form-data;boundary=" + boundary);
            //Create a POST method and send the data to the URL specified

            DataOutputStream request = new DataOutputStream(
                    httpUrlConnection.getOutputStream());

            request.writeBytes(twoHyphens + boundary + crlf);
            request.writeBytes("Content-Disposition: form-data; name=\"" +
                    attachmentName + "\";filename=\"" +
                    attachmentFileName + "\"" + crlf);
            request.writeBytes(crlf);

            //Convert file to bytes?

            //End wrapper
            request.writeBytes(crlf);
            request.writeBytes(twoHyphens + boundary +
                    twoHyphens + crlf);

            request.flush();
            request.close();

          } catch (MalformedURLException e) {
            e.printStackTrace();
            finView.setText(String.format("HTTP Fail"));
            //finish();
          } catch (IOException e) {
            e.printStackTrace();
            //The code is Failing here !!
            finView.setText(String.format("HTTP I/O Fail"));
            //finish();
          }

        //Close the opened file
          try {
            fout.close();
          } catch (IOException e) {
            e.printStackTrace();
            //finish();
          }

        if(val == 1) {
          //finView.setText(String.format("Completed"));
          //finish(); //Without this The thread keeps running forever
        }
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

  //@WorkerThread
  //@Nullable
  protected int forWard(final Bitmap bitmap, Module module){
    for(j = 0; j < 10; j++) {
      try {
        // loading serialized torchscript module from packaged into app android asset model.pt,
        // app/src/model/assets/model.pt
        module = Module.load(assetFilePath(this,"model.pt"));
      } catch (IOException e) {
        Log.e("PytorchHelloWorld", "Error reading assets", e);
        finish();
      }



      // preparing input tensor
      final Tensor inputTensor = TensorImageUtils.bitmapToFloat32Tensor(bitmap,
              TensorImageUtils.TORCHVISION_NORM_MEAN_RGB, TensorImageUtils.TORCHVISION_NORM_STD_RGB);

      // running the model
      float moduleForwardStartTime = SystemClock.elapsedRealtime();
      final Tensor outputTensor = module.forward(IValue.from(inputTensor)).toTensor();
      moduleForwardDuration = SystemClock.elapsedRealtime() - moduleForwardStartTime;

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

      try {
        fout = openFileOutput("output.txt", MODE_APPEND);
        fout.write(Float.toString(moduleForwardDuration).getBytes());
      } catch (FileNotFoundException e) {
        e.printStackTrace();
        finish();
      } catch (IOException e) {
        finish();
        e.printStackTrace();
      }

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

}


//@UiThread
//protected abstract void disPlay(ImageView imageView, String className, TextView textView, TextView msView, TextView fileView);
