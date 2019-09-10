package com.son.ai.mirror;

import java.io.File;
import java.util.ArrayList;
import org.bytedeco.javacpp.Loader;
import org.bytedeco.javacpp.opencv_java;
import org.opencv.core.Mat;
import org.opencv.core.MatOfInt;
import org.opencv.face.EigenFaceRecognizer;
import org.opencv.imgcodecs.Imgcodecs;

public class FaceRecognition {
  public static String basePath = System.getProperty("user.dir");
  static {
    System.out.println("Loading library..");
    Loader.load(opencv_java.class);
    System.out.println("Library loaded!!");
  }
  public static ArrayList<Mat> images = new ArrayList();
  public static ArrayList<Integer> labels = new ArrayList();
  public static ArrayList<String> name = new ArrayList();
  private static int index = 0;
  private static MatOfInt labelsMat;
  private static EigenFaceRecognizer efr;
  private static String unknowImageLink = "unknown_face/s1/9.pgm";

  private static String folderTraining = "training_faces";
  public static int indexLabel = 0;

  public static void main(String[] args) {
    readTrainingData(images, labels);
    Mat testSample = Imgcodecs.imread(unknowImageLink, 0);
    labelsMat = new MatOfInt();
    labelsMat.fromList(labels);
    efr = EigenFaceRecognizer.create();
    System.out.println("Starting training...");
    efr.train(images, labelsMat);

    int[] outLabel = new int[1];
    double[] outConf = new double[1];
    System.out.println("Starting Prediction...");
    efr.predict(testSample, outLabel, outConf);

    System.out.println("***Predicted label is " + outLabel[0] + ".***");

    System.out.println("***Confidence value is " + outConf[0] + ".***");

  }

  private static void readTrainingData(ArrayList<Mat> images,
      ArrayList<Integer> labels) {
    File file = new File(folderTraining);
    File[] listFiles = file.listFiles();
    for (File listFile : listFiles) {
      File[] listImages = listFile.listFiles();
      for (File imageFile : listImages) {
        images.add(Imgcodecs.imread(imageFile.getAbsolutePath(), 0));
        labels.add(indexLabel);
      }
      indexLabel++;
    }
  }

}
