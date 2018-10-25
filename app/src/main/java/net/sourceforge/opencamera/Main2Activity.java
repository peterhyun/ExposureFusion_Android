package net.sourceforge.opencamera;

import android.graphics.Bitmap;
import android.graphics.BitmapFactory;
import android.os.Bundle;
import android.app.Activity;
import android.util.DisplayMetrics;
import android.util.Log;
import android.view.View;
import android.content.Intent;
import android.widget.ImageView;
import android.graphics.Bitmap;
import android.net.Uri;
import android.os.Bundle;

import java.io.ByteArrayOutputStream;
import java.io.File;
import java.io.IOException;
import java.nio.file.Files;
import java.nio.file.Paths;
import java.util.ArrayList;
import java.util.List;
import android.util.Log;
import android.widget.ImageView;
import android.widget.TextView;

import android.graphics.Bitmap;
import android.net.Uri;
import android.support.v7.app.AppCompatActivity;
import android.os.Bundle;

import java.io.File;
import java.io.IOException;
import java.nio.file.Files;
import java.nio.file.Paths;
import java.util.ArrayList;
import java.util.List;
import android.util.Log;
import android.widget.ImageView;
import android.widget.TextView;

import org.opencv.android.OpenCVLoader;
import org.opencv.core.Core;
import org.opencv.core.CvType;
import org.opencv.core.Mat;
import org.opencv.core.MatOfDouble;
import org.opencv.core.Scalar;
import org.opencv.core.Size;
import org.opencv.imgcodecs.Imgcodecs;
import org.opencv.photo.CalibrateDebevec;
import org.opencv.photo.MergeDebevec;
import org.opencv.photo.MergeMertens;
import org.opencv.photo.Photo;
import org.opencv.photo.TonemapDurand;
import org.opencv.android.Utils;
import org.opencv.imgproc.Imgproc;
import org.opencv.core.CvException;

import static org.opencv.core.Core.BORDER_REPLICATE;
import static org.opencv.core.CvType.*;



public class Main2Activity extends Activity {
    ArrayList<byte []> arr = new ArrayList<>();
    ArrayList<Bitmap> arr2 = new ArrayList<>();

    static {
        if (!OpenCVLoader.initDebug()) {
            // Handle initialization error
        }
    }

    @Override
    protected void onCreate(Bundle savedInstanceState) {

        super.onCreate(savedInstanceState);
        setContentView(R.layout.activity_main2);

        Intent ss = getIntent();
        if (ss.getExtras() != null) {
            arr.add(ss.getByteArrayExtra("B1"));
            Bitmap bitmap1 = BitmapFactory.decodeByteArray(arr.get(0), 0, arr.get(0).length);
            arr2.add(bitmap1);
            ImageView img1 = findViewById(R.id.img1);
            img1.setImageBitmap(bitmap1);
            //img1.setRotation(90);

            arr.add(ss.getByteArrayExtra("B2"));
            Bitmap bitmap2 = BitmapFactory.decodeByteArray(arr.get(1), 0, arr.get(1).length);
            arr2.add(bitmap2);
            ImageView img2 = findViewById(R.id.img2);
            img2.setImageBitmap(bitmap2);
            //img2.setRotation(90);

            arr.add(ss.getByteArrayExtra("B3"));
            Bitmap bitmap3 = BitmapFactory.decodeByteArray(arr.get(2), 0, arr.get(2).length);
            arr2.add(bitmap3);
            ImageView img3 = findViewById(R.id.img3);
            img3.setImageBitmap(bitmap3);
            //img3.setRotation(90);




            //arr.add(ss.getByteArrayExtra("B2"));
            //arr.add(ss.getByteArrayExtra("B3"));
        }
        //arr.add(ss.getByteArrayExtra("1"));
        //arr.add(ss.getByteArrayExtra("2"));
        //arr.add(ss.getByteArrayExtra("3"));
        /*String check = ss.getStringExtra("string");
        if (check == "3") {
            arr.add(ss.getByteArrayExtra("B1"));
            arr.add(ss.getByteArrayExtra("B2"));
            arr.add(ss.getByteArrayExtra("B3"));
        }*/

        //byte[] a = ss.getByteArrayExtra("B1");
        //byte[] b = ss.getByteArrayExtra("B2");
        //byte[] c = ss.getByteArrayExtra("B3");
        //Log.d("Main2Activity", "got intent");
        /*if (ss.hasExtra("B1")) {
            Log.d("Main2Activity", "I got B1");
            byte[] a = ss.getByteArrayExtra("B1");
        }
        if (ss.hasExtra("B2")) {
            Log.d("Main2Activity", "I got B2");
            byte[] b = ss.getByteArrayExtra("B2");
        }
        if (ss.hasExtra("B3")) {
            Log.d("Main2Activity", "I got B3");
            byte[] c = ss.getByteArrayExtra("B3");
        }*/

    }

    public void doHdrImaging(View view) {

        Intent hdrAct = new Intent(this, Main3Activity.class);
        if (arr.size() == 3) {
            hdrAct.putExtra("b1", arr.get(0));
            hdrAct.putExtra("b2", arr.get(1));
            hdrAct.putExtra("b3", arr.get(2));
        }
        startActivity(hdrAct);





       /* Mat mat1= new Mat();
        Mat mat2= new Mat();
        Mat mat3= new Mat();

        Bitmap bmp1 = arr2.get(0).copy(Bitmap.Config.ARGB_8888, true);
        Bitmap bmp2 = arr2.get(1).copy(Bitmap.Config.ARGB_8888, true);
        Bitmap bmp3 = arr2.get(2).copy(Bitmap.Config.ARGB_8888, true);
        Utils.bitmapToMat(bmp1, mat1);
        Utils.bitmapToMat(bmp2, mat2);
        //8UC4, RGBA format
        Utils.bitmapToMat(bmp3, mat3);

        Imgproc.cvtColor(mat1,mat1,Imgproc.COLOR_RGBA2RGB);
        Imgproc.cvtColor(mat2,mat2,Imgproc.COLOR_RGBA2RGB);
        Imgproc.cvtColor(mat3,mat3,Imgproc.COLOR_RGBA2RGB);



        List<Mat> images = new ArrayList<Mat>();
        images.add(mat1);
        images.add(mat2);
        images.add(mat3);
        Mat resultImage = exposureFusion(images, 1,1,1,3);

        Bitmap bmp = Bitmap.createBitmap(resultImage.cols(), resultImage.rows(), Bitmap.Config.ARGB_8888);
        Utils.matToBitmap(resultImage, bmp);*/

        //Intent hdrAct = new Intent(this, Main3Activity.class);
        //ByteArrayOutputStream stream = new ByteArrayOutputStream();
        //bmp1.compress(Bitmap.CompressFormat.PNG, 100, stream);
        //byte[] bytes1 = stream.toByteArray();
        //ByteArrayOutputStream stream2 = new ByteArrayOutputStream();
        //bmp.compress(Bitmap.CompressFormat.PNG, 100, stream2);
        //byte[] bytes2 = stream2.toByteArray();
        //hdrAct.putExtra("standard", bmp1);
        //hdrAct.putExtra("result", bmp);
        //startActivity(hdrAct);




        /*
            // LOAD IMAGES
        List<Mat> images = new ArrayList<>();
        images.add(readImageFromResources(R.drawable.a));
        images.add(readImageFromResources(R.drawable.b));
        images.add(readImageFromResources(R.drawable.c));
        images.add(readImageFromResources(R.drawable.d));
        Bitmap bmp = Bitmap.createBitmap(images.get(0).cols(), images.get(0).rows(), Bitmap.Config.ARGB_8888);
        Utils.matToBitmap(images.get((0+images.size())/2), bmp);
        originalImage.setImageBitmap(bmp);
        // PROCESS EXPOSURE FUSION
        // w_c, w_s, w_e : weights for contrast, saturation, well-exposedness
        // depth : pyramid depth
        Mat resultImage = exposureFusion(images,1,1,1,3);

        // CHANGE MAT TO BITMAP
        //Imgproc.cvtColor(resultImage, resultImage, Imgproc.COLOR_BGR2RGB);
         bmp = Bitmap.createBitmap(resultImage.cols(), resultImage.rows(), Bitmap.Config.ARGB_8888);
        Utils.matToBitmap(resultImage, bmp);
        hdrImageView.setImageBitmap(bmp);
             */

    }

    /*private Mat exposureFusion (List<Mat> images, double w_c, double w_s, double w_e, int depth)
    {
        if(images.size()<2)
        {
            //print("Input has to be a list of at least two images");
            return null;
        }

        Size size = images.get(0).size();
        for(int i = 1; i < images.size(); i++)
        {
            if(!images.get(i).size().equals(size))
            {
                //print("Input images have to be of the same size");
                return null;
            }
        }
        int r = (int) size.height;
        int c = (int) size.width;
        int N = images.size();




        // COMPUTE WEIGHT MAPS
        List<Mat> weights = computeWeights(images,w_c,w_s,w_e);


        // COMBINE WEIGHT MAP WITH PYRAMIDS
        List<List<Mat>> lps = new ArrayList<List<Mat>>();
        List<List<Mat>> gps = new ArrayList<List<Mat>>();

        for(int i=0;i<N;i++)
        {
            lps.add(laplacianPyramid(images.get(i),depth));
            gps.add(gaussianPyramid(weights.get(i),depth));
        }

        //System.out.println("gaussianPyramid length is "+gps.get(0).size());
        //System.out.println("laplacianPyramid length is "+lps.get(0).size());


        List<Mat> pyramid = new ArrayList<Mat>();

        for(int lev=0;lev<depth;lev++)
        {
            Mat ls = Mat.zeros(lps.get(0).get(lev).size(),CV_8UC3);
            for(int i=0;i<N;i++)
            {
                Mat lp = lps.get(i).get(lev);
                //8UC4?
                System.out.println("Before convertTo lp type is "+lp.type());

                //32FC4
                lp.convertTo(lp,CV_32FC3);

                System.out.println("After convertTo lp type is "+lp.type());

                Mat gpsFloat = new Mat(gps.get(i).get(lev).size(),CV_32F);
                gps.get(i).get(lev).convertTo(gpsFloat,CV_32FC1,(1.0/255.0));
                List<Mat> gpList = new ArrayList<Mat>();
                Mat gpsOnes = Mat.ones(gps.get(i).get(lev).size(),CV_32FC1);
                Mat gpsValues = new Mat(gps.get(i).get(lev).size(),CV_32FC1);
                Core.multiply(gpsOnes,Scalar.all(0.5),gpsValues);
                gpList.add(gpsFloat);
                gpList.add(gpsFloat);
                gpList.add(gpsFloat);
                Mat gp = new Mat(gpsFloat.size(),CV_32FC3);
                Core.merge(gpList,gp);
                Mat lp_gp = new Mat(lp.size(),CV_32FC3);
                System.out.println("lp size is "+lp.rows()+", "+lp.cols());
                System.out.println("gp size is "+gp.rows()+", "+gp.cols());
                System.out.println("lg_gp size is "+lp_gp.rows()+", "+lp_gp.cols());
                //lp is 32FC4???
                System.out.println("lp type is "+lp.type());
                System.out.println("gp type is "+gp.type());
                System.out.println("lp_gp type is "+lp_gp.type());


                Core.multiply(lp,gp,lp_gp,255,CV_8UC3);
                Core.add(ls,lp_gp,ls);
            }
            pyramid.add(ls);
        }


        // COLLAPSE PYRAMID
        Mat resultImage = pyramid.get(depth-1);

        for(int i=depth-2;i>=0;i--)
        {
            Core.add(image_expand(resultImage, new Size(pyramid.get(i).width(), pyramid.get(i).height())),pyramid.get(i),resultImage);
        }


        return resultImage;
    }

    private Mat getGaussianKernel(){
        return Imgproc.getGaussianKernel(5, 0.4);
    }

    private Mat image_reduce(Mat image){
        Mat kernel = getGaussianKernel();
        Mat result = new Mat(image.size(), CV_32FC3);
        Imgproc.filter2D(image, result, -1 ,kernel);
        Imgproc.resize(result, result, new Size() ,0.5, 0.5, Imgproc.INTER_LINEAR);
        return result;
    }

    private Mat image_expand(Mat image, Size size){
        Mat kernel = getGaussianKernel();
        Mat clone = image.clone();
        Mat result = new Mat(size, CV_32FC3);
        Imgproc.resize(clone, clone, size);
        Imgproc.filter2D(clone,result,-1,kernel);
        return result;
    }

    private List<Mat> gaussianPyramid(Mat mat,int depth)
    {
        Mat copy = mat.clone();
        List<Mat> pyramid = new ArrayList<Mat>();
        pyramid.add(copy);
        for(int i =0;i<depth;i++)
        {
            //Mat m = new Mat((mat.rows() + 1) / 2, (mat.cols() + 1) / 2, mat.type());
            //Imgproc.pyrDown(mat, m);
            copy = image_reduce(copy);
            pyramid.add(copy);
            //mat = m;
        }
        return pyramid;
    }

    private List<Mat> laplacianPyramid(Mat mat,int depth)
    {
        List<Mat> pyramid = new ArrayList<Mat>();
        List<Mat> gPyramid = gaussianPyramid(mat,depth+1);
        pyramid.add(gPyramid.get(depth-1));
        for(int i=depth-1;i>=1;i--)
        {
            Mat m = gPyramid.get(i);
            //Mat dst = new Mat();//new Mat(gPyramid.get(i-1).rows(),gPyramid.get(i-1).cols(),CV_32FC1);
            Mat dst = image_expand(m,new Size(gPyramid.get(i-1).width(), gPyramid.get(i-1).height()));
            //Imgproc.pyrUp(m, dst, new Size(gPyramid.get(i-1).width(), gPyramid.get(i-1).height()));
            System.out.println(dst.rows()+" "+dst.cols());
            System.out.println(gPyramid.get(i-1).rows()+" "+gPyramid.get(i-1).cols());
            Core.subtract(gPyramid.get(i-1),dst,dst);
            pyramid.add(0,dst);
        }
        return pyramid;
    }


    private List<Mat> computeWeights (List<Mat> images, double w_c, double w_s, double w_e) {

        int r = images.get(0).rows();
        int c = images.get(0).cols();
        double regularization = 1.0;

        List<Mat> weights = new ArrayList<Mat>();
        Mat weightsSum = Mat.zeros(images.get(0).size(),CV_32FC1);

        for(int i = 0; i < images.size(); i++)
        {
            Mat img = images.get(i);    //8UC3 right now

            Mat imgFloat = new Mat(r,c,CV_32FC3);
            img.convertTo(imgFloat,CV_32FC3,(1.0/255.0));

            Mat grayScale = new Mat(r,c,CV_8UC1);
            Imgproc.cvtColor(img,grayScale,Imgproc.COLOR_BGR2GRAY);



            Mat weight = Mat.ones(r,c,CV_32FC1);

            //CONTRAST
            Mat imgGray = new Mat(r,c,CV_32FC1); //Float version
            grayScale.convertTo(imgGray, CV_32FC1, (1.0/255.0));
            Mat imgLaplacian = new Mat(r,c,CV_32FC1);
            Imgproc.Laplacian(imgGray,imgLaplacian,CV_32FC1, 1, 1, 0, BORDER_REPLICATE);
            Mat contrastWeight = new Mat(r,c,CV_32FC1);
            Core.absdiff(imgLaplacian,Scalar.all(0),contrastWeight);
            Core.pow(contrastWeight,w_c,contrastWeight);
            Core.add(contrastWeight,Scalar.all(regularization),contrastWeight);
            Core.multiply(weight,contrastWeight,weight);

            //SATURATION
            Mat saturationWeight = new Mat(r,c,CV_32FC1);
            List<Mat> rgbChannels = new ArrayList<Mat>();
            Core.split(imgFloat,rgbChannels);
            Mat R = rgbChannels.get(0);
            Mat G = rgbChannels.get(1);
            Mat B = rgbChannels.get(2);
            Mat mean = Mat.zeros(r,c,CV_32FC1);
            Core.add(mean,R,mean);
            Core.add(mean,G,mean);
            Core.add(mean,B,mean);
            Core.divide(mean,Scalar.all(3),mean);
            Core.subtract(R,mean,R);
            Core.subtract(G,mean,G);
            Core.subtract(B,mean,B);
            Core.pow(R,2,R);
            Core.pow(G,2,G);
            Core.pow(B,2,B);
            Mat std = Mat.zeros(r,c,CV_32FC1);
            Core.add(std,R,std);
            Core.add(std,G,std);
            Core.add(std,B,std);
            Core.divide(std,Scalar.all(3),std);
            Core.pow(std,0.5,std);
            Core.pow(std, w_s, saturationWeight);
            Core.add(saturationWeight,Scalar.all(regularization),saturationWeight);
            Core.multiply(weight,saturationWeight,weight);

            //WELL-EXPOSEDNESS

            double sigma = 0.2;
            Mat gaussianCurve = imgFloat.clone();
            Core.subtract(gaussianCurve,Scalar.all(0.5),gaussianCurve);
            Core.pow(gaussianCurve,2,gaussianCurve);
            //Core.divide(gaussianCurve,Scalar.all(-2 * sigma2),gaussianCurve);
            Core.divide(gaussianCurve,Scalar.all(-2 * sigma * sigma),gaussianCurve);
            Core.exp(gaussianCurve,gaussianCurve);
            List<Mat> rgbGaussianCurves = new ArrayList<Mat>();
            Core.split(gaussianCurve,rgbGaussianCurves);
            Mat exposednessWeight = Mat.ones(r,c,CV_32FC1);
            Core.multiply(exposednessWeight,rgbGaussianCurves.get(0),exposednessWeight);
            Core.multiply(exposednessWeight,rgbGaussianCurves.get(1),exposednessWeight);
            Core.multiply(exposednessWeight,rgbGaussianCurves.get(2),exposednessWeight);
            Core.pow(exposednessWeight, w_e, exposednessWeight);
            Core.add(exposednessWeight,Scalar.all(regularization),exposednessWeight);
            Core.multiply(weight,exposednessWeight,weight);

            Core.add(weightsSum,weight,weightsSum);
            weights.add(weight);
        }

        Core.add(weightsSum,Scalar.all(1e-12),weightsSum);

        for(int i = 0; i<weights.size(); i++)
        {
            Mat normWeight = new Mat(r,c,CV_32FC1);
            Core.divide(weights.get(i),weightsSum,normWeight);
            weights.set(i,normWeight);
        }

        return weights;

    }

    private Mat readImageFromResources(int data) {
        Mat img = null;
        try {
            img = Utils.loadResource(this, data);
            Imgproc.cvtColor(img, img, Imgproc.COLOR_RGB2BGR);
        } catch (IOException e) {
            e.printStackTrace();
            //Log.e(TAG,Log.getStackTraceString(e));
        }
        return img;
    }*/

}
