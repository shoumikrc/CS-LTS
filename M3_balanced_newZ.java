/*
Copyright (c) 2017, Shoumik Roychoudhury, DABI, Temple University
All rights reserved.

Redistribution and use in source and binary forms, with or without
modification, are permitted provided that the following conditions are met:
    * Redistributions of source code must retain the above copyright
      notice, this list of conditions and the following disclaimer.
    * Redistributions in binary form must reproduce the above copyright
      notice, this list of conditions and the following disclaimer in the
      documentation and/or other materials provided with the distribution.
    * Neither DABI, Temple Univeristy nor the
      names of its contributors may be used to endorse or promote products
      derived from this software without specific prior written permission.

THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS" AND
ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED
WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
DISCLAIMED. IN NO EVENT SHALL <COPYRIGHT HOLDER> BE LIABLE FOR ANY
DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES
(INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES;
LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND
ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
(INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS
SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
 */
package costSensitive;

// Required Packages
import java.util.ArrayList;
import java.util.Collections;
import java.util.List;
import java.util.Random;
import Utilities.Logging;
import Utilities.Logging.LogLevel;
import Utilities.Sigmoid;
import Utilities.StatisticalUtilities;
import Clustering.KMeans;
import DataStructures.DataSet;
import DataStructures.Matrix;
import java.io.File;
import java.io.FileNotFoundException;
import java.io.FileOutputStream;
import java.io.PrintStream;
import org.happy.commons.concurrent.loops.ForEachTask_1x0;
import org.happy.commons.concurrent.loops.Parallel_1x0;
/**
 *
 * @author shoumik
 */
public class M3_balanced_newZ {
    double del;
    double[] data;
    int size;
    long seed; 
    double ss;
    double c_FP;// Misclassification costs
    double u; //variable for logarithm of misclassificaiton cost c_FP
    double theta;
    double delta;
    public int ITrain, ITest; // number of training and testing instances
    public int Q; // length of a time-series
        
    // length of shapelet
    public int L[]; 
    public int L_min;
        
    public int K; // number of latent patterns
    public int R; // scales of the shapelet length
    public int C; // number of classes
    public int J[]; // number of segments
    double Shapelets[][][]; // shapelets

    // classification weights
    double W[][];
    double biasW;
    
    // accumulate the gradients
    double GradHistShapelets[][][];
    double GradHistW[][];
    
    double GradHistBiasW;
    double CFPHist;
    double deltaHist;
	
    // the softmax parameter
    public double alpha;
	
    // time series data and the label 
    public Matrix T;
    public Matrix Y; //Y_b;
		
    public int maxIter;// the number of iterations
    public double eta; // the learning rate
    public int kMeansIter;
    public double lambdaW;// the regularization parameters
	
    public List<Double> nominalLabels;
	
    // structures for storing the precomputed terms
    double D[][][][];
    double E[][][][]; 
    double M[][][];
    double Psi[][][]; 
    double sigY[]; 
    double sigZ[];

    Random rand = new Random();
       	
    List<Integer> instanceIdxs;
    List<Integer> rIdxs;
	
    // constructor
    public M3_balanced_newZ (){
    	kMeansIter = 100;
    }
    // initialize the data structures
    public void Initialize(){ 
	// avoid K=0 
	if(K == 0) 
            K = 1;
              
		
		
	// set the labels to be binary 0 and 1, needed for the logistic loss
	for(int i = 0; i < ITrain+ITest; i++)
            if(Y.get(i) != 1.0) 
                Y.set(i, 0, 0.0);
	C = nominalLabels.size(); 
        
         double positive = 0;
                double negative = 0;
		for(int i = 0; i < ITrain; i++){
                    if(Y.get(i) == 1)
                        positive = positive+1;
                    if(Y.get(i) == 0)
                        negative = negative+1;
                }
                System.out.println("positive = " + positive);
                System.out.println("negative = " + negative);
                theta = negative/positive;
            // initialize the shapelets (complete initialization during the clustering)
        Shapelets = new double[R][][];
		
        GradHistShapelets = new double[R][][];
		
        // initialize the number of shapelets and the length of the shapelets 
        J = new int[R]; 
        L = new int[R];
        // set the lengths of shapelets and the number of segments
        // at each scale r
        int totalSegments = 0;
        for(int r = 0; r < R; r++){
            L[r] = (r+1)*L_min;
            J[r] = Q - L[r];
            totalSegments += ITrain*J[r]; 
        }
            
		// set the total number of shapelets per scale as a rule of thumb 
		// to the logarithm of the total segments
	if( K < 0)
            K = (int) Math.log(totalSegments) * (C-1); 
        Logging.println("Modified LTS");
        Logging.println("ITrain="+ITrain + ", ITest="+ITest + ", Q="+Q + ", Classes="+C, LogLevel.DEBUGGING_LOG);
        Logging.println("K="+K + ", L_min="+ L_min + ", R="+R, LogLevel.DEBUGGING_LOG);
        Logging.println("eta=" + eta + ", maxIter="+ maxIter, LogLevel.DEBUGGING_LOG);
        Logging.println("lambdaW="+lambdaW + ", alpha="+ alpha, LogLevel.DEBUGGING_LOG);
        Logging.println("totalSegments="+totalSegments + ", K="+ K, LogLevel.DEBUGGING_LOG);
		
            // initialize an array of the sizes
        rIdxs = new ArrayList<Integer>();
        for(int r = 0; r < R; r++)
            rIdxs.add(r);
		
		// initialize shapelets
        InitializeShapeletsKMeans();
		
            // initialize the terms for pre-computation
        D = new double[ITrain+ITest][R][K][];
        E = new double[ITrain+ITest][R][K][];
		
        for(int i=0; i <ITrain+ITest; i++)
            for(int r = 0; r < R; r++)
                for(int k = 0; k < K; k++){
                    D[i][r][k] = new double[J[r]];
                    E[i][r][k] = new double[J[r]];
                }
		
            // initialize the placeholders for the precomputed values
        M = new double[ITrain+ITest][R][K];
        Psi = new double[ITrain+ITest][R][K];
        sigY = new double[ITrain+ITest];
        sigZ = new double[ITrain+ITest];
		
        // initialize the weights
		
        W = new double[R][K];

		
        GradHistW = new double[R][K];
       
                		
            
            rand.setSeed(seed);
            for(int r = 0; r < R; r++){
		for(int k = 0; k < K; k++){
                    rand.setSeed(seed);
                    W[r][k] = 2*rand.nextDouble()-1;
                    GradHistW[r][k] = 0;
		}
            }
			
		biasW = 2*rand.nextDouble()-1;
                c_FP = 1; //5*rand.nextDouble()-1;
                delta = del;//theta*rand.nextDouble();
                //System.out.println("delta = " + " "  + delta[c]);
                //System.out.println("SEED = " + seed);
                u = Math.log(c_FP);
                //v[c] = Math.log(delta[c]);
                GradHistBiasW = 0;
                CFPHist= 0;
                deltaHist= 0;
                        
		
	
		// precompute the M, Psi, sigY, used later for setting initial W
		for(int i=0; i < ITrain+ITest; i++)
			PreCompute(i); 
		
		// store all the instances indexes for
		instanceIdxs = new ArrayList<Integer>();
		for(int i = 0; i < ITrain; i++)
			instanceIdxs.add(i);
		// shuffle the order for a better convergence
		Collections.shuffle(instanceIdxs, rand); 
		
		//Logging.println("Initializations Completed!", LogLevel.DEBUGGING_LOG);
    }


// initialize the shapelets from the centroids of the segments
	public void InitializeShapeletsKMeans(){		
		// a multi-threaded parallel implementation of the clustering
		// on thread for each scale r, i.e. for each set of K shapelets at
		// length L_min*(r+1)
		Parallel_1x0.ForEach(rIdxs, new ForEachTask_1x0<Integer>(){
			public void iteration(Integer r){
				//Logging.println("Initialize Shapelets: r="+r+", J[r]="+J[r]+", L[r]="+L[r], LogLevel.DEBUGGING_LOG);
				
				double [][] segmentsR = new double[ITrain*J[r]][L[r]];
				
				for(int i= 0; i < ITrain; i++) 
					for(int j= 0; j < J[r]; j++) 
						for(int l = 0; l < L[r]; l++)
							segmentsR[i*J[r] + j][l] = T.get(i, j+l);
	
				// normalize segments
				for(int i= 0; i < ITrain; i++) 
					for(int j= 0; j < J[r]; j++) 
						for(int l = 0; l < L[r]; l++)
							segmentsR[i*J[r] + j] = StatisticalUtilities.Normalize(segmentsR[i*J[r] + j]);
				
				
				KMeans kmeans = new KMeans();
				Shapelets[r] = kmeans.InitializeKMeansPP(segmentsR, K, 100); 
				
                                //System.out.print(Shapelets[r]);
				// initialize the gradient history of shapelets
				GradHistShapelets[r] = new double[K][ L[r] ]; 
				for(int k= 0; k < K; k++) 
					for(int l = 0; l < L[r]; l++)
						GradHistShapelets[r][k][l] = 0.0; 
				
				
				if( Shapelets[r] == null)
					System.out.println("P not set");
			}
		});
	}
	// predict the label value vartheta_i
        public double linearPredict(int i){
		double Y_hat_ic = biasW;
               	for(int r = 0; r < R; r++)
			for(int k = 0; k < K; k++)
                            Y_hat_ic += M[i][r][k] * W[r][k];
                return  Y_hat_ic;
	}
	public double Predict(int i){
		double Y_hat_ic = biasW;
                double z = 0;
                for(int r = 0; r < R; r++){
			for(int k = 0; k < K; k++){
                                Y_hat_ic += M[i][r][k] * W[r][k];
                        }
                }
                z = (1/(((theta*c_FP) + delta)+c_FP))*(Y_hat_ic + Math.log((theta*c_FP + delta)/c_FP));
              
                return z;
	}
        // Method to precompute some paramemters  	
        public void PreCompute(int i){
		// precompute terms
		for(int r = 0; r < R; r++){
			for(int k = 0; k < K; k++){
				for(int j = 0; j < J[r]; j++){
					// precompute D
					D[i][r][k][j] = 0;
					double err = 0;
					for(int l = 0; l < L[r]; l++){
                                                err = T.get(i, j+l)- Shapelets[r][k][l];
                                                D[i][r][k][j] += err*err;
                                        }
					
					D[i][r][k][j] /= (double)L[r]; 
					//D[i][r][k][j] = Math.sqrt(D[i][r][k][j]);
					// precompute E
					E[i][r][k][j] = Math.exp(alpha * D[i][r][k][j]);
				}
				
				// precompute Psi 
				Psi[i][r][k] = 0; 
				for(int j = 0; j < J[r]; j++) 
					Psi[i][r][k] +=  Math.exp( alpha * D[i][r][k][j] );
				
				// precompute M 
				M[i][r][k] = 0;
				
				for(int j = 0; j < J[r]; j++)
					M[i][r][k] += D[i][r][k][j]* E[i][r][k][j];
				
                                
				M[i][r][k] /= Psi[i][r][k];
			}
		}
		
		
			sigZ[i] = Sigmoid.Calculate(Predict(i)); 
                        sigY[i] = Sigmoid.Calculate(linearPredict(i));
                
	}
	// compute the MCR on the test set
	private double GetMCRTrainSet() 
	{
		int numErrors = 0;
		
		for(int i = 0; i < ITrain; i++)
		{
			PreCompute(i);
			//double label_i = Sigmoid.Calculate(linearPredict(i));
                        double label_i = Sigmoid.Calculate(Predict(i)); 
			
			if( (Y.get(i) == 1 && label_i < 0.5) || (Y.get(i) == 0 && label_i >= 0.5) )
						numErrors++;
		}
		
		return (double)numErrors/(double)ITrain;
	}
	// compute the MCR on the test set
	private double[] GetMCRTestSet(String predictfile, PrintStream ps2) throws FileNotFoundException{
		int numErrors = 0;
                double TP=0;
                double TN=0; 
                double FP=0;
                double FN = 0;
                double sensitivity = 0;
                double specificity = 0;
                double GM = 0;
                double AGM = 0;
                //double fpr = 0;
                //double fnr = 0;
                double[] prediction = new double[ ITrain+ITest];
                double[] Prob1 = new double[ ITrain+ITest];
                double[] Prob2 = new double[ ITrain+ITest];
                double[] yhat1 = new double[ ITrain+ITest];
                double[] yhat2 = new double[ ITrain+ITest];
		FileOutputStream fos2 = new FileOutputStream(predictfile);
		
		for(int i = ITrain; i < ITrain+ITest; i++)
		{
			PreCompute(i);
			//double label_i = Sigmoid.Calculate(linearPredict(i));
                        double label_i = Sigmoid.Calculate(Predict(i)); 
                         Prob1[i] = label_i;
                         Prob2[i] = 1 - label_i;
                         
                         
			//prediction[i] = label_i;
			if( (Y.get(i) == 1 && label_i < 0.5) || (Y.get(i) == 0 && label_i >= 0.5) )
                        {
                            numErrors++;
                                                
                            
                            if((Y.get(i) == 1 && label_i < 0.5))
                            {
                                                 
                                FN++;
                                prediction[i] = 0;
                            }
                            else if((Y.get(i) == 0 && label_i >= 0.5))
                            {
                                FP++;
                                prediction[i] = 1;
                            }
				
                        }
                        if((Y.get(i) == 1 && label_i >= 0.5) || (Y.get(i) == 0 && label_i < 0.5))
                        {
                            if((Y.get(i) == 0 && label_i < 0.5))
                            {
                                TN++;
                                prediction[i] = 0;
                            }
                            else if((Y.get(i) == 1 && label_i >= 0.5))
                            {
                                TP++;
                                prediction[i] = 1;
                        }
                            
                        }
                        //ps2.print("Positive Class:" + " " + Prob1[i] + " " + yhat1[i] + " " + "Negative Class:" + " " + Prob2[i] + " " + yhat2[i] + " "+ "Original:" + " " + nominalLabels.indexOf(Y.get(i)) + " " + "Predicted:" + " " + Predict[i]);
                        ps2.print("Positive Class:" + " " + Prob1[i] + " " + "Negative Class:" + " " + Prob2[i] + " "+ "Original:" + " " + (Y.get(i)) + " " + "Predicted:" + " " + prediction[i]);
                        ps2.println();
		}
                    double accuracy = (TP+TN)/(TP+FP+FN+TN);
                    //double accuracy = 1 - (double)numErrors/(double)ITest; 
                    int[] beta = {1,2,3};
                    double[] weighted_sensitivity = {0,0,0};
                    double[] weighted_Accuracy = {0,0,0};
                    double[] weighted_F = {0,0,0};
                    sensitivity = TP/(TP+FN);
                    //fnr = 1 - sensitivity;
                    specificity = TN/(TN+FP);
                    GM = Math.sqrt((TP/(TP+FN))*(TN/(FP+TN)));
                    if(sensitivity>0)
                        AGM = (GM+specificity*(FP+TN))/(1+FP+TN);
                    if(sensitivity==0)
                        AGM = 0;
                    //fpr = 1 - specificity;
                    for(int loop1 = 0;loop1<beta.length;loop1++){
                        weighted_sensitivity[loop1] = TP/(TP+(1+Math.pow(beta[loop1], 2)*FN));
                        weighted_Accuracy[loop1] = (weighted_sensitivity[loop1]+specificity) /2;
                        weighted_F[loop1] = ((1+Math.pow(beta[loop1], 2))*TP)/(((1+Math.pow(beta[loop1], 2))*TP) + ((Math.pow(beta[loop1], 2))*FN) + FP);
                    }
                    
                    ps2.println("Sensitivity:" + " " + sensitivity + " " + "Specificity:" + " " + specificity + " " + "F1: " + " " + weighted_F[0] + " " + "F2: " + " " + weighted_F[1] + " " + "F3: " + " " + weighted_F[2] + " " + "GM: " + " " + GM + " " + "AGM: " + " " + AGM);
                    ps2.println();
                
                
		
                   return new double[]{(double)numErrors/(double)ITest,sensitivity,specificity,accuracy,weighted_F[0],weighted_F[1],weighted_F[2], GM, AGM};
	
		//return new double[]{numErrors,sensitivity,specificity,weighted_Accuracy[0],weighted_Accuracy[1],weighted_Accuracy[2]};
	}
	// compute the accuracy loss of instance i according to the 
	// smooth hinge loss 
	public double AccuracyLoss(int i){
		double Y_hat_ic = Predict(i);
                double sig_y_ic = Sigmoid.Calculate(Y_hat_ic);
                double sig_y_ic1 = Sigmoid.Calculate(Y_hat_ic,(theta*c_FP+delta));
                double sig_y_ic2 = Sigmoid.Calculate(Y_hat_ic,(c_FP));
                return -Y.get(i)*Math.log(sig_y_ic1) - (1-Y.get(i))*Math.log(1-sig_y_ic2); 
               
	}
	// compute the accuracy loss of the train set
	public double AccuracyLossTrainSet(){
		double accuracyLoss = 0;
		
		for(int i = 0; i < ITrain ; i++){
			PreCompute(i);
		
			
				accuracyLoss += AccuracyLoss(i);
                        
                        //System.out.println("example" +  " " + i);
                        //System.out.println("accuracyLoss = " + accuracyLoss);
                        
		}
		
		return accuracyLoss;
	}
	// compute the accuracy loss of the train set
	public double AccuracyLossTestSet(){
		double accuracyLoss = 0;
		
		for(int i = ITrain; i < ITrain+ITest; i++){
			PreCompute(i);
			
			
				accuracyLoss += AccuracyLoss(i); 
		}
		return accuracyLoss;
	}
	//method for gradient computation
	public void LearnF(){ 
		// parallel implementation of the learning, one thread per instance
		// up to as much threads as JVM allows
		Parallel_1x0.ForEach(instanceIdxs, new ForEachTask_1x0<Integer>(){
			public void iteration(Integer i){
				double regWConst = ((double)2.0*lambdaW) / ((double) ITrain);
				double zss = 0, dss = 0, tmp2 = 0, tmp1 = 0, dLdY = 0, dMdS=0, gradS_rkl = 0, gradBiasW_c = 0, gradW_crk = 0,dzdy = 0, gradC_FN = 0, gradu = 0,gradv = 0,gradtheta = 0; 
				double eps = 0.000001;
                                
                                        double diffc = 0;
					PreCompute(i);
                                        double Y_hat_ic = Predict(i);
                                        double sig_y_ic1 = Sigmoid.Calculate(Y_hat_ic,(theta*c_FP+delta));
                                        double sig_y_ic2 = Sigmoid.Calculate(Y_hat_ic,(c_FP));
					dLdY = ((1-Y.get(i))*sig_y_ic2*c_FP)-((Y.get(i))*(1-sig_y_ic1)*(theta*c_FP+delta));
                                        dzdy = 1/(c_FP+(c_FP*theta+delta));
                                        //dzdy = 1/(c_FP*(c_FP*theta+delta));
                                        for(int r = 0; r < R; r++){
						for(int k = 0; k < K; k++){
							// gradient with respect to W_crk
							gradW_crk = dLdY*dzdy*M[i][r][k] + regWConst*W[r][k];
							// add gradient square to the history
							GradHistW[r][k] += gradW_crk*gradW_crk;
							
							// update the weights
                                                        double temp =  W[r][k];
							W[r][k] -= (eta / ( Math.sqrt(GradHistW[r][k]) + eps))*gradW_crk; 
                                                        double diff = W[r][k] - temp;
							diffc += Math.pow(diff, 2);
							tmp1 = ( 2.0 / ( (double) L[r] * Psi[i][r][k]) );
							
							for(int l = 0; l < L[r]; l++){
								tmp2=0;
								for(int j = 0; j < J[r]; j++)
									tmp2 += E[i][r][k][j]*(1 + alpha*(D[i][r][k][j] - M[i][r][k]))*(Shapelets[r][k][l] - T.get(i, j+l));
								
								gradS_rkl =  dLdY*dzdy*W[r][k]*tmp1*tmp2;
								
								// add the gradient to the history
								GradHistShapelets[r][k][l] += gradS_rkl*gradS_rkl;
								Shapelets[r][k][l] -= (eta / ( Math.sqrt(GradHistShapelets[r][k][l]) + eps))*gradS_rkl;
								
							}				
						}
					}
                                        ss = Math.sqrt(diffc);
					// the gradient 
					gradBiasW_c = dLdY*dzdy;
                                        //zss = -(1/Math.pow(((theta*c_FP+delta)+c_FP),2))*(linearPredict(i) + theta*Math.log(theta*c_FP+delta) - (theta*((theta*c_FP+delta) +c_FP)/(theta*c_FP+delta)) - theta*Math.log(c_FP) + ((theta*c_FP+delta) +c_FP)/(c_FP) );
                                        zss = -(1/Math.pow(((theta*c_FP+delta)+c_FP),2))*(linearPredict(i) + theta*Math.log(theta*c_FP+delta) - (theta*((theta*c_FP+delta) +c_FP)/(theta*c_FP+delta)) - theta*Math.log(c_FP) + ((theta*c_FP+delta) +c_FP)/(c_FP) );

                                        gradu = Math.exp(u)*(((1-Y.get(i))*sig_y_ic2*(Predict(i)+c_FP*zss)) - (Y.get(i)*(1-sig_y_ic1)*(theta*Predict(i)+(theta*c_FP+delta)*zss))) ;
                                          
                                        // add the gradient to the history
					GradHistBiasW += gradBiasW_c*gradBiasW_c;
                                        CFPHist+=gradu*gradu;
                                        
					biasW -= (eta / ( Math.sqrt(GradHistBiasW) + eps))*gradBiasW_c; 
                                        u -= (eta / ( Math.sqrt(CFPHist) + eps))*gradu;
                                        
                                        dss = -(1/Math.pow(((theta*c_FP+delta)+c_FP),2))*(linearPredict(i) + Math.log(theta*c_FP+delta) - (((theta*c_FP+delta) +c_FP)/(theta*c_FP+delta)) - Math.log(c_FP) );
                                        gradv = (((1-Y.get(i))*sig_y_ic2*(c_FP*dss))- Y.get(i)*(1-sig_y_ic1)*(Predict(i)+(theta*c_FP+delta)*dss));
                                        deltaHist+=gradv*gradv;
                                        delta -= (eta / ( Math.sqrt(deltaHist) + eps))*gradv;
                                        c_FP = Math.exp(u);
                                        
                                			 	
                        }
		
		});
	}
	// optimize the objective function
	//public double[] Learn(String predictfile) throws FileNotFoundException
        public void Learn(String predictfile, PrintStream ps2) throws FileNotFoundException{
		// initialize the data structures
		Initialize();
                //PrintShapeletsAndWeights();
				
		//List<Double> lossHistory = new ArrayList<Double>();
                List<Double> lossHistory1 = new ArrayList<Double>();
                List<Double> errorTrain = new ArrayList<Double>();
		//lossHistory.add(Double.MIN_VALUE);
                lossHistory1.add(Double.MIN_VALUE);
                errorTrain.add(Double.MIN_VALUE);
		int indLossHist = 0;
                int indLossHist1 = 0;
		// apply the stochastic gradient descent in a series of iterations
		for(int iter = 0; iter <= maxIter; iter++){
			// learn the latent matrices
			LearnF(); 
			
			// measure the loss
			     
                        if( iter % 200 == 0){
				double mcrTrain1 = GetMCRTrainSet();
				double mcrTest1[] = GetMCRTestSet(predictfile, ps2); 
				//int i = iter;
				double lossTrain1 = AccuracyLossTrainSet();
				double lossTest1 = AccuracyLossTestSet();
				lossHistory1.add(lossTrain1);
                                errorTrain.add(mcrTrain1);
                                indLossHist1++;
                                                             
                                
                                //System.out.println(lossTest);
				//if(iter == 0)
                                Logging.println("It=" + iter + ", alpha= "+alpha+", lossTrain="+ lossTrain1 + ", lossTest="+ lossTest1  +
					", errorTrain=" +mcrTrain1*100 + ", errorTest=" +mcrTest1[0]*100 //+ ", SVM=" + mcrSVMTest
					, LogLevel.DEBUGGING_LOG);
				
				//System.out.println( eta/Math.sqrt(GradHistBiasW[0]) );
				//System.out.println( eta/Math.sqrt(GradHistW[0][1][5]) );
				
				// if divergence is detected start from the beggining 
				// at a lower learning rate
                                if( Double.isNaN(lossTrain1) || mcrTrain1 == 1.0 ){
					iter = 0;
					eta /= 3;
					lossHistory1.clear();
                                        errorTrain.clear();
					indLossHist1 = 0;
					Initialize();
					
					Logging.println("Divergence detected. Restarting at eta=" + eta, LogLevel.DEBUGGING_LOG);
				}
                                
				if( lossHistory1.size() > 100 ) 
					if( lossTrain1 > lossHistory1.get( lossHistory1.size() - 2  ) || mcrTrain1>errorTrain.get(errorTrain.size()-2) )
						break;
                                if(mcrTrain1*100 == 0.0)
                                    break;
				
			}
                        
                }
		
		//return GetMCRTestSet(predictfile); 
	}
        
        //Printing methods for Cost shapelets shapelet-transformed data. 
	public void PrintCost(String outputfile) throws FileNotFoundException{
                FileOutputStream fos = new FileOutputStream(outputfile);
                PrintStream ps = new PrintStream(fos);
                //for(int c = 0; c < C; c++){
			//for(int r = 0; r < R; r++)
			//{
                            //System.out.print(" [ ");
                            //System.out.print(c_FP[c]*theta+delta[c] + " ");
                            ps.print("C_FN" + " " + "C_FP" );
                            ps.println();
                            ps.print(c_FP*theta+delta + " ");
                            //System.out.print(c_FP[c]*theta + " ");
                            //System.out.print(c_FP[c] + " ");
                            ps.print(c_FP + " ");
                            ps.println();
                            ps.println(c_FP/(c_FP*theta+delta + c_FP));
                            
                            //System.out.print(delta[c] + " ");
                            //System.out.println("]");
                            
                        //}
                //}
                ps.close();
        }
        public void PrintShapeletsAndWeights(String outputfile) throws FileNotFoundException{
                FileOutputStream fos = new FileOutputStream(outputfile);
                PrintStream ps = new PrintStream(fos);
		for(int r = 0; r < R; r++){
			for(int k = 0; k < K; k++){
				//System.out.print("Shapelets("+r+","+k+")= [ ");
                                //ps.print("Shapelets("+r+","+k+")= [ ");
				
				for(int l = 0; l < L[r]; l++){
					//System.out.print(Shapelets[r][k][l] + " ");
                                        ps.print(Shapelets[r][k][l] + " ");
				}
				
				//System.out.println("]");
                                ps.println();
			}
		}
                ps.println();
		//for(int c = 0; c < C; c++){
			for(int r = 0; r < R; r++){
				//System.out.print("W("+c+","+r+")= [ ");
                                //ps.print("W("+c+","+r+")= [ ");
				
                                for(int k = 0; k < K; k++){
					//System.out.print(W[c][r][k] + " ");
                                         ps.print(W[r][k] + " ");
                                }
				
				//System.out.print(biasW[c] + " ");
                                ps.print(biasW + " ");
				//System.out.println("]");
                                //ps.println("]");
			}
		//}
                ps.close();
	}
	public void PrintProjectedData(String outputfile1) throws FileNotFoundException{
                FileOutputStream fos4 = new FileOutputStream(outputfile1);
                PrintStream ps4 = new PrintStream(fos4);
		int r = 0, c = 0;
		
		//System.out.print("Data= [ ");
		
		for(int i = 0; i < ITrain +ITest; i++){
			PreCompute(i); 
			
			//System.out.print(Y_b.get(i, c) + " "); 
			
			for(int k = 0; k < K; k++){
				//System.out.print(M[i][r][k] + " ");
                                ps4.print(M[i][r][k]+ " ");
			}
                        
                        ps4.println();
			
			//System.out.println(";");
		}
                
		ps4.close();
		//System.out.println("];");
	}
	
        // Statistical methods for computing mean and standard deviation
        private void Statistics(double[] data){
                this.data = data;
                size = data.length;
        }   
        public double getMean(double[] m){
            double sum = 0;
            for (int i = 0; i < m.length; i++) {
                    sum += m[i];
            }
            return sum / m.length;
    }
        public double getVariance(double [] v){
            double mean = getMean(v);
            double temp = 0;
            for(double a:data)
                    temp += (mean-a)*(mean-a);
            return temp/size;
    }
        public double getStdDev(double [] s){
            return Math.sqrt(getVariance(s));
    }
    
    
	// the main execution of the program
	public static void main(String [] args) throws FileNotFoundException
	{
                // Main outer directory for sensor selection
                String dataSetdirectory = "C:\\shoumik\\DABI\\datasets\\ECML2017conference\\balanced_versionZ";
                File file = new File(dataSetdirectory);
                String[] names = file.list();
                for(String name : names){
                        long par1 = 0;
                        double par2 = 0;
                        double par3 = 0;
                        System.out.println(name);
                        //String name = "NumTweets";
                        //int[] data0 = {1};
                        int cou = 0;
                        double[] sensi = new double[10];
                        double[] spec = new double[10];
                        double[] accu = new double[10];
                        //double[] accu_int = new double[3];
                        double[] F1 = new double[10];
                        double[] F2 = new double[10];
                        double[] F3 = new double[10];
                        double[] F1_int = new double[4]; 
                        double[] GG = new double[10];
                        double[] AGM = new double[10];
                        //int size0 = data0.length;
                        String resultfile = dataSetdirectory + "\\" + name+"\\result" +  ".txt";
                        FileOutputStream fos1 = new FileOutputStream(resultfile);
                        PrintStream ps1 = new PrintStream(fos1);
               
                        for (int loop1 = 0;loop1<1;loop1++){
                                //int ll1 = data0[loop1];
                                //for(int loop2 = 1;loop2<=3;loop2++ ){
                                //String l1 = String.valueOf(ll1);
                                //String l12 = String.valueOf(loop2);
                                String dir = dataSetdirectory+"\\",
                                //String dir =    "C:\\shoumik\\DABI\\datasets\\ucr\\",
                                folder = name + "\\",       
                                ds = name;// + "_" + l1 + "_" + l12;
                                File file1 = new File(dataSetdirectory+ "\\"  + folder + "\\train");
                                String[] names1 = file1.list();
                                String outputfile = dir+folder+ds+"_extractedShapelets.txt";
                                String outputfile1 = dir+folder+ds+"_features.txt";
                                String outputfile4 = dir+folder+ds+"_cost.txt";
                                //String predictfile = "C:\\Users\\shoumik\\Dropbox\\Twitter_Shoumik_Fang\\CIKM\\Experiments\\paper\\section5_2\\Republican\\" + name + "\\" + ds + "\\predict" +  ".txt";
                                String predictfile = dataSetdirectory + "\\" + name + "\\" + "\\predict" +  ".txt";
                                FileOutputStream fos2 = new FileOutputStream(predictfile);
                                PrintStream ps2 = new PrintStream(fos2);
                              
                
                                if (loop1 == 0){
                            
                                        //String resultfile = "C:\\Users\\shoumik\\Dropbox\\Twitter_Shoumik_Fang\\CIKM\\Experiments\\paper\\section5_2\\Republican\\" + name+"\\result" +  ".txt";
                                        //FileOutputStream fos1 = new FileOutputStream(resultfile);
                                        //PrintStream ps1 = new PrintStream(fos1);
                                        //long[] data1 = {1,10,100,1000,10000,100000,1000000,10000000,100000000};
                                        long[] data1 = {1000};
                                        //double[] data2 = {0.001,0.01,0.1,10,100,1000};
                                        double[] data2 = {0.1};
                                        //double[] data3 = {1,5,10,25,50,100};
                                        double[] data3 = {1000};
                                        int counter = 0;
                                        int size1 = data1.length;
                                        int size2= data2.length;
                                        int size3= data3.length;
                                        
                                        
                                        System.out.println("###########Starting internal crossvalidation######################");
                                        double[]  mean_G = new double[size1*size2*size3];
                                        double[]  std_G = new double[size1*size2*size3];
                                        long[]  param1 = new long[size1*size2*size3];
                                        double[]  param2 = new double[size1*size2*size3];
                                        double[]  param3 = new double[size1*size2*size3];
                                        
                                        param1[0] = data1[0];
                                        param2[0] = data2[0];
                                        param3[0] = data3[0];
                                        /*for(int lo1 = 0;lo1<size1;lo1++){
                                                long l1min = data1[lo1];
                                                //String l_min = String.valueOf(l1min);
                                                for (int l2 = 0;l2<size2;l2++){
                                                        double lamb = data2[l2];
                                                        //String lambda = String.valueOf(lamb);
                                                        for (int l3 = 0;l3<size3;l3++){
                                                                double learn = data3[l3];
                                                                 //String rate = String.valueOf(learn);
                      
                                                                System.out.println(l1min);
                                                                System.out.println(lamb);
                                                                System.out.println(learn);
                      
                                        
                                                                double[] G = new double[3];
                                                                int count = 0;
                                    
                                                                for(String name1 : names1){
                                                                        System.out.println(name1);
                                                                        String ds1 = name1;
                                         
                                                                        String predictfile1 = dataSetdirectory +"\\" + folder + "train\\" + name1 + "\\predict" +  ".txt";
                                                                        FileOutputStream fos3 = new FileOutputStream(predictfile1);
                                                                        PrintStream ps3 = new PrintStream(fos3); 
               
            
                                                                        //if (args.length == 0) {
                                                                                    //String dir = "E:\\Data\\classification\\timeseries\\",
                                                                                    //String ds;
                                                                                    //ds = name;
		
                
                                                                                    //ds = "BirdChicken";no
                                                                        String sp = File.separator; 
                                                                        // System.out.println(ds + " " +l1 + " "+ l2);
                                                                        args = new String[] {  
                                                                                //"trainSet=" + dir + ds + sp + "folds" + sp + "default" + sp  
                                                                                                //+ ds + "_TRAIN",  
                            
                                                                                //"trainSet=" + dir + ds + sp + ds + "_TRAIN.txt",  
                                                                                //"testSet=" + dir + ds + sp + ds + "_TEST.txt",
                                                                                "trainSet=" + dir + folder + "train" + sp + ds1 + sp + ds1 +  "_TRAIN",  
                                                                                "testSet=" + dir + folder  +  "train" + sp + ds1 + sp + ds1 +  "_TEST",
                                                                                //"testSet=" + dir + ds + sp + "folds" + sp + "default" + sp  
                                                                                //		+ ds + "_TEST",  
 				
                                                                                "alpha=-100",
                                                                                "eta=0.01",
                                                                                "maxEpochs=5000",
                                                                                "K=-1",		
                                                                                "L=0.1", 
                                                                                "R=3", 
                                                                                "lambdaW=0.01" 
                                                                        };
					
                                                                //}
                                                                        // values of hyperparameters
                                                                        double eta = -1, lambdaW = -1, alpha = -1, L = -1, K = -1;
                                                                        int maxEpochs = -1, R = -1;
                                                                        String trainSetPath = "", testSetPath = "";
		
                                                                        // read and parse parameters
                                                                        // read and parse parameters
                                                                        for (String arg : args) {
                                                                                String[] argTokens = arg.split("=");
			
                                                                                if (argTokens[0].compareTo("eta") == 0) 
                                                                                        eta = Double.parseDouble(argTokens[1]);
                                                                                else if (argTokens[0].compareTo("lambdaW") == 0)
                                                                                        lambdaW = Double.parseDouble(argTokens[1]);
                                                                                else if (argTokens[0].compareTo("alpha") == 0)
                                                                                        alpha = Integer.parseInt(argTokens[1]); 
                                                                                else if (argTokens[0].compareTo("maxEpochs") == 0)
                                                                                        maxEpochs = Integer.parseInt(argTokens[1]);
                                                                                else if (argTokens[0].compareTo("R") == 0)
                                                                                        R = Integer.parseInt(argTokens[1]);
                                                                                else if (argTokens[0].compareTo("L") == 0)
                                                                                        L = Double.parseDouble(argTokens[1]);
                                                                                else if (argTokens[0].compareTo("K") == 0)
                                                                                        K = Double.parseDouble(argTokens[1]);
                                                                                else if (argTokens[0].compareTo("trainSet") == 0)
                                                                                        trainSetPath = argTokens[1];
                                                                                else if (argTokens[0].compareTo("testSet") == 0)
                                                                                        testSetPath = argTokens[1];
                                                                        }
		
		
                                                                        // set predefined parameters if none set
                                                                        if(R < 0) R = 4;
                                                                        if(L < 0) L = 0.15;
                                                                        if(eta < 0) eta = 0.01;
                                                                        if(alpha > 0) alpha = -30;
                                                                        if(maxEpochs < 0) maxEpochs = 10000;
		
                                                                        long startTime = System.currentTimeMillis();
		 
		// load dataset
                                                                        DataSet trainSet = new DataSet();
                                                                        trainSet.LoadDataSetFile(new File(trainSetPath));
                                                                        DataSet testSet = new DataSet();
                                                                        testSet.LoadDataSetFile(new File(testSetPath));

                                                                        // normalize the data instance
                                                                        trainSet.NormalizeDatasetInstances();
                                                                        testSet.NormalizeDatasetInstances();
		
		// predictor variables T
                                                                        Matrix T = new Matrix();
                                                                        T.LoadDatasetFeatures(trainSet, false);
                                                                        T.LoadDatasetFeatures(testSet, true);
        // outcome variable O
                                                                        Matrix O = new Matrix();
                                                                        O.LoadDatasetLabels(trainSet, false);
                                                                        O.LoadDatasetLabels(testSet, true);

                                                                        M3_imbalanced_new lsg = new M3_imbalanced_new();   
        // initialize the sizes of data structures
                                                                        lsg.ITrain = trainSet.GetNumInstances();  
                                                                        lsg.ITest = testSet.GetNumInstances();
                                                                        lsg.Q = T.getDimColumns();
        // set the time series and labels
                                                                        lsg.T = T;
                                                                        lsg.Y = O;
        // set the learn rate and the number of iterations
                                                                        lsg.maxIter = maxEpochs;
        // set te number of patterns
                                                                        lsg.K = (int)(K*T.getDimColumns());
                //System.out.println(lsg.K);
                                                                        lsg.L_min = (int)(L*T.getDimColumns());
                                                                        lsg.R = R;
        // set the regularization parameter
                                                                        lsg.theta = learn;
                                                                        lsg.seed = l1min;
                                                                        lsg.del = lamb;
                                                                        lsg.lambdaW = lambdaW;  
                                                                        lsg.eta = eta;  
        
                                                                        lsg.alpha = alpha; 
                                                                        trainSet.ReadNominalTargets();
                                                                        lsg.nominalLabels =  new ArrayList<Double>(trainSet.nominalLabels);
        
        // learn the model
                                                                        lsg.Learn(predictfile,ps2); 
        // learn the local convolutions
        //ps2.println("L_min" + " " + l_min + " " + "lambaW" + " " + lambda +  " "+ "eta" + " " + rate);
                                                                        double [] arrayRet = lsg.GetMCRTestSet(predictfile1,ps3);
                                                                        long endTime = System.currentTimeMillis(); 
        
		//System.out.println( 
		//		"Sensitivity=" + arrayRet[1]  + " " + "Specificity=" + arrayRet[2] + " " + "Weighted Accuracy=" + arrayRet[3] + " " 
		//		+ "L=" + L	+ " " 
		//		+ "R=" + R	+ " " 
		//		+ "lW=" + lambdaW + " "
		//		+ "alpha=" + alpha + " " 
		//		+ "eta=" + eta + " " 
		//		+ "maxEpochs="+ maxEpochs + " " 
		//		+ "time="+ (endTime-startTime) 
		//		); 
                //System.out.println(arrayRet[1] + " " + arrayRet[2] + " " +arrayRet[3] + " " +arrayRet[4] + " " +arrayRet[5] );
                                                                        System.out.println("Sensitivity = " + " " + arrayRet[1] + " " + "Specificity = " + " " + arrayRet[2] + " " + "Accuracy = " + " " + arrayRet[3] + " " + "F1 = " + " " + arrayRet[4] + " " + "F2 = " + " " + arrayRet[5] + " " + "F3 = " + " " + arrayRet[6] + " " + "GM = " + " " + arrayRet[7]  + " " + "AGM = " + " " + arrayRet[8]); 

				//System.out.println(lsg.K);
                //sensi[count] =  arrayRet[1];
                //spec[count] =  arrayRet[2];
                //accu_int[count] = arrayRet[3];
                                                                        F1_int[count] = arrayRet[5];
                //G[count] = arrayRet[7];
                                                                        count = count+1;
                                                                }
                                
                                               
                                                        M3_imbalanced_new lsg1 = new M3_imbalanced_new ();
            
                                                        lsg1.Statistics(F1_int);
                                                        mean_G[counter] = lsg1.getMean(F1_int);
                                                        std_G[counter] = lsg1.getStdDev(F1_int);
            
                                                        param1[counter] = l1min;
                                                        param2[counter] = lamb;
                                                        param3[counter] = learn;
                                                        counter = counter +1;
                 //System.out.println("Errors" + arrayRet[0]);
                 //System.out.println(100*(lsg.ITest-arrayRet[0])/lsg.ITest);
                 //lsg.PrintShapeletsAndWeights(outputfile);
                 //lsg.PrintProjectedData(outputfile1);
                 //lsg.PrintProjectedData();
                
                   
                                                        }
            
                                                }
                                        }           
                                        double max_G =0;
            
                                        int size4 = mean_G.length;
                                        for (int b = 0; b<size4;b++){
                                                if (max_G <=mean_G[b]){
                                                            max_G = mean_G[b];
                                                            par1 = param1[b];
                                                            par2 = param2[b];
                                                            par3 = param3[b];
                                                }
                                        }*/
        
                                        par1 = param1[0];
                                        par2 = param2[0];
                                        par3 = param3[0];
                                        System.out.println("###########Finishing internal crossvalidation######################");
                        
                        
        
                                }
                                System.out.print("Iteration : " + loop1 + " ");
            
                                String sp = File.separator; 
                                // System.out.println(ds + " " +l1 + " "+ l2);
                                args = new String[] {  
				//"trainSet=" + dir + ds + sp + "folds" + sp + "default" + sp  
						//+ ds + "_TRAIN",  
                                //"trainSet=" + dir + ds + sp + ds + "_TRAIN.txt",  
                                //"testSet=" + dir + ds + sp + ds + "_TEST.txt",
                                "trainSet=" + dir + folder + ds + "_TRAIN",  
                                "testSet=" + dir + folder +  ds + "_TEST",
				//"testSet=" + dir + ds + sp + "folds" + sp + "default" + sp  
				//		+ ds + "_TEST",  
 				"alpha=-100",
				"eta=0.01",
				"maxEpochs=5000",
				"K=-1",		
				"L=0.1", 
				"R=3", 
				"lambdaW = 0.01"
				};
					
                        //}
		// values of hyperparameters
                                double eta = -1, lambdaW = -1, alpha = -1, L = -1, K = -1;
                                int maxEpochs = -1, R = -1;
                                String trainSetPath = "", testSetPath = "";
		
		// read and parse parameters
		// read and parse parameters
                                for (String arg : args) {
                                        String[] argTokens = arg.split("=");
			
                                        if (argTokens[0].compareTo("eta") == 0) 
                                                eta = Double.parseDouble(argTokens[1]);
                                        else if (argTokens[0].compareTo("lambdaW") == 0)
                                                lambdaW = Double.parseDouble(argTokens[1]);
                                        else if (argTokens[0].compareTo("alpha") == 0)
                                                alpha = Integer.parseInt(argTokens[1]); 
                                        else if (argTokens[0].compareTo("maxEpochs") == 0)
                                                maxEpochs = Integer.parseInt(argTokens[1]);
                                        else if (argTokens[0].compareTo("R") == 0)
                                                R = Integer.parseInt(argTokens[1]);
                                        else if (argTokens[0].compareTo("L") == 0)
                                                L = Double.parseDouble(argTokens[1]);
                                        else if (argTokens[0].compareTo("K") == 0)
                                                K = Double.parseDouble(argTokens[1]);
                                        else if (argTokens[0].compareTo("trainSet") == 0)
                                                trainSetPath = argTokens[1];
                                        else if (argTokens[0].compareTo("testSet") == 0)
                                                testSetPath = argTokens[1];
                                }
		
		
		// set predefined parameters if none set
                                if(R < 0) R = 4;
                                if(L < 0) L = 0.15;
                                if(eta < 0) eta = 0.01;
                                if(alpha > 0) alpha = -30;
                                if(maxEpochs < 0) maxEpochs = 10000;
		
                                long startTime = System.currentTimeMillis();
		 
		// load dataset
                                DataSet trainSet = new DataSet();
                                trainSet.LoadDataSetFile(new File(trainSetPath));
                                DataSet testSet = new DataSet();
                                testSet.LoadDataSetFile(new File(testSetPath));

		// normalize the data instance
                                trainSet.NormalizeDatasetInstances();
                                testSet.NormalizeDatasetInstances();
		
		// predictor variables T
                                Matrix T = new Matrix();
                                T.LoadDatasetFeatures(trainSet, false);
                                T.LoadDatasetFeatures(testSet, true);
        // outcome variable O
                                Matrix O = new Matrix();
                                O.LoadDatasetLabels(trainSet, false);
                                O.LoadDatasetLabels(testSet, true);

                                M3_balanced_newZ lsg3 = new M3_balanced_newZ();   
        // initialize the sizes of data structures
                                lsg3.ITrain = trainSet.GetNumInstances();  
                                lsg3.ITest = testSet.GetNumInstances();
                                //System.out.println("# Train examples");
                                lsg3.Q = T.getDimColumns();
        // set the time series and labels
                                lsg3.T = T;
                                lsg3.Y = O;
        // set the learn rate and the number of iterations
                                lsg3.maxIter = maxEpochs;
        // set te number of patterns
                                lsg3.K = (int)(K*T.getDimColumns());
                //System.out.println(lsg.K);
                                lsg3.L_min = (int)(L*T.getDimColumns());
                                lsg3.R = R;
        // set the regularization parameter
                                lsg3.theta = par3;
                                lsg3.seed = par1;
                                lsg3.del =  par2;
                                lsg3.lambdaW = 0.01;  
                                lsg3.eta = eta;  
        
                                lsg3.alpha = alpha; 
                                trainSet.ReadNominalTargets();
                                lsg3.nominalLabels =  new ArrayList<Double>(trainSet.nominalLabels);
        
        // learn the model
                                lsg3.Learn(predictfile, ps2); 
                                double [] arrayRet1 = lsg3.GetMCRTestSet(predictfile,ps2);
                                lsg3.PrintProjectedData(outputfile1);
                                lsg3.PrintShapeletsAndWeights(outputfile);
                                lsg3.PrintCost(outputfile4);
                                long endTime = System.currentTimeMillis(); 
        
		//System.out.println( 
		//		"Sensitivity=" + arrayRet[1]  + " " + "Specificity=" + arrayRet[2] + " " + "Weighted Accuracy=" + arrayRet[3] + " " 
		//		+ "L=" + L	+ " " 
		//		+ "R=" + R	+ " " 
		//		+ "lW=" + lambdaW + " "
		//		+ "alpha=" + alpha + " " 
		//		+ "eta=" + eta + " " 
		//		+ "maxEpochs="+ maxEpochs + " " 
		//		+ "time="+ (endTime-startTime) 
		//		); 
        //System.out.println(arrayRet1[1] + " " + arrayRet1[2] + " " +arrayRet1[3] + " " +arrayRet1[4] + " " +arrayRet1[5] );
                                System.out.println("Sensitivity = " + " " + arrayRet1[1] + " " + "Specificity = " + " " + arrayRet1[2] + " " + "Accuracy = " + " " + arrayRet1[3] + " " + "F1 = " + " " + arrayRet1[4] + " " + "F2 = " + " " + arrayRet1[5] + " " + "F3 = " + " " + arrayRet1[6] + " " + "GM = " + " " + arrayRet1[7]  + " " + "AGM = " + " " + arrayRet1[8]); 
    
                                sensi[cou] =  arrayRet1[1];
                                spec[cou] =  arrayRet1[2];
                                accu[cou] = arrayRet1[3];
                                F1[cou] = arrayRet1[4];
                                F2[cou] = arrayRet1[5];
                                F3[cou] = arrayRet1[6];
                                GG[cou] = arrayRet1[7];
                                AGM[cou] = arrayRet1[8];
                                cou = cou+1;               
                                //ps2.close();
                        }
         
             
              //}}
                        M3_balanced_newZ lsg4 = new M3_balanced_newZ ();
                        lsg4.Statistics(sensi);
                        double mean_sensi = lsg4.getMean(sensi);
                        double std_sensi = lsg4.getStdDev(sensi);
            
                        lsg4.Statistics(spec);
                        double spec_mean = lsg4.getMean(spec);
                        double spec_std = lsg4.getStdDev(spec);
            
                        lsg4.Statistics(accu);
                        double accu_mean = lsg4.getMean(accu);
                        double accu_std = lsg4.getStdDev(accu);
            
                        lsg4.Statistics(F1);
                        double f1_mean = lsg4.getMean(F1);
                        double f1_std = lsg4.getStdDev(F1);
			
                        lsg4.Statistics(F2);
                        double mean_F2 = lsg4.getMean(F2);
                        double std_F2 = lsg4.getStdDev(F2);
            
                        lsg4.Statistics(F3);
                        double mean_F3 = lsg4.getMean(F3);
                        double std_F3 = lsg4.getStdDev(F3);
            
                        lsg4.Statistics(GG);
                        double g_mean = lsg4.getMean(GG);
                        double g_std = lsg4.getStdDev(GG);
            
                        lsg4.Statistics(AGM);
                        double mean_AGM = lsg4.getMean(AGM);
                        double std_AGM = lsg4.getStdDev(AGM);
            
            //ps1.print(mean_sensi*100 + "\t"+ spec_mean*100 +"\t" + accu_mean*100 );
                        ps1.println("SEED" + " " + par1 +" " +  "delta" + " " + par2 + " " + "theta" + " " + par3);
                        ps1.println("Sensitivity:" + " " + mean_sensi + "\u00B1"+ std_sensi );
                        ps1.println("Specificity:" + " " + spec_mean + "\u00B1" + spec_std);
                        ps1.println("Accuracy:" + " " + accu_mean + "\u00B1" + accu_std);
                        ps1.println("F1:" + " " + f1_mean + "\u00B1" + f1_std);
                        ps1.println("F2:" + " " + mean_F2 + "\u00B1" + std_F2);
                        ps1.println("F3:" + " " + mean_F3 + "\u00B1" + std_F3);
                        ps1.println("GM:" + " " + g_mean + "\u00B1" + g_std);
                        ps1.println("AGM:" + " " + mean_AGM + "\u00B1" + std_AGM);
                        ps1.println();
                        ps1.close();  
                }
        }
    
}
