package Utilities;

import java.io.BufferedInputStream;
import java.io.BufferedReader;
import java.io.File;
import java.io.FileInputStream;
import java.io.FileReader;
import java.io.IOException;
import java.io.InputStream;

import org.apache.commons.math3.transform.DctNormalization;
import org.apache.commons.math3.transform.FastCosineTransformer;
import org.apache.commons.math3.transform.TransformType;

public class ReadMultivariateSeriesDataset
{
	public int numInstances, numChannels, numPoints;
	
	public double X [][][];
	public double Y[];

	public void LoadDatasetTimeSeries(String tsFile)
	{
		try
		{
			BufferedReader in = new BufferedReader(new FileReader(new File(tsFile)));
		
			// first read the line with the dimensions
			String line = in.readLine();
			String[] tokens = line.split(" ");
		     
			numInstances = Integer.parseInt( tokens[0] );
	    	numChannels = Integer.parseInt( tokens[1] );
	    	numPoints = Integer.parseInt( tokens[2] );
		    
	    	Logging.println("Reading time-series data with dimensions ["+numInstances+", "+numChannels+", "+ numPoints +"]" );
	    	
	    	X = new double[numInstances][numChannels][numPoints];
	    	
	    	for(int i=0; i<numInstances; i++)
	    	{
	    		for(int c=0; c<numChannels; c++)
	    		{
	    			line = in.readLine();
	    			tokens = line.split(" ");
	    			
	    			if(tokens.length != numPoints)
	    				Logging.println("Series: (i="+i+", c="+c + ") : Real length=" 
    							+ tokens.length +" and doesn't match the parameter length=" + numPoints + ".");
	    			
	    			for(int p=0; p<numPoints; p++)
	    			{
	    				 X[i][c][p] = Double.parseDouble( tokens[p] );
	    			}
	    		} 
	    	} 
	    	
		}
		catch( Exception exc )
		{
			exc.printStackTrace();
		}
	}
	
	
	// normalize the time-series data
	public void NormalizeData()
	{
		Logging.println("Normalizing data.");
		
		for(int i=0; i<numInstances; i++)
    	{
    		for(int c=0; c<numChannels; c++)
    		{
    			X[i][c] = StatisticalUtilities.Normalize(X[i][c]);
    		}
    	}
	}
	
	
	public void LoadDatasetLabels(String labelsFile)
	{
		try
		{
			BufferedReader in = new BufferedReader(new FileReader(new File(labelsFile)));
			
			int lineNumber = -1;
			
			for (String line = in.readLine(); line != null ; line = in.readLine()) {
			    
			     String[] tokens = line.split(" ");
			     
			     if(lineNumber == -1)
			     {
			    	 numInstances = Integer.parseInt( tokens[0] );
			    	 
			    	 Y = new double[numInstances];
			
			    	 Logging.println("Reading time-series labels with dimensions ["+numInstances+"]" );
			    	 // Logging.println("Initialized time-series labels with dimensions [" + numInstances + "]"); 
			     }
			     else
			     {
			         Y[lineNumber] = Double.parseDouble( tokens[0] );
			     }
			    
			     lineNumber++;
			}
			
		}
		catch( Exception exc )
		{
			exc.printStackTrace();
		}
	}
	
	
	
}
