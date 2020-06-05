/*
 * To change this template, choose Tools | Templates
 * and open the template in the editor.
 */
package Utilities;

import java.io.BufferedInputStream;
import java.io.BufferedReader;
import java.io.File;
import java.io.FileInputStream;
import java.io.FileReader;
import java.io.InputStreamReader;
import java.io.LineNumberReader;
import java.util.ArrayList;
import java.util.List;
import java.util.StringTokenizer;

import umontreal.iro.lecuyer.util.Num;

import Utilities.Logging.LogLevel;

import DataStructures.Tripple;

/**
 *
 * @author Josif Grabocka
 */
public class IOUtils 
{
    public static void Remove(File file)
    {
        if( file.isDirectory() )
        {
            File[] subFiles = file.listFiles();
            for( File subfile : subFiles)
            {
                if( subfile.isDirectory() )
                {
                    Remove(subfile);
                }
                else
                {
                    subfile.delete();
                }
            }
            
            file.delete();
        }
        else
        {
            file.delete();
        }
    }
    
    public static List<List<Double>> LoadFile(String filePath)
    {
    	return LoadFile(filePath, false);
    }
    
    public static List<List<Double>> LoadFile(String filePath, boolean headerRow)
    {
		List<List<Double>> data = new ArrayList<List<Double>>(); 
  	    
		try
        {
			// a line number reader created from the file path
			LineNumberReader lr = new LineNumberReader(
					new InputStreamReader(
							new BufferedInputStream(
									new FileInputStream(filePath))));
	        
			String delimiters = "\t ,;";
	        
	        String line = null;
	         
	        if(headerRow)
	        	lr.readLine();
	        
	        int numMissingValues = 0;
	        int numTotalValues = 0;
	        
	        while ( (line = lr.readLine()) != null )
	        {
	        	List<Double> instanceVariables = new ArrayList<Double>(); 
	        	
	        	try
	        	{
		        	// read tokens and write into the 
		        	StringTokenizer tokenizer = new StringTokenizer(line, delimiters);
		        	while( tokenizer.hasMoreTokens() )
		        	{
		        		String token = tokenizer.nextToken();
		        		
		        		if( token.compareTo("?") == 0 || token.compareTo("-") == 0 )
		        		{
		        			instanceVariables.add( GlobalValues.MISSING_VALUE );
		        			numMissingValues++;
		        		}
		        		else
		        			instanceVariables.add( Double.parseDouble( token ) );
		        		
		        		numTotalValues++;
		        	}
		        	
		        	data.add( instanceVariables );
	        	}
	        	catch(Exception exc)
	        	{
	        		Logging.println("Error Loading CSV: " + exc.getMessage(), LogLevel.ERROR_LOG);
	        	}
	        	
	        }
	        
	        Logging.println("RegressionDataSet::LoadFile:: Percentage of missing values=" + 
	        		(double) numMissingValues / (double) numTotalValues, LogLevel.ERROR_LOG);
			
	    }
        catch( Exception exc )
        {
        	Logging.println("RegressionDataSet::LoadFile::" + exc.getMessage(), LogLevel.ERROR_LOG);
        }	
				
	    return data;
        
	}
    
    
    
    
    
}
