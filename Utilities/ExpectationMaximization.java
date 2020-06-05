/*
 * To change this template, choose Tools | Templates
 * and open the template in the editor.
 */
package Utilities;

import DataStructures.DataSet;
import DataStructures.FeaturePoint.PointStatus;
import weka.core.Instances;

import weka.filters.Filter;
import weka.filters.unsupervised.attribute.EMImputation;

/**
 *
 * @author Josif Grabocka
 */
public class ExpectationMaximization 
{
    public void ImputeMissing(DataSet ds)
    {
        Instances data = ds.ToWekaInstances();
        
        String[] options = new String[]{ "-N", "-1", "-E_i", "1.0E-4", "-Q", "1.0E-8" };
        Instances imputedData  = null;
        
        try
        {
            EMImputation emImputationFilter = new EMImputation();             // new instance of filter
            emImputationFilter.setOptions(options);                           // set options
            emImputationFilter.setInputFormat(data);                          // inform filter about dataset **AFTER** setting options

            imputedData = Filter.useFilter(data, emImputationFilter);  
        
        }
        catch(Exception exc)
        {
            Logging.println("ERROR: " + exc.getMessage(), Logging.LogLevel.PRODUCTION_LOG);
        }
        
        
        for(int i = 0; i < ds.instances.size(); i++)
        {
            for(int j = 0; j < ds.numFeatures; j++)
            {
                if( ds.instances.get(i).features.get(j).status == PointStatus.MISSING )
                {
                    ds.instances.get(i).features.get(j).value = imputedData.get(i).value(j); 
                    ds.instances.get(i).features.get(j).status = PointStatus.PRESENT;
                }
            }
        }
           
    }
    
}
