/*
 * To change this template, choose Tools | Templates
 * and open the template in the editor.
 */
package Utilities;

import java.io.File;

/**
 *
 * @author Josif Grabocka
 */
public class FileOperations 
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
}
