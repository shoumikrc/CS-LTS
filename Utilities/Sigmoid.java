/*
 * To change this template, choose Tools | Templates
 * and open the template in the editor.
 */
package Utilities;

/**
 * Sigmoid function
 * @author Josif Grabocka
 */
public class Sigmoid 
{
    public static double Calculate(double x)
    {
        return 1 / (1 + Math.exp(-x));
    }
    
    public static double Calculate(double x, double sigmoidicity)
    {
        return 1 / (1 + Math.exp(-sigmoidicity*x)); 
    }
}
