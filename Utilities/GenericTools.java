/*
 * To change this license header, choose License Headers in Project Properties.
 * To change this template file, choose Tools | Templates
 * and open the template in the editor.
 */
package utilities;

import java.text.DecimalFormat;
import java.util.ArrayList;
import java.util.Formatter;
import java.util.Locale;
import java.util.Random;

/**
 *
 * @author raj09hxu
 */
public class GenericTools {
    
    public static final DecimalFormat RESULTS_DECIMAL_FORMAT = new DecimalFormat("#.######");
    
    public static double indexOfMax(double[] dist) {
        double max = dist[0];
        int maxInd = 0;
        
        for (int i = 1; i < dist.length; ++i) {
            if (dist[i] > max) {
                max = dist[i];
                maxInd = i;
            }
        }
        return maxInd;
    }
    
    public static double indexOf(double[] array, double val){
        for(int i=0; i<array.length; i++)
            if(array[i] == val)
               return i;
        
        return -1;
    }
    
    public static <E> ArrayList<E> cloneArrayList(ArrayList<E> list){
        ArrayList<E> temp = new ArrayList<>();
        for (E el : list) {
            temp.add(el);
        }
        return temp;              
    }

    public static <E> ArrayList<E> twoDArrayToList(E[][] twoDArray) {
        ArrayList<E> list = new ArrayList<>();
        for (E[] array : twoDArray){
            if(array == null) continue;
            
            for(E elm : array){
                if(elm == null) continue;
                
                list.add(elm);
            }
        }
        return list;
    }
    
    //this is inclusive of the top value.
    public static int randomRange(Random rand, int min, int max){
        return rand.nextInt((max - min) + 1) + min;
    }
    
    
    public static String sprintf(String format, Object... strings){
        StringBuilder sb = new StringBuilder();
        String out;
        try (Formatter ft = new Formatter(sb, Locale.UK)) {
            ft.format(format, strings);
            out = ft.toString();
        }
        return out;
    }
    
}
