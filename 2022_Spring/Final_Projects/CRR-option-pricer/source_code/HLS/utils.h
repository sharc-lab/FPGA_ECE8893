#ifndef __UTILS_H__
#define __UTILS_H__

//#include <vector>  
#include <stdio.h>  
#include <math.h>  
#include <iostream>  
#include <string>



#define n 50  //number of steps

#define Spot 100              // Spot Price  

#define K 100                 // Strike Price  

#define T 1                   // Years to maturity  

#define q 0.05                //dividend yield

#define r 0.05                    // Risk Free Rate  

#define v 0.20                    // Volatility  

#define dt (T) /(n)  //DetlaT 0.01

#define u 1.0286880693018583 //Up factor  //1.0286880693018583 exp((v) * sqrt(dt))  //1.0202013400267558 

#define d 1/u  //down factor      //exp((-v) * sqrt((dt)))  //0.9801986733067554

#define p 0.49292940355494647  //probability     //0.4752629255391314exp(((r)-(q))*(dt)) - (d)/ ((u) - (d)) //0.48250412422538935

#endif