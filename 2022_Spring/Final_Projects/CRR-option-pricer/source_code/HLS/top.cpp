#include "utils.h"

float max(float a, float b)  
{
    return (a>b) ? a : b;
}

//build price tree using S array
void buildtree(float mat[n+1][n+1])
{

    for (int i = 0; i <= n; i++) 
    {
        #pragma HLS pipeline
        for (int j = 0; j <= n; j++) 
        { 
            #pragma HLS unroll  
            mat[j][i] = Spot*powf(u, i - j)*powf(d, j);             
        }  
    }   
}


//Compute option value at final nodes for all 4 option arrays given price tree
void terminalPayoff(float matAC[n+1][n+1],float matAP[n+1][n+1],float matEC[n+1][n+1],
float matEP[n+1][n+1], float matS[n+1][n+1])
{
    for (int i = 0; i <= n; i++) 
    {
         #pragma HLS dataflow   //#pragma HLS unroll this unroll made no change on utlization or latency ie latency stayed at 98.60us 
        matEC[i][n] = max(matS[i][n] - K, 0.0);  
        matAC[i][n] = max(matS[i][n] - K, 0.0);  
        matEP[i][n] = max(K - matS[i][n], 0.0);  
        matAP[i][n] = max(K - matS[i][n], 0.0);  
    }  
}


// void subfunction(float matAC[n+1][n+1],float matAP[n+1][n+1],
// float matEC[n+1][n+1],float matEP[n+1][n+1], float matS[n+1][n+1],int k)
// {     
//     for(int i = 0;i<=n-1;i++)
//     {
 
//         matEC[i][k] =expf(-r*dt)*(p*matEC[i][k + 1] + (1 - p)*matEC[i + 1][k + 1]);  
//         matEP[i][k] =expf(-r*dt)*(p*matEP[i][k + 1] + (1 - p)*matEP[i + 1][k + 1]);  
//         matAC[i][k] = max(matS[i][k] - K,expf(-r*dt)*(p*matAC[i][k + 1] + (1 - p)*matAC[i + 1][k + 1]));  
//         matAP[i][k] = max(K - matS[i][k],expf(-r*dt)*(p*matAP[i][k + 1] + (1 - p)*matAP[i + 1][k + 1]));  
//     }
// }


// void backwardRecursion(float matAC[n+1][n+1],float matAP[n+1][n+1],float matEC[n+1][n+1],
// float matEP[n+1][n+1], float matS[n+1][n+1])
// {
//     for (int j = n - 1; j >= 0; j--) 
//     {  
//          #pragma HLS datflow                             
//         subfunction(matAC,matAP,matEC,matEP,matS,j);  
//     } 
// }

//pipeline and unroll cause over utilization
void backwardRecursion(float matAC[n+1][n+1],float matAP[n+1][n+1],
float matEC[n+1][n+1],
float matEP[n+1][n+1], float matS[n+1][n+1])
{
    //#pragma HLS loop_merge force
    for (int j = n - 1; j >= 0; j--) 
    {
       //#pragma HLS DATAFLOW //#pragma HLS pipeline both of these made latency worse  
      for (int i = 0; i <= n-1; i++) 
        {
            #pragma HLS loop_flatten //loop_flatten worked fine here        //#pragma HLS unroll this also failed
            matEC[i][j] =expf(-r*dt)*(p*matEC[i][j + 1] + (1 - p)*matEC[i + 1][j + 1]);  

            matEP[i][j] =expf(-r*dt)*(p*matEP[i][j + 1] + (1 - p)*matEP[i + 1][j + 1]);  

            matAC[i][j] = max(matS[i][j] - K,expf(-r*dt)*(p*matAC[i][j + 1] + (1 - p)*matAC[i + 1][j + 1]));  

            matAP[i][j] = max(K - matS[i][j],expf(-r*dt)*(p*matAP[i][j + 1] + (1 - p)*matAP[i + 1][j + 1]));  

        }

    } 
}

void moveTobram(float matB[n+1][n+1],float matD[n+1][n+1])
{
    for (int i = 0; i < (n+1);i++)
    {
        //#pragma HLS LOOP_FLATTEN off
        for(int j = 0; j <(n+1);j++)
        {
            matB[i][j] = matD[i][j];
        }
    }
}

void top(float AC_DR[n+1][n+1],float AP_DR[n+1][n+1],float EC_DR[n+1][n+1],float EP_DR[n+1][n+1],
float S_DR[n+1][n+1])
{
    #pragma HLS interface m_axi depth=51*51 port=AC_DR offset=slave bundle=memAC
    #pragma HLS interface m_axi depth=51*51 port=AP_DR offset=slave bundle=memAP
    #pragma HLS interface m_axi depth=51*51 port=EC_DR offset=slave bundle=memEC
    #pragma HLS interface m_axi depth=51*51 port=EP_DR offset=slave bundle=memEP
    #pragma HLS interface m_axi depth=51*51 port=S_DR offset=slave bundle=memS
    #pragma HLS interface s_axilite port=return

    float AC_B[n+1][n+1]; float AP_B[n+1][n+1];float EC_B[n+1][n+1];float EP_B[n+1][n+1];float S_B[n+1][n+1];

    #pragma HLS ARRAY_PARTITION variable=S_B dim = 1 factor = 1 cyclic
    #pragma HLS ARRAY_PARTITION variable=AC_B dim = 1 factor = 1 cyclic
    #pragma HLS ARRAY_PARTITION variable=AP_B dim = 1 factor = 1 cyclic
    #pragma HLS ARRAY_PARTITION variable=EC_B dim = 1 factor = 1 cyclic
    #pragma HLS ARRAY_PARTITION variable=EP_B dim = 1 factor = 1 cyclic

    moveTobram(AC_B,AC_DR);
    moveTobram(AP_B,AP_DR);
    moveTobram(EC_B,EC_DR);
    moveTobram(EP_B,EP_DR);
    moveTobram(S_B,S_DR);

    buildtree(S_B);
    terminalPayoff(AC_B,AP_B,EC_B,EP_B,S_B);
    backwardRecursion(AC_B,AP_B,EC_B,EP_B,S_B);

    //write back to DRAM
    for (int i = 0; i<(n+1);i++)
    {

        for (int j = 0; j <(n+1);j++)
        {
            AC_DR[i][j] = AC_B[i][j];
            AP_DR[i][j] = AP_B[i][j];
            EC_DR[i][j] = EC_B[i][j];
            EP_DR[i][j] = EP_B[i][j];
            S_DR[i][j]  = S_B[i][j]; 
        }
    }
}
