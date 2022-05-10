#include "utils.h"
using namespace std;  

void top(float AC_DR[n+1][n+1],float AP_DR[n+1][n+1],float EC_DR[n+1][n+1],float EP_DR[n+1][n+1],
float S_DR[n+1][n+1]);


int main() 
{  

    //Initilize Price array and 4 options arrays
    float S[n+1][n+1];
    float EC[n+1][n+1];
    float EP[n+1][n+1];
    float AC[n+1][n+1];
    float AP[n+1][n+1];

    top(AC,AP,EC,EP,S);      

// Output of prices of calls and puts  

    cout << "The Cox Ross Rubinstein prices using " << n << " steps are... " << endl;  

    cout << "European Call " << EC[0][0] << endl;  

    cout << "European Put  " << EP[0][0] << endl;  

    cout << "American Call " << AC[0][0] << endl;  

    cout << "American Put  " << AP[0][0] << endl;  

    cout << endl;  
    return 0;

}  