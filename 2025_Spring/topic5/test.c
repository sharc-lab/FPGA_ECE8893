
//Command line : python pseudo.py 16 2**255-19

#include <stdio.h>
#include <stdint.h>



#ifdef BIT16
#include <16.h>
#elif BIT32
#include <32.h>
#else
#include <64.h>
#endif




void test_modmul() {
	spint x[20],y[20],z[20];
	int i,j;

	x[0]=0x199d; x[1]=0x1d8b; x[2]=0x1fe8; x[3]=0x62; x[4]=0x11c8; x[5]=0xd68; x[6]=0x199c; x[7]=0x1f60; x[8]=0x1640; x[9]=0x1dee; x[10]=0x1495; x[11]=0xcd0; x[12]=0x1904; x[13]=0x189d; x[14]=0x18fa; x[15]=0xadb; x[16]=0x1924; x[17]=0x7a1; x[18]=0x1718; x[19]=0x23; 
	y[0]=0xa9; y[1]=0x44b; y[2]=0x9ef; y[3]=0x1a3f; y[4]=0x11a3; y[5]=0x16b6; y[6]=0x14f0; y[7]=0x1675; y[8]=0x1c66; y[9]=0x5ec; y[10]=0x1854; y[11]=0x8cb; y[12]=0x34e; y[13]=0xe92; y[14]=0xa2e; y[15]=0x13f0; y[16]=0x1641; y[17]=0x11a0; y[18]=0x550; y[19]=0x97; 

	// i and j contains the number of iterations. 
	for (i=0;i<100000;i++)
		for (j=0;j<200;j++) {
			modmul(x,y,z);
			modmul(z,x,y);
			modmul(y,z,x);
			modmul(x,y,z);
			modmul(z,x,y);
		}

	printf("modmul check 0x%06x\n",(int)z[0]&0xFFFFFF);
}



int main() {

#ifdef BIT16
	printf("16-bit multiplication\n");
#elif BIT32
	printf("32-bit multiplication\n");
#else
	printf("64-bit multiplication\n");
#endif


	test_modmul();
	return 0;
}

