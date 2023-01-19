// top.c
void top(int a[100], int b[100], int sum[100]) 
{
	#pragma HLS interface m_axi port=a depth=100 offset=slave bundle = A
	#pragma HLS interface m_axi port=b depth=100 offset=slave bundle = B
	#pragma HLS interface m_axi port=sum depth=100 offset=slave bundle = SUM
	
  #pragma HLS interface s_axilite register port=return
	
  for (int i = 0; i < 100; i++) 
  {
		sum[i] = a[i] + b[i];
	}
}
