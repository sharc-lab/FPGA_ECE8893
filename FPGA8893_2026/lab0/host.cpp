#include <stdio.h>
#include <stdlib.h>

#include "dcl.h"

int main()
{
    DATA_TYPE a[100];
    DATA_TYPE b[100];
    DATA_TYPE c_HLS[100];     // computed by top_kernel
    DATA_TYPE c_golden[100];  // golden reference

    // Initialize inputs and golden output
    for (int i = 0; i < 100; i++) {
        a[i] = i;
        b[i] = i + 1;
        c_HLS[i] = 0;
        c_golden[i] = a[i] * b[i];
    }

    // Call HLS kernel
    top_kernel(a, b, c_HLS);

    // Check results
    int error = 0;
    for (int i = 0; i < 100; i++) {
        if (c_HLS[i] != c_golden[i]) {
            printf("Mismatch at index %d: HLS = %d, Ref = %d\n",
                   i, c_HLS[i], c_golden[i]);
            error = 1;
            break;
        }
    }

    if (error) {
        printf("TEST FAILED\n");
        return 1;
    } else {
        printf("TEST PASSED\n");
        return 0;
    }
}
