// C++ program to multiply two matrices

#include <bits/stdc++.h>
using namespace std;

// Edit MACROs here, according to your Matrix Dimensions for
// mat1[R1][C1] and mat2[R2][C2]

#define N 1024
// #define R1 2 // number of rows in Matrix-1
// #define C1 2 // number of columns in Matrix-1
// #define R2 2 // number of rows in Matrix-2
// #define C2 2 // number of columns in Matrix-2

int mat1[N][N], mat2[N][N];

void mulMat()
{
    int rslt[N][N];

    // cout << "Multiplication of given two matrices is:\n";

    for (int i = 0; i < N; i++) {
        for (int j = 0; j < N; j++) {
            rslt[i][j] = 0;

            for (int k = 0; k < N; k++) {
                rslt[i][j] += mat1[i][k] * mat2[k][j];
            }

            // cout << rslt[i][j] << "\t";
        }

        // cout << endl;
    }
}

// Driver code
int main()
{
    
    // R1 = 4, C1 = 4 and R2 = 4, C2 = 4 (Update these
    // values in MACROs)
    // int mat1[R1][C1] = { { 1, 1 },
    //                      { 2, 2 } };

    // int mat2[R2][C2] = { { 1, 1 },
    //                      { 2, 2 } };

    for(int i=0; i<N; i++){
        for(int j=0; j<N; j++){
            mat1[i][j] = i+j;
            if(i==j)mat2[i][j]=1;
        }
    }

    // if (C1 != R2) {
    //     cout << "The number of columns in Matrix-1  must "
    //             "be equal to the number of rows in "
    //             "Matrix-2"
    //          << endl;
    //     cout << "Please update MACROs according to your "
    //             "array dimension in #define section"
    //          << endl;

    //     exit(EXIT_FAILURE);
    // }

      // Function call
    mulMat();

    return 0;
}

// This code is contributed by Manish Kumar (mkumar2789)
