//
//  main.cpp
//  Intel_test
//
//  Created by Xu on 2021/12/27.

#include <bits/stdc++.h>
#include <iostream>
#include <vector>
#include <chrono>
using namespace std;
using namespace std::chrono;

// Helper method to find the max elements of an 4D tensor for the first position of a row
int first_max(vector<vector<vector<vector<int>>>> v, int n, int c, int row_l, int row_h, int col_l, int col_h){
    int max = v[n][c][row_l][col_l];
    for (int i = row_l; i < row_h; i++){
        for (int j = col_l; j < col_h; j++){
            if (v[n][c][i][j] > max){
                max = v[n][c][i][j];
            }
        }
    }
    return max;
}

// Helper method to find the max elements of an 4D tensor
int find_max(vector<vector<vector<vector<int>>>> v, int n, int c, int row_l, int row_h, int col_l, int col_h, int pre_max){
    int max = v[n][c][row_l][col_l];
    for (int i = row_l; i < row_h; i++){
        for (int j = col_l; j < col_h; j++){
            if (v[n][c][i][j] > max){
                max = v[n][c][i][j];
            }
        }
    }
    // No need to make further comparisons if we get a value bigger than previous group's max
    if (max >= pre_max){
        return max;
    }
    for (int i = row_l; i < row_h; i++){
        if (max < v[n][c][i][col_l - 1]){
            max = v[n][c][i][col_l - 1];
        }
    }
    return max;
}

// #
// # The writing test method:
// # dst[32,64,56,56] = add(max_pooling(src1[32,64,112,112]), src2[32,1,56,56])
// #
// # max_pooling op is fixed to kernel [3,3], pad [1,1], stride [2, 2], src1 and src2 are the two input tensors,
// # dst is the output tensor.
// #
// # Since this is a function built for a specific formula, some important parameters are preset,
// # and further modifications will be necessary for general formula computation.
// #
void intel_test(vector<vector<vector<vector<int>>>> src1, vector<vector<vector<vector<int>>>> src2, vector<vector<vector<vector<int>>>> res){
    // # Since padding is non-zero, the src1 is implicitly padded with negative infinity on both sides
    // # for padding number of points.
    // # Since the stride is 2, the stride of the window is 2.
    for (int i = 0; i < 32; i++){
        for (int j = 0; j < 64; j++){
            // The first window of the first row
            res[i][j][0][0] = first_max(src1, i, j, 0, 2, 0, 2) + src2[i][0][0][0];
            // Following windows of the first row
            for (int fir_col = 1; fir_col < 56; fir_col++){
                res[i][j][0][fir_col] = find_max(src1, i, j, 0, 2, fir_col * 2, (fir_col + 1) * 2, res[i][j][0][fir_col - 1]) + src2[i][0][0][fir_col];
            }
            // Go through each row
            for (int row_in = 1; row_in < 56; row_in++){
                // The first window of each row
                res[i][j][row_in][0] = first_max(src1, i, j, row_in * 2 - 1, (row_in + 1) * 2, 0, 2) + src2[i][0][row_in][0];
                // Go through each col
                for (int col_in = 1; col_in < 56; col_in++){
                    res[i][j][row_in][col_in] = find_max(src1, i, j, row_in * 2 - 1, (row_in + 1) * 2, col_in * 2, (col_in + 1) * 2, res[i][j][row_in][col_in - 1]) + src2[i][0][row_in][col_in];
                }
            }
        }
    }
}

int main()
{
    // Initialize input vectors
    vector<int> v1_rand(112);
    generate(v1_rand.begin(), v1_rand.end(), rand);
    
    vector<int> v2_rand(56);
    generate(v2_rand.begin(), v2_rand.end(), rand);
    
    std::vector<vector<vector<vector<int>>>> vec_1(32, vector<vector<vector<int>>>(64, vector<vector<int>>(112, v1_rand)));
    std::vector<vector<vector<vector<int>>>> vec_2(32, vector<vector<vector<int>>>(1, vector<vector<int>>(56, v2_rand)));
    
    std::vector<vector<vector<vector<int>>>> res(32, vector<vector<vector<int>>>(64, vector<vector<int>>(56, vector<int>(56))));
    
    auto start = high_resolution_clock::now();
    
    intel_test(vec_1, vec_2, res);
    
    auto stop = high_resolution_clock::now();
    auto duration = duration_cast<microseconds>(stop - start);
    
    std::cout << duration.count() << " microseconds" << endl;

    return 0;
}
