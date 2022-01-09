#
#   Intel SDE test
#   Created by Xu Hai on 2021/12/23.
#

import torch
import torch.nn as nn
import numpy as np
import time
import unittest


#
# The writing test method:
# dst[32,64,56,56] = add(max_pooling(src1[32,64,112,112]), src2[32,1,56,56])
#
# max_pooling op is fixed to kernel [3,3], pad [1,1], stride [2, 2], src1 and src2 are the two input tensors,
# dst is the output tensor.
#
# Since this is a function built for a specific formula, some important parameters are preset,
# and further modifications will be necessary for general formula computation.
#
def intel_formula(in_tensor_1, in_tensor_2):
    in_shape_1 = in_tensor_1.shape
    in_shape_2 = in_tensor_2.shape
    n_1 = in_shape_1[0]
    c_1 = in_shape_1[1]
    s_1 = in_shape_1[2]

    n_2 = in_shape_2[0]
    c_2 = in_shape_2[1]
    s_2 = in_shape_2[2]

    if n_1 != 32 or c_1 != 64 or s_1 != 112 or n_2 != 32 or c_2 != 1 or s_2 != 56:
        print("Invalid input, input tensors should satisfy src1[32,64,112,112] and src2[32,1,56,56]")
        quit()

    out_tensor = torch.tensor(np.zeros((32, 64, 56, 56)))

    start_time = time.time()

    # Since padding is non-zero, the src1 is implicitly padded with negative infinity on both sides
    # for padding number of points.
    # Since the stride is 2, the stride of the window is 2.
    for i in range(n_1):
        for j in range(c_1):
            # The first window of the first row
            out_tensor[i, j, 0, 0] = find_max(in_tensor_1, i, j, 0, 2, 0, 2) + in_tensor_2[i, 0, 0, 0]
            # Following windows of the first row
            for col_in in range(1, 56):
                out_tensor[i, j, 0, col_in] = find_max(in_tensor_1, i, j, 0, 2, col_in * 2 - 1, (col_in + 1) * 2) + \
                                              in_tensor_2[i, 0, 0, col_in]
            # Go through each row
            for row_in in range(1, 56):
                # The first window of each row
                out_tensor[i, j, row_in, 0] = find_max(in_tensor_1, i, j, row_in * 2 - 1, (row_in + 1) * 2, 0, 2) + \
                                              in_tensor_2[i, 0, row_in, 0]
                # Go through each col
                for col_index in range(1, 56):
                    out_tensor[i, j, row_in, col_index] = find_max(
                        in_tensor_1, i, j, row_in * 2 - 1, (row_in + 1) * 2, col_index * 2 - 1, (col_index + 1) * 2) + \
                                                       in_tensor_2[i, 0, row_in, col_index]

    print("The execution time is: " + str(time.time() - start_time))
    return out_tensor


# Helper method to find the max elements of an 4D tensor
def find_max(in_tensor, n, c, row_l, row_h, col_l, col_h):
    local_max = in_tensor[n, c, row_l, col_l]
    for i in range(row_l, row_h):
        for j in range(col_l, col_h):
            if in_tensor[n, c, i, j] > local_max:
                local_max = in_tensor[n, c, i, j]

    return local_max


# Verification method
def verification(in_tensor_1, in_tensor_2):
    max_pool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

    max_pool_input_1 = max_pool(in_tensor_1)
    return max_pool_input_1 + in_tensor_2


# Unit test
class IntelTestCase(unittest.TestCase):
    def test_result(self):
        src1 = torch.rand(32, 64, 112, 112)
        src2 = torch.rand(32, 1, 56, 56)
        dst = intel_formula(src1, src2)

        dst_test = verification(src1, src2)
        res = torch.eq(dst, dst_test).sum()

        # Expected same elements
        exp = 32 * 64 * 56 * 56
        self.assertEqual(res, exp)


if __name__ == '__main__':
    # Test the function
    unittest.main()
