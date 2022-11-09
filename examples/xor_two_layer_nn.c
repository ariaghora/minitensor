#include <stdio.h>

#include "../minitensor.h"

int main(void) {
        MTContext *ctx = mt_new_context();

        /* XOR feature */
        float x_data[8] = {
            0, 0,
            0, 1,
            1, 0,
            1, 1};

        /* XOR target */
        float y_data[4] = {
            0,
            1,
            1,
            0};

        /* Layer 1 weight & bias data */
        float w1_data[6] = {
            0.5, 0.5, 0.5,
            0.5, 0.5, 0.5};
        float b1_data[3] = {1, 1, 1};

        /* Layer 2 weight & bias data */
        float w2_data[3] = {
            0.5,
            0.5,
            0.5};
        float b2_data[1] = {1};

        MTTensor *x  = mt_new_tensor(ctx, x_data, Arr(int, 4, 2), 2);
        MTTensor *y  = mt_new_tensor(ctx, y_data, Arr(int, 4, 1), 2);
        MTTensor *lr = mt_new_scalar(ctx, 0.01); /* learning rate */

        MTTensor *w1 = mt_new_tensor(ctx, w1_data, Arr(int, 2, 3), 2);
        MTTensor *b1 = mt_new_tensor(ctx, b1_data, Arr(int, 3), 1);
        MTTensor *w2 = mt_new_tensor(ctx, w2_data, Arr(int, 3, 1), 2);
        MTTensor *b2 = mt_new_tensor(ctx, b2_data, Arr(int, 1), 1);

        mt_tensor_enable_grad(w1), mt_tensor_enable_grad(b1);
        mt_tensor_enable_grad(w2), mt_tensor_enable_grad(b2);

        (void)(lr);
        (void)(x), (void)(y);
        (void)(w1), (void)(b1);
        (void)(w2), (void)(b2);

        mt_context_free(ctx);
}