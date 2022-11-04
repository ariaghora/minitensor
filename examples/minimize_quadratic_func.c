#include <stdio.h>

#include "../minitensor.h"

int main(void) {
        MTContext *ctx = mt_new_context();

        MTTensor *x = mt_new_tensor(
            ctx,
            Arr(float, 10, -10, 10, -5, 6, 3, 1), /* the data */
            Arr(int, 6),                          /* shape (6,) */
            1);                                   /* ndims */
        mt_tensor_enable_grad(x);

        MTTensor *lr    = mt_new_scalar(ctx, 0.01);
        MTTensor *sumsq = NULL;
        MTTensor *sq    = NULL;

        for (int i = 0; i < 9; i++) {
                mt_tensor_zero_grad(x);
                sq    = mt_tensor_sum(mt_tensor_mul(x, x), -1, 0);
                sumsq = mt_tensor_sum(sq, -1, 0);

                printf("SSE = %f\n", mt_tensor_get_v(sumsq));

                mt_tensor_backward(sumsq, NULL);
                x = mt_tensor_sub(x, mt_tensor_mul(lr, x->grad));
        }

        mt_context_free(ctx);
}