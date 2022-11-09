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

        MTTensor *lr     = mt_new_scalar(ctx, 0.01);
        MTTensor *sumsq  = NULL;
        MTTensor *msumsq = NULL;
        MTTensor *scale  = mt_new_scalar(ctx, 1 / 6.0);

        for (int i = 0; i < 100; i++) {
                mt_tensor_zero_grad(x);

                sumsq  = mt_tensor_sum(mt_tensor_mul(x, x), -1, 0);
                msumsq = mt_tensor_mul(sumsq, scale);

                mt_tensor_backward(msumsq, NULL);

                /*
                Gradient descent update rule:
                x = x - lr * ∇x
                */
                x = mt_tensor_sub(x, mt_tensor_mul(lr, x->grad));

                printf("%d sum of squared = %f\n", i, mt_tensor_get_v(msumsq));
        }

        mt_context_free(ctx);
}