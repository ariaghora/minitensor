#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "minitensor.h"

/**
 * A helper to add dependency of a tensor (as a node in computation graph).
 * A tensor will be added as a dependency iff it requires grad.
 */
inline void __mt_push_deps_at(MTTensor *t, MTTensor *dep, int at,
                              TensorBackwardFunc grad_fn) {
        if (dep->req_grad) {
                t->deps[at]  = dep;
                dep->parent  = t;
                dep->grad_fn = grad_fn;
        } else {
                t->deps[at] = NULL;
        }
        t->ndeps++;
}

/**
 * The low-level implementation of general tensor reduce at a certain dimension
 * with reduce function `bfunc`. For example, if dim=0 and bfunc=mt_tensor_add,
 * then it is equivalent to summing the tensor on the first dimension.
 */
MTTensor *mt_tensor_reduce(MTTensor *t, int dim, TensorBFunc bfunc,
                           int keepdims) {
        MTTensor *res = mt_tensor_slice(t->context, t, dim, Arr(int, 0), 1);
        for (long i = 1; i < t->shape[dim]; i++) {
                MTTensor *sl  = mt_tensor_slice(t->context, t,
                                                dim, Arr(int, i), 1);
                MTTensor *tmp = res;
                res           = bfunc(res, sl);
                mt_tensor_free(tmp);
                mt_tensor_free(sl);
        }
        if (!keepdims) {
                mt_squeeze_at_dim(dim, res->shape, res->strides,
                                  res->indices, res->ndims);
                res->ndims--;
        }

        /* There will be possibly many allocations (and deallocations) inside
         * above loop, so we might better defrag the context here. */
        mt_context_defrag(t->context);

        return res;
}

/**
 * The low-level implementation of general binary functions. Typically we
 * don't use this directly (in the user's code). This function is used to
 * help defining more "concrete" binary functions, e.g., addition, subtract-
 * ion, division, etc.
 */
MTTensor *mt_tensor_bfunc(MTTensor *a, MTTensor *b, BFunc bfunc) {
        if (a->context != b->context)
                EXIT_WITH_ERROR("a and b cannot be in different context");

        /* We first attempt broadcasting and return early when broadcasting rule
         * unfullfilled */
        BcastResult bcr = mt_broadcast_lr(a, b);
        if (bcr.status == BC_STATUS_FAILURE)
                EXIT_WITH_ERROR("a and b have incompatible sizes");

        a = bcr.left == NULL ? a : bcr.left;
        b = bcr.right == NULL ? b : bcr.right;

        float *resdata = mt_newptr(
            float,
            a->ndims > b->ndims ? a->datalen : b->datalen);

        if (bcr.status == BC_STATUS_SKIP_SCALAR_HANDLING) {
                /* Case 1, when the broadcasting result suggests tensor-scalar
                 * or scalar-scalar binary operation */
                if (a->ndims == 0) {
                        float val = mt_tensor_get_v(a);
                        for (long i = 0; i < b->datalen; i++)
                                resdata[i] = bfunc(val, b->data[i]);
                } else {
                        float val = mt_tensor_get_v(b);
                        for (long i = 0; i < a->datalen; i++)
                                resdata[i] = bfunc(a->data[i], val);
                }
        } else {
                /* Case 2, when the broadcasting result suggests tensor-tensor
                 * binary operation */
                for (long i = 0; i < a->datalen; i++)
                        resdata[i] = bfunc(a->data[i], b->data[i]);
        }

        /* Reaching this line means that either broadcasting is successful or
         * no broadcasting is required. We can use `resdata` to create a new
         * tensor to return, while using the new shape and ndims. We can now
         * assume that a->shape == b->shape and a->ndims == b->ndims, so using
         * information only from a (or b) alone is fine. */
        MTTensor *res = mt_new_tensor(
            a->context,
            resdata,
            a->shape,
            a->ndims);
        res->isleaf = 0;

        free(resdata);
        mt_tensor_free(bcr.left), mt_tensor_free(bcr.right);

        /**
         * Defrag a->context, but no need to defrag b->context since they point
         * to the same address */
        mt_context_defrag(a->context);

        return res;
}

/**
 * The low-level implementation of general unary functions. Typically we
 * don't use this directly (in the user's code). This function is used to
 * help defining more "concrete" unary functions, e.g., negation, recip-
 * rocation, exponentiation, etc.
 */
MTTensor *mt_tensor_ufunc(MTTensor *t, UFunc ufunc) {
        float *resdata = mt_newptr(float, t->datalen);
        for (int i = 0; i < t->datalen; i++)
                resdata[i] = ufunc(t->data[i]);

        MTTensor *res = mt_new_tensor(t->context, resdata, t->shape, t->ndims);
        if (t->req_grad) {
                mt_tensor_enable_grad(res);
        }
        free(resdata);
        return res;
}

/**
 * The following subsection defines the implementation of arithmetical
 * operations on tensors, along with their respective backward functions
 * for all their dependencies (if any/defined).
 */

#define __set_grad_fn(t, fn) ({ t->grad_fn = t->req_grad ? fn : NULL; })

MTTensor *__mt_grad_unbroadcast(MTTensor *grad, MTTensor *wrt_tensor) {
        /* sum out added dims */
        int ndims_added = grad->ndims - wrt_tensor->ndims;
        for (int i = 0; i < ndims_added; i++)
                grad = mt_tensor_sum(grad, 0, 0);
        /* sum across broadcasted (but non-added dims) */
        for (int i = 0; i < wrt_tensor->ndims; i++) {
                if (wrt_tensor->shape[i] == 1)
                        grad = mt_tensor_sum(grad, i, 1);
        }
        return grad;
}

/* addition operation */
float     __add(float a, float b) { return a + b; }
MTTensor *__add_backward_a(MTTensor **prtdeps, MTTensor *grad) {
        MTTensor *a = prtdeps[0];
        return __mt_grad_unbroadcast(grad, a);
}
MTTensor *__add_backward_b(MTTensor **prtdeps, MTTensor *grad) {
        MTTensor *b = prtdeps[1];
        return __mt_grad_unbroadcast(grad, b);
}
MTTensor *mt_tensor_add(MTTensor *a, MTTensor *b) {
        MTTensor *res = mt_tensor_bfunc(a, b, __add);
        if (a->req_grad || b->req_grad) mt_tensor_enable_grad(res);

        __mt_push_deps_at(res, a, 0, __add_backward_a);
        __mt_push_deps_at(res, b, 1, __add_backward_b);

        return res;
}

/* subtraction operation */
inline float __sub(float a, float b) { return a - b; }
MTTensor    *__sub_backward_a(MTTensor **prtdeps, MTTensor *grad) {
           MTTensor *a = prtdeps[0];
           return __mt_grad_unbroadcast(grad, a);
}
MTTensor *__sub_backward_b(MTTensor **prtdeps, MTTensor *grad) {
        MTTensor *b = prtdeps[1];
        return __mt_grad_unbroadcast(mt_tensor_neg(grad), b);
}
MTTensor *mt_tensor_sub(MTTensor *a, MTTensor *b) {
        MTTensor *res = mt_tensor_bfunc(a, b, __sub);
        if (a->req_grad || b->req_grad) mt_tensor_enable_grad(res);
        __mt_push_deps_at(res, a, 0, __sub_backward_a);
        __mt_push_deps_at(res, b, 1, __sub_backward_b);
        return res;
}

/* element-wise multiplication operation */
inline float __mul(float a, float b) { return a * b; }
MTTensor    *mt_tensor_mul(MTTensor *a, MTTensor *b) {
           MTTensor *res = mt_tensor_bfunc(a, b, __mul);
           return res;
}

/* negation operation */
inline float __neg(float x) { return -x; }
MTTensor    *mt_tensor_neg(MTTensor *t) {
           MTTensor *res = mt_tensor_ufunc(t, __neg);
           return res;
}

/* sum operation */
MTTensor *__sum_backward(MTTensor **prtdeps, MTTensor *grad) {
        MTTensor *self = prtdeps[0];
        MTTensor *ones = mt_new_tensor_full(grad->context, 1.0, self->shape,
                                            self->ndims);
        return mt_tensor_mul(grad, ones);
}
MTTensor *mt_tensor_sum(MTTensor *t, int dim, int keepdim) {
        if (dim > -1) return mt_tensor_reduce(t, dim, mt_tensor_add, keepdim);

        float sum = 0;
        for (long i = 0; i < t->datalen; i++)
                sum += t->data[i];

        MTTensor *res = NULL;
        if (!keepdim) {
                res = mt_new_scalar(t->context, sum);
        } else {
                int shape[t->ndims];
                for (int i = 0; i < t->ndims; i++) shape[i] = 1;
                res = mt_new_tensor(t->context, Arr(float, sum), shape, t->ndims);
        }

        res->isleaf = 0;
        if (t->req_grad) {
                mt_tensor_enable_grad(res);
                __mt_push_deps_at(res, t, 0, __sum_backward);
        }
        return res;
}