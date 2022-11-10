#include "minitensor.h"

#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#define INITIAL_CAP 8
#define INITIAL_N_DEPS 4
#define MT_EPS 1e-6

#define __mt_newptr(type, len) ((type *)calloc((len), sizeof(type)))
#define __mt_memcpy(to, from, len) (memcpy(to, from, (len) * sizeof(*from)))
#define __mt_arrsame_eps(a, b, len) ({                                       \
        int __mt_issame = 1;                                                 \
        for (long i = 0; i < len; i++)                                       \
                __mt_issame = __mt_issame && (abs(a[i]) - (b[i]) <= MT_EPS); \
        __mt_issame;                                                         \
})
#define __mt_arrsame(a, b, len) ({                               \
        int __mt_issame = 1;                                     \
        for (long i = 0; i < len; i++)                           \
                __mt_issame = __mt_issame && ((a[i]) == (b[i])); \
        __mt_issame;                                             \
})
#define EXIT_WITH_ERROR(msg) ({     \
        printf("error: %s\n", msg); \
        exit(1);                    \
})

#define __find_in_list(container, to_find, len) ({   \
        int __found = -1;                            \
        for (int __i = 0; __i < (len); __i++)        \
                if ((to_find) == (container)[__i]) { \
                        __found = __i;               \
                        break;                       \
                }                                    \
        __found;                                     \
})

#define __max(a, b) (a > b ? a : b)
#define __min(a, b) (a < b ? a : b)

#define __printarr(arr, len, format) ({                  \
        printf("{");                                     \
        for (int __i = 0; __i < len; __i++) {            \
                printf(format, arr[__i]);                \
                printf("%s", __i < len - 1 ? ", " : ""); \
        }                                                \
        printf("}");                                     \
})

#define __prod(arr, len, T) ({        \
        T __p = 1;                    \
        for (int i = 0; i < len; i++) \
                __p *= arr[i];        \
        __p;                          \
})

MTTensor *mt_alloc_empty_tensor(MTContext *ctx) {
        MTTensor *t = __mt_newptr(MTTensor, 1);
        t->context  = ctx;
        t->data     = NULL;
        t->datalen  = 0;
        t->deps     = __mt_newptr(Dependency *, INITIAL_N_DEPS);
        t->grad     = NULL;
        t->indices  = NULL;
        t->isleaf   = 1;
        t->ndeps    = 0;
        t->ndims    = 0;
        t->parent   = NULL;
        t->req_grad = 0;
        t->shape    = NULL;
        t->strides  = NULL;
        mt_context_push_tensor(ctx, t);
        return t;
}

inline void __init_strides(MTTensor *t) {
        t->strides = __mt_newptr(int, t->ndims);
        for (int i = 0; i < t->ndims; i++) {
                int prod = 1;
                for (int j = i + 1; j < t->ndims; j++) {
                        prod *= t->shape[j];
                }
                t->strides[i] = prod;
        }
}

void __free_indices(MTTensor *t) {
        if (t->indices != NULL)
                for (int i = 0; i < t->ndims; i++)
                        free(t->indices[i]);
        free(t->indices);
}

inline void __init_indices(MTTensor *t) {
        if (t->indices != NULL) __free_indices(t);

        t->indices = __mt_newptr(int *, t->ndims);
        for (int i = 0; i < t->ndims; i++) {
                int *idx = __mt_newptr(int, t->shape[i]);
                for (long j = 0; j < t->shape[i]; j++) idx[j] = j;

                t->indices[i] = idx;
        }
}

MTTensor *mt_new_tensor(MTContext *context,
                        float *data, int *shape,
                        int ndims) {
        int datalen = __prod(shape, ndims, int);

        MTTensor *t = mt_alloc_empty_tensor(context);
        t->data     = __mt_newptr(float, datalen);
        t->datalen  = datalen;
        t->ndims    = ndims;
        t->shape    = __mt_newptr(int, ndims);
        __mt_memcpy(t->data, data, datalen);
        __mt_memcpy(t->shape, shape, ndims);
        __init_strides(t);
        __init_indices(t);
        return t;
}

MTTensor *mt_new_tensor_full(MTContext *ctx, float val,
                             int *shape, int ndims) {
        int    datalen = __prod(shape, ndims, int);
        float *data    = __mt_newptr(float, datalen);
        for (long i = 0; i < datalen; i++) data[i] = val;
        MTTensor *t = mt_new_tensor(ctx, data, shape, ndims);
        free(data);
        return t;
}

inline float mt_tensor_get(MTTensor *t, int *idx, int ndims) {
        switch (ndims) {
                case 0:
                        return mt_tensor_get_v(t);
                case 1:
                        return mt_tensor_get_1(t, idx[0]);
                case 2:
                        return mt_tensor_get_2(t, idx[0], idx[1]);
                case 3:
                        return mt_tensor_get_3(t, idx[0], idx[1], idx[2]);
                default:
                        EXIT_WITH_ERROR("tensor access for order >3 is unsupported yet");
                        break;
        }
        return 0;
}

inline float mt_tensor_get_v(MTTensor *t) {
        if (t->ndims != 0) EXIT_WITH_ERROR("t must be 0-tensor");
        return t->data[0];
}

inline float mt_tensor_get_1(MTTensor *t, int i) {
        if (t->ndims != 1) EXIT_WITH_ERROR("t must be 1-tensor");
        return t->data[i];
}

inline float mt_tensor_get_2(MTTensor *t, int i, int j) {
        if (t->ndims != 2) EXIT_WITH_ERROR("t must be 2-tensor");
        return t->data[i * t->strides[0] + j * t->strides[1]];
}

inline float mt_tensor_get_3(MTTensor *t, int i, int j, int k) {
        if (t->ndims != 3) EXIT_WITH_ERROR("t must be 3-tensor");
        return t->data[i * t->strides[0] + j * t->strides[1] + k * t->strides[2]];
}

MTTensor *mt_new_scalar(MTContext *context, float val) {
        float sc[] = {val};
        return mt_new_tensor(context, sc, NULL, 0);
}

/**
 * IdxIterator helps the iteration through multidimensional indices of
 * a tensor. For example, a tensor with shape 3 by 3 has indices of {0, 0},
 * {0, 1}, {0, 2}, {1, 0}, ..., {2, 2}. IdxIterator can generate next
 * subsequent index efficiently without storing all possible permutations of
 * indices to begin with.
 */
typedef struct {
        int **indices;
        int  *nextidx;
        int  *nextactidx;
        int  *shape;
        int   ndims;
        int   nxt;
        int   i;
} IdxIterator;

/**
 * Allocate a new IdxIterator. Each element of `indices` is an integer array
 * containing index we want to iterate through, from each dimension.
 */
IdxIterator *mt_new_idxiterator(int **indices, int *shape, int ndims) {
        IdxIterator *it = __mt_newptr(IdxIterator, 1);
        it->ndims       = ndims;
        it->nextidx     = __mt_newptr(int, ndims);
        it->nextactidx  = __mt_newptr(int, ndims);
        it->indices     = indices;
        it->shape       = shape;
        it->i           = -1;
        it->nxt         = 0;

        for (int i = 0; i < it->ndims; i++)
                it->nextidx[i] = 0;
        for (int i = 0; i < it->ndims; i++)
                it->nextactidx[i] = it->indices[i][it->nextidx[i]];
        return it;
}

/* Get the next (subsequent) multidimensional index from an IdxIterator */
int *mt_idxiterator_next(IdxIterator *it) {
        if (it->i++ < 0) return it->nextactidx;

        it->nxt = it->ndims - 1;
        while (it->nxt >= 0 && (it->nextidx[it->nxt] + 1 >= it->shape[it->nxt]))
                it->nxt--;

        if (it->nxt < 0)
                EXIT_WITH_ERROR("trying to access out-of-bound indices");

        it->nextidx[it->nxt]++;
        for (int i = it->nxt + 1; i < it->ndims; i++) it->nextidx[i] = 0;

        for (int i = 0; i < it->ndims; i++)
                it->nextactidx[i] = it->indices[i][it->nextidx[i]];
        return it->nextactidx;
}

void mt_idxiterator_free(IdxIterator *it) {
        free(it->nextidx);
        free(it->nextactidx);
        free(it);
}

MTTensor *mt_tensor_slice(MTContext *ctx, MTTensor *t, int dim,
                          int *index, int indexlen) {
        int **newindices = __mt_newptr(int *, t->ndims);
        int  *newshape   = __mt_newptr(int, t->ndims);
        for (int i = 0; i < t->ndims; i++) {
                if (i == dim) {
                        newshape[i]   = indexlen;
                        newindices[i] = index;
                } else {
                        newshape[i]   = t->shape[i];
                        newindices[i] = t->indices[i];
                }
        }

        int    newlen  = __prod(newshape, t->ndims, int);
        float *newdata = __mt_newptr(float, newlen);

        IdxIterator *it = mt_new_idxiterator(newindices, newshape, t->ndims);
        for (int i = 0; i < newlen; i++) {
                int *idxs  = mt_idxiterator_next(it);
                newdata[i] = mt_tensor_get(t, idxs, t->ndims);
        }
        MTTensor *newtensor = mt_new_tensor(ctx, newdata, newshape, t->ndims);
        newtensor->isleaf   = t->isleaf;
        mt_idxiterator_free(it), free(newshape), free(newindices), free(newdata);

        return newtensor;
}

/**
 * Access the tensor data with customized indices, shape, strides, and ndims
 * constraints. This is useful for especially to access data of a tensor
 * without necessarily obeying contiguous order. For example, we can pass
 * swapped shape and swapped strides of a tensor into this function (and fix
 * the other variables) to get the tensor data in a transposed order.
 */
float *mt_tensor_get_all_data_constrained(MTTensor *t, int **indices,
                                          int *shape, int *strides, int ndims) {
        /* Temporarily alter the tensor's properties */
        int **oldindices = t->indices;
        int  *oldshape   = t->shape;
        int  *oldstrides = t->strides;
        int   oldndims   = t->ndims;

        t->indices = indices;
        t->shape   = shape;
        t->strides = strides;
        t->ndims   = ndims;

        IdxIterator *it     = mt_new_idxiterator(indices, shape, ndims);
        int          outlen = __prod(shape, ndims, int);
        float       *res    = __mt_newptr(float, outlen);
        for (int i = 0; i < outlen; i++) {
                int *nextidx = mt_idxiterator_next(it);
                res[i]       = mt_tensor_get(t, nextidx, ndims);
        }

        mt_idxiterator_free(it);

        /* Restore the tensor's properties */
        t->indices = oldindices;
        t->shape   = oldshape;
        t->strides = oldstrides;
        t->ndims   = oldndims;
        return res;
}

int *__range(int start, int end) {
        int *r = __mt_newptr(int, end - start);
        for (int i = (start); i < (end); i++)
                r[i] = i;
        return r;
}

BcastResult mt_broadcast_lr(MTTensor *left, MTTensor *right) {
        BcastResult res = {.left = NULL, .right = NULL, .status = BC_STATUS_FAILURE};

        /**
         * Return early (with both left and right BcastResult NULL) if both
         * tensors have the same shape OR if one of them is a scalar.
         */
        if (left->ndims == 0 || right->ndims == 0) {
                res.status = BC_STATUS_SKIP_SCALAR_HANDLING;
                return res;
        }
        if (left->ndims == right->ndims)
                if (__mt_arrsame(left->shape, right->shape, left->ndims)) {
                        res.status = BC_STATUS_NO_BCAST_REQUIRED;
                        return res;
                }

        /**
         * We are going to build target left and right shapes. If either of
         * the tensors has less dims, prepend the shape array until both have
         * the same number of dimensions.
         */
        int   outndims = __max(left->ndims, right->ndims);
        int   lnewshape[outndims], ltmpstrides[outndims];
        int   rnewshape[outndims], rtmpstrides[outndims];
        int **lnewindices = __mt_newptr(int *, outndims);
        int **rnewindices = __mt_newptr(int *, outndims);
        int   lddiff      = abs(outndims - left->ndims);
        int   rddiff      = abs(outndims - right->ndims);

        for (int i = 0; i < outndims; i++) {
                lnewshape[i]   = i < lddiff ? 1 : left->shape[i - lddiff];
                ltmpstrides[i] = i < lddiff ? 0 : left->strides[i - lddiff];
                rnewshape[i]   = i < rddiff ? 1 : right->shape[i - rddiff];
                rtmpstrides[i] = i < rddiff ? 0 : right->strides[i - rddiff];

                if (lnewshape[i] != rnewshape[i]) {
                        ltmpstrides[i] = lnewshape[i] < rnewshape[i] ? 0 : ltmpstrides[i];
                        rtmpstrides[i] = lnewshape[i] > rnewshape[i] ? 0 : rtmpstrides[i];

                        if (lnewshape[i] == 1 || rnewshape[i] == 1) {
                                lnewshape[i] = __max(lnewshape[i], rnewshape[i]);
                                rnewshape[i] = __max(lnewshape[i], rnewshape[i]);
                        } else {
                                res.status = BC_STATUS_FAILURE;
                                return res;
                        }
                }

                lnewindices[i] = __range(0, lnewshape[i]);
                rnewindices[i] = __range(0, rnewshape[i]);
        }

        /**
         * Determine whether we should allocate new tensor if left/right
         * is broadcast.
         * when lnewshape == left->shape, then res.left should be NULL,
         * and no new tensor allocation is required. The same applies on right
         * side.
         */

        int lshouldbc = 0;
        lshouldbc     = outndims != left->ndims;
        if (!lshouldbc && (left->ndims == outndims))
                if (!__mt_arrsame(lnewshape, left->shape, left->ndims))
                        lshouldbc = 1;

        int rshouldbc = 0;
        rshouldbc     = outndims != right->ndims;
        if (!rshouldbc && (right->ndims == outndims))
                if (!__mt_arrsame(rnewshape, right->shape, right->ndims))
                        rshouldbc = 1;

        float *ldata = NULL;
        if (lshouldbc) {
                ldata            = mt_tensor_get_all_data_constrained(left,
                                                                      lnewindices,
                                                                      lnewshape,
                                                                      ltmpstrides,
                                                                      outndims);
                res.left         = mt_new_tensor(left->context, ldata,
                                                 lnewshape, outndims);
                res.left->isleaf = left->isleaf;
        }

        float *rdata = NULL;
        if (rshouldbc) {
                rdata             = mt_tensor_get_all_data_constrained(right,
                                                                       rnewindices,
                                                                       rnewshape,
                                                                       rtmpstrides,
                                                                       outndims);
                res.right         = mt_new_tensor(right->context, rdata,
                                                  rnewshape, outndims);
                res.right->isleaf = right->isleaf;
        }

        free(ldata), free(rdata);
        for (int i = 0; i < outndims; i++) {
                free(lnewindices[i]), free(rnewindices[i]);
        }
        free(lnewindices), free(rnewindices);

        res.status = BC_STATUS_SUCCESS;
        return res;
}

void mt_squeeze_at_dim(int targetdim, int *shape, int *strides, int **indices, int ndims) {
        if (shape[targetdim] != 1) return;
        free(indices[targetdim]);
        for (int i = targetdim; i < ndims - 1; i++) {
                shape[targetdim]   = shape[targetdim + 1];
                strides[targetdim] = strides[targetdim + 1];
                indices[targetdim] = indices[targetdim + 1];
        }
}

void mt_tensor_free(MTTensor *t) {
        if (t != NULL) {
                /* Null the index of node's tracker that points to this node */
                int idxtracker = __find_in_list(t->context->tracked,
                                                t,
                                                t->context->ntracked);
                if (idxtracker > -1) t->context->tracked[idxtracker] = NULL;

                for (int i = 0; i < t->ndeps; i++) free(t->deps[i]);

                free(t->deps);

                free(t->data);
                free(t->shape);
                free(t->strides);
                __free_indices(t);
                free(t);
        }
}

/* remove NULLs in the tracked list */
void mt_context_defrag(MTContext *ctx) {
        MTTensor **newtracked = __mt_newptr(MTTensor *, ctx->ntracked);

        int cnt = 0;
        for (int i = 0; i < ctx->ntracked; i++)
                if (ctx->tracked[i] != NULL) newtracked[cnt++] = ctx->tracked[i];
        // free the old `tracked` pointers
        free(ctx->tracked);
        ctx->ntracked = cnt;
        ctx->tracked  = newtracked;
        ctx->tracked  = realloc(ctx->tracked, ctx->cap * sizeof(*ctx->tracked));
}

void mt_context_free(MTContext *ctx) {
        for (int i = 0; i < ctx->ntracked; i++) {
                if (ctx->tracked[i] != NULL) {
                        mt_tensor_free(ctx->tracked[i]);
                        ctx->tracked[i] = NULL;
                }
        }
        free(ctx->tracked);
        free(ctx);
}

MTContext *mt_new_context(void) {
        MTContext *ctx = __mt_newptr(MTContext, 1);
        ctx->withgrads = CGM_OVERRIDE;
        ctx->ntracked  = 0;
        ctx->cap       = INITIAL_CAP;
        ctx->tracked   = __mt_newptr(MTTensor *, INITIAL_CAP);
        return ctx;
}

void mt_context_push_tensor(MTContext *ctx, MTTensor *t) {
        ctx->tracked[ctx->ntracked] = t;
        ctx->ntracked++;
        if (ctx->ntracked >= ctx->cap / 2) {
                ctx->cap *= 2;
                size_t sz    = sizeof(*ctx->tracked);
                ctx->tracked = (MTTensor **)realloc(ctx->tracked, ctx->cap * sz);
        }
}

void mt_tensor_print_debug(MTTensor *t) {
        printf("ndims   : %d\n", t->ndims);
        printf("ndeps   : %d\n", t->ndeps);
        printf("shape   : "), __printarr(t->shape, t->ndims, "%d"), printf("\n");
        printf("strides : "), __printarr(t->strides, t->ndims, "%d"), printf("\n");
        printf("data \n");
        printf("  - datalen : %ld\n", t->datalen);
        printf("  - content : ");
        if (t->ndims > 0)
                __printarr(t->data, t->datalen, "%.2f");
        else
                printf("%f", mt_tensor_get_v(t));
        printf("\n");
        printf("\n");
}

int mt_is_tensor_eq(MTTensor *a, MTTensor *b) {
        /* NULL guard */
        if ((a == NULL) && (b != NULL)) return 0;
        if ((a != NULL) && (b == NULL)) return 0;

        if (a->ndims != b->ndims) return 0;
        return __mt_arrsame(a->data, b->data, a->datalen) &&
               __mt_arrsame(a->shape, b->shape, a->ndims);
}

int mt_is_tensor_almost_eq(MTTensor *a, MTTensor *b) {
        /* NULL guard */
        if ((a == NULL) && (b != NULL)) return 0;
        if ((a != NULL) && (b == NULL)) return 0;

        if (a->ndims != b->ndims) return 0;
        return __mt_arrsame_eps(a->data, b->data, a->datalen) &&
               __mt_arrsame_eps(a->shape, b->shape, a->ndims);
}

/**
 * Math implementation
 */

inline float __add(float a, float b) { return a + b; }
inline float __sub(float a, float b) { return a - b; }
inline float __mul(float a, float b) { return a * b; }
inline float __div(float a, float b) { return a / b; }
inline float __neg(float x) { return -x; }
inline float __mt_log(float x) { return logf(x); }
MTTensor    *__mt_tensor_sum(MTTensor *t, int dim, int keepdim);
MTTensor    *__mt_tensor_add(MTTensor *a, MTTensor *b);
MTTensor    *__mt_tensor_sub(MTTensor *a, MTTensor *b);
MTTensor    *__mt_tensor_mul(MTTensor *a, MTTensor *b);
MTTensor    *__mt_tensor_matmul(MTTensor *a, MTTensor *b);
MTTensor    *__mt_tensor_div(MTTensor *a, MTTensor *b);
MTTensor    *__mt_tensor_neg(MTTensor *t);
MTTensor    *__mt_tensor_transpose(MTTensor *t);

/**
 * A helper to add dependency of a tensor (as a node in computation graph).
 * A tensor will be added as a dependency iff it requires grad.
 */
inline void __mt_push_deps_at(MTTensor *t, MTTensor *t_dep, int at,
                              TensorBackwardFunc grad_fn) {
        if (t_dep->req_grad) {
                Dependency *dep = __mt_newptr(Dependency, 1);
                dep->tensor     = t_dep;
                dep->grad_fn    = grad_fn;
                t->deps[at]     = dep;
                t_dep->parent   = t;
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
MTTensor *mt_tensor_reduce(MTTensor *t, int dim, BFunc bfunc,
                           int keepdims) {
        MTTensor *res = mt_tensor_slice(t->context, t, dim, Arr(int, 0), 1);

        for (long i = 1; i < t->shape[dim]; i++) {
                MTTensor *sl = mt_tensor_slice(t->context, t,
                                               dim, Arr(int, i), 1);

                float *sldata    = sl->data;
                long   sldatalen = sl->datalen;
                for (long j = 0; j < sldatalen; j++)
                        res->data[j] = bfunc(res->data[j], sldata[j]);

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

        float *resdata = __mt_newptr(
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
        float *resdata = __mt_newptr(float, t->datalen);
        for (int i = 0; i < t->datalen; i++)
                resdata[i] = ufunc(t->data[i]);

        MTTensor *res = mt_new_tensor(t->context, resdata, t->shape, t->ndims);
        if (t->req_grad) {
                mt_tensor_enable_grad(res);
        }
        free(resdata);
        res->isleaf = 0;
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
                grad = __mt_tensor_sum(grad, 0, 0);
        /* sum across broadcasted (but non-added dims) */
        for (int i = 0; i < wrt_tensor->ndims; i++) {
                if (wrt_tensor->shape[i] == 1)
                        grad = __mt_tensor_sum(grad, i, 1);
        }
        return grad;
}

/* addition operation */
MTTensor *__add_backward_a(Dependency **prtdeps, MTTensor *grad) {
        MTTensor *a = prtdeps[0]->tensor;
        return __mt_grad_unbroadcast(grad, a);
}

MTTensor *__add_backward_b(Dependency **prtdeps, MTTensor *grad) {
        MTTensor *b = prtdeps[1]->tensor;
        return __mt_grad_unbroadcast(grad, b);
}

MTTensor *__mt_tensor_add(MTTensor *a, MTTensor *b) {
        return mt_tensor_bfunc(a, b, __add);
}

MTTensor *mt_tensor_add(MTTensor *a, MTTensor *b) {
        MTTensor *res = __mt_tensor_add(a, b);
        if (a->req_grad || b->req_grad) mt_tensor_enable_grad(res);
        __mt_push_deps_at(res, a, 0, __add_backward_a);
        __mt_push_deps_at(res, b, 1, __add_backward_b);
        return res;
}

/* subtraction operation */
MTTensor *__sub_backward_a(Dependency **prtdeps, MTTensor *grad) {
        MTTensor *a = prtdeps[0]->tensor;
        return __mt_grad_unbroadcast(grad, a);
}

MTTensor *__sub_backward_b(Dependency **prtdeps, MTTensor *grad) {
        MTTensor *b = prtdeps[1]->tensor;
        return __mt_grad_unbroadcast(__mt_tensor_neg(grad), b);
}

MTTensor *__mt_tensor_sub(MTTensor *a, MTTensor *b) {
        return mt_tensor_bfunc(a, b, __sub);
}

MTTensor *mt_tensor_sub(MTTensor *a, MTTensor *b) {
        MTTensor *res = __mt_tensor_sub(a, b);
        if (a->req_grad || b->req_grad) mt_tensor_enable_grad(res);
        __mt_push_deps_at(res, a, 0, __sub_backward_a);
        __mt_push_deps_at(res, b, 1, __sub_backward_b);
        return res;
}

/* element-wise multiplication operation */
MTTensor *__mul_backward_a(Dependency **prtdeps, MTTensor *grad) {
        MTTensor *b = prtdeps[1]->tensor;
        grad        = __mt_tensor_mul(grad, b);
        return __mt_grad_unbroadcast(grad, b);
}

MTTensor *__mul_backward_b(Dependency **prtdeps, MTTensor *grad) {
        MTTensor *a = prtdeps[0]->tensor;
        grad        = __mt_tensor_mul(grad, a);
        return __mt_grad_unbroadcast(grad, a);
}

MTTensor *__mt_tensor_mul(MTTensor *a, MTTensor *b) {
        return mt_tensor_bfunc(a, b, __mul);
}

MTTensor *mt_tensor_mul(MTTensor *a, MTTensor *b) {
        MTTensor *res = __mt_tensor_mul(a, b);
        if (a->req_grad || b->req_grad) mt_tensor_enable_grad(res);
        __mt_push_deps_at(res, a, 0, __mul_backward_a);
        __mt_push_deps_at(res, b, 1, __mul_backward_b);
        return res;
}

/* matrix multiplication operation*/
inline void __mt_axpy(float *resdata, MTTensor *a, MTTensor *b,
                      int i, int j, int k) {
        resdata[i * b->shape[1] + j] +=
            mt_tensor_get_2(a, i, k) * mt_tensor_get_2(b, k, j);
}

MTTensor *__mt_tensor_matmul(MTTensor *a, MTTensor *b) {
        if ((a->ndims != 2) || (b->ndims != 2))
                EXIT_WITH_ERROR("both a and b must be 2-tensor");
        if (a->shape[1] != b->shape[0])
                EXIT_WITH_ERROR("the shapes of a and b are incompatible");

        long   resdatalen = a->shape[0] * b->shape[1];
        float *resdata    = __mt_newptr(float, resdatalen);

        for (long i = 0; i < resdatalen; i++) resdata[i] = 0.0;

        for (int i = 0; i < a->shape[0]; i++) {
                for (int j = 0; j < b->shape[1]; j++) {
                        for (int k = 0; k < a->shape[1]; k++) {
                                __mt_axpy(resdata, a, b, i, j, k);
                        }
                }
        }

        MTTensor *res = mt_new_tensor(a->context, resdata,
                                      Arr(int, a->shape[0], b->shape[1]), 2);
        free(resdata);
        return res;
}

MTTensor *__matmul_backward_a(Dependency **prtdeps, MTTensor *grad) {
        MTTensor *b = prtdeps[1]->tensor;
        return __mt_tensor_matmul(grad, __mt_tensor_transpose(b));
}

MTTensor *__matmul_backward_b(Dependency **prtdeps, MTTensor *grad) {
        MTTensor *a = prtdeps[0]->tensor;
        return __mt_tensor_matmul(__mt_tensor_transpose(a), grad);
}

MTTensor *mt_tensor_matmul(MTTensor *a, MTTensor *b) {
        MTTensor *res = __mt_tensor_matmul(a, b);
        if (a->req_grad || b->req_grad) mt_tensor_enable_grad(res);
        __mt_push_deps_at(res, a, 0, __matmul_backward_a);
        __mt_push_deps_at(res, b, 1, __matmul_backward_b);
        return res;
}

/* division operation */
MTTensor *__mt_tensor_div(MTTensor *a, MTTensor *b) {
        return mt_tensor_bfunc(a, b, __div);
}

MTTensor *__div_backward_a(Dependency **prtdeps, MTTensor *grad) {
        MTTensor *a = prtdeps[0]->tensor;
        MTTensor *b = prtdeps[1]->tensor;
        grad        = __mt_tensor_div(grad, b);
        return __mt_grad_unbroadcast(grad, a);
}

MTTensor *__div_backward_b(Dependency **prtdeps, MTTensor *grad) {
        MTTensor *a = prtdeps[0]->tensor;
        MTTensor *b = prtdeps[1]->tensor;

        /* unbroadcast((-grad * a) / (b * b)) w.r.t. b */
        grad = __mt_tensor_mul(__mt_tensor_neg(grad), a);
        grad = __mt_tensor_div(grad, __mt_tensor_mul(b, b));
        return __mt_grad_unbroadcast(grad, b);
}

inline float __recip(float x) { return 1 / x; }

MTTensor *mt_tensor_div(MTTensor *a, MTTensor *b) {
        MTTensor *res = __mt_tensor_div(a, b);
        if (a->req_grad || b->req_grad) mt_tensor_enable_grad(res);
        __mt_push_deps_at(res, a, 0, __div_backward_a);
        __mt_push_deps_at(res, b, 1, __div_backward_b);

        // MTTensor *res = mt_tensor_mul(a, mt_tensor_ufunc(b, __recip));
        return res;
}

/* exponentiation operation */
inline float __expf(float x) { return expf(x); }

MTTensor *__mt_tensor_exp(MTTensor *t) {
        return mt_tensor_ufunc(t, __expf);
}

MTTensor *__exp_backward(Dependency **prtdeps, MTTensor *grad) {
        MTTensor *t = prtdeps[0]->tensor;
        return mt_tensor_mul(grad, __mt_tensor_exp(t));
}

MTTensor *mt_tensor_exp(MTTensor *t) {
        MTTensor *res = __mt_tensor_exp(t);
        if (t->req_grad) mt_tensor_enable_grad(res);
        __mt_push_deps_at(res, t, 0, __exp_backward);
        return res;
}

/* negation operation */
MTTensor *__neg_backward(Dependency **prtdeps, MTTensor *grad) {
        return __mt_tensor_neg(grad);
}

MTTensor *__mt_tensor_neg(MTTensor *t) {
        return mt_tensor_ufunc(t, __neg);
}

MTTensor *mt_tensor_neg(MTTensor *t) {
        MTTensor *res = __mt_tensor_neg(t);
        if (t->req_grad) mt_tensor_enable_grad(res);
        __mt_push_deps_at(res, t, 0, __neg_backward);
        return res;
}

/* (natural) logarithm operation */
MTTensor *__mt_tensor_log(MTTensor *t) {
        return mt_tensor_ufunc(t, __mt_log);
}

MTTensor *__log_backward(Dependency **prtdeps, MTTensor *grad) {
        MTTensor *t = prtdeps[0]->tensor;
        return mt_tensor_bfunc(grad, t, __div);
}

MTTensor *mt_tensor_log(MTTensor *t) {
        MTTensor *res = __mt_tensor_log(t);
        if (t->req_grad) mt_tensor_enable_grad(res);
        __mt_push_deps_at(res, t, 0, __log_backward);
        return res;
}

/* relu operation */
inline float __relu(float x) { return __max(0, x); }
inline float __drelu(float t, float g) { return t > 0 ? g : 0; }

MTTensor *__relu_backward(Dependency **prtdeps, MTTensor *grad) {
        MTTensor *t = prtdeps[0]->tensor;
        return mt_tensor_bfunc(t, grad, __drelu);
}

MTTensor *mt_tensor_relu(MTTensor *t) {
        MTTensor *res = mt_tensor_ufunc(t, __relu);
        if (t->req_grad) mt_tensor_enable_grad(res);
        __mt_push_deps_at(res, t, 0, __relu_backward);
        return res;
}

/* transpose operation */
MTTensor *__mt_tensor_transpose(MTTensor *t) {
        int **indices_tr = __mt_newptr(int *, t->ndims);
        for (int i = t->ndims; i > 0; i--) {
                indices_tr[t->ndims - i] =
                    t->indices[i - 1];
        }
        int shape_tr[t->ndims];
        for (int i = t->ndims; i > 0; i--) {
                shape_tr[t->ndims - i] = t->shape[i - 1];
        }
        int strides_tr[t->ndims];
        for (int i = t->ndims; i > 0; i--) {
                strides_tr[t->ndims - i] = t->strides[i - 1];
        }

        float *transposed_data =
            mt_tensor_get_all_data_constrained(t, indices_tr, shape_tr,
                                               strides_tr, t->ndims);
        MTTensor *res = mt_new_tensor(t->context, transposed_data,
                                      shape_tr, t->ndims);
        free(transposed_data);
        free(indices_tr);
        return res;
}
MTTensor *mt_tensor_transpose(MTTensor *t) {
        MTTensor *res = __mt_tensor_transpose(t);
        return res;
}

/* sum operation */
MTTensor *__sum_backward(Dependency **prtdeps, MTTensor *grad) {
        MTTensor *self = prtdeps[0]->tensor;
        MTTensor *ones = mt_new_tensor_full(grad->context, 1.0, self->shape,
                                            self->ndims);
        return __mt_tensor_mul(grad, ones);
}

MTTensor *__mt_tensor_sum(MTTensor *t, int dim, int keepdim) {
        if (dim > -1) return mt_tensor_reduce(t, dim, __add, keepdim);

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
        return res;
}
MTTensor *mt_tensor_sum(MTTensor *t, int dim, int keepdim) {
        MTTensor *res = __mt_tensor_sum(t, dim, keepdim);
        if (t->req_grad) {
                mt_tensor_enable_grad(res);
                __mt_push_deps_at(res, t, 0, __sum_backward);
        }
        return res;
}

/**
 * AUTOGRAD
 */
void mt_tensor_enable_grad(MTTensor *t) {
        t->req_grad = 1;
        mt_tensor_zero_grad(t);
}

void mt_tensor_disable_grad(MTTensor *t) {
        t->req_grad = 0;
        if (t->grad != NULL) {
                mt_tensor_free(t->grad);
                t->grad = NULL;
                mt_context_defrag(t->context);
        }
}

void mt_tensor_backward(MTTensor *t, MTTensor *grad) {
        if (!t->req_grad) return;

        if (grad == NULL) {
                if (t->ndims == 0)
                        grad = mt_new_scalar(t->context, 1.0);
                else
                        EXIT_WITH_ERROR("grad must be specified for non scalar tensor");
        }
        t->grad = __mt_tensor_add(t->grad,
                                  grad);

        /* recursively compute gradient of t's non-null children */
        for (int i = 0; i < t->ndeps; i++) {
                if (t->deps[i] != NULL) {
                        if (t->deps[i]->grad_fn == NULL)
                                EXIT_WITH_ERROR("fatal: no grad_fn defined");
                        MTTensor *bwgrad = t->deps[i]->grad_fn(t->deps, grad);
                        mt_tensor_backward(t->deps[i]->tensor, bwgrad);
                }
        }
}

void mt_remove_intermediary_nodes(MTContext *ctx) {
        for (int i = 0; i < ctx->ntracked; i++) {
                mt_tensor_free(ctx->tracked[i]);
                ctx->tracked[i] = NULL;
        }
}

void mt_tensor_zero_grad(MTTensor *t) {
        mt_tensor_free(t->grad);
        t->grad = mt_new_tensor_full(t->context, 0., t->shape, t->ndims);
}