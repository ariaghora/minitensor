#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "minitensor.h"

#define INITIAL_CAP 8
#define INITIAL_N_DEPS 4

#define __find_in_list(container, to_find, len) ({   \
        int __found = -1;                            \
        for (int __i = 0; __i < (len); __i++)        \
                if ((to_find) == (container)[__i]) { \
                        __found = __i;               \
                        break;                       \
                }                                    \
        __found;                                     \
})

#define __printarr(arr, len, format) ({          \
        printf("{");                             \
        for (int __i = 0; __i < len; __i++) {    \
                printf(format, arr[__i]);        \
                if (__i < len - 1) printf(", "); \
        }                                        \
        printf("}");                             \
})

#define __prod(arr, len, T) ({        \
        T __p = 1;                    \
        for (int i = 0; i < len; i++) \
                __p *= arr[i];        \
        __p;                          \
})

MTTensor *mt_alloc_empty_tensor(MTContext *ctx) {
        MTTensor *t = mt_newptr(MTTensor, 1);
        t->context  = ctx;
        t->data     = NULL;
        t->datalen  = 0;
        t->deps     = mt_newptr(MTTensor *, INITIAL_N_DEPS);
        t->grad     = NULL;
        t->grad_fn  = NULL;
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
        t->strides = mt_newptr(int, t->ndims);
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
        // if (t->ndims == 0) return;

        t->indices = mt_newptr(int *, t->ndims);
        for (int i = 0; i < t->ndims; i++) {
                int *idx = mt_newptr(int, t->shape[i]);
                for (long j = 0; j < t->shape[i]; j++) idx[j] = j;

                t->indices[i] = idx;
        }
}

MTTensor *mt_new_tensor(MTContext *context,
                        float *data, int *shape,
                        int ndims) {
        int datalen = __prod(shape, ndims, int);

        MTTensor *t = mt_alloc_empty_tensor(context);
        t->data     = mt_newptr(float, datalen);
        t->datalen  = datalen;
        t->ndims    = ndims;
        t->shape    = mt_newptr(int, ndims);
        mt_memcpy(t->data, data, datalen);
        mt_memcpy(t->shape, shape, ndims);
        __init_strides(t);
        __init_indices(t);
        return t;
}

MTTensor *mt_new_tensor_full(MTContext *ctx, float val,
                             int *shape, int ndims) {
        int    datalen = __prod(shape, ndims, int);
        float *data    = mt_newptr(float, datalen);
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

typedef struct {
        int **indices;
        int  *nextidx;
        int  *nextactidx;
        int  *shape;
        int   ndims;
        int   nxt;
        int   i;
} IdxIterator;

IdxIterator *mt_new_idxiterator(int **indices, int *shape, int ndims) {
        IdxIterator *it = mt_newptr(IdxIterator, 1);
        it->ndims       = ndims;
        it->nextidx     = mt_newptr(int, ndims);
        it->nextactidx  = mt_newptr(int, ndims);
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

// get the next multidimensional index from the iterator
int *mt_idxiterator_next(IdxIterator *it) {
        if (it->i++ < 0) return it->nextactidx;

        it->nxt = it->ndims - 1;
        // TODO: `if` instead of `while`?
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
        int **newindices = mt_newptr(int *, t->ndims);
        int  *newshape   = mt_newptr(int, t->ndims);
        for (int i = 0; i < t->ndims; i++) {
                if (i == dim) {
                        newshape[i]   = indexlen;
                        newindices[i] = index;
                } else {
                        newshape[i]   = t->shape[i];
                        newindices[i] = t->indices[i];
                }
        }

        int   newlen = __prod(newshape, t->ndims, int);
        float newdata[newlen];

        IdxIterator *it = mt_new_idxiterator(newindices, newshape, t->ndims);
        for (int i = 0; i < newlen; i++) {
                int *idxs  = mt_idxiterator_next(it);
                newdata[i] = mt_tensor_get(t, idxs, t->ndims);
        }
        MTTensor *newtensor = mt_new_tensor(ctx, newdata, newshape, t->ndims);
        mt_idxiterator_free(it), free(newshape), free(newindices);

        return newtensor;
}

void mt_squeeze_aspects_at_dim(int targetdim, int *shape, int *strides, int **indices, int ndims) {
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
                // null the index of node's parent's children that points
                // to this node
                int idxpchild = -1;
                if (t->parent != NULL)
                        idxpchild = __find_in_list(t->parent->deps, t,
                                                   t->parent->ndeps);
                if (idxpchild > -1) t->parent->deps[idxpchild] = NULL;

                // also null the index of node's tracker that points to this node
                int idxtracker = __find_in_list(t->context->tracked,
                                                t,
                                                t->context->ntracked);
                if (idxtracker > -1) t->context->tracked[idxtracker] = NULL;

                for (int i = 0; i < t->ndeps; i++)
                        if (t->deps[i] != NULL) mt_tensor_free(t->deps[i]);

                free(t->deps);
                free(t->data);
                free(t->shape);
                free(t->strides);
                __free_indices(t);
        }
        free(t);
}

// remove NULLs in the tracked list
void mt_context_defrag(MTContext *ctx) {
        MTTensor **newtracked = mt_newptr(MTTensor *, ctx->ntracked);

        int cnt = 0;
        for (int i = 0; i < ctx->ntracked; i++)
                if (ctx->tracked[i] != NULL) newtracked[cnt++] = ctx->tracked[i];
        // free the old `tracked` pointers
        free(ctx->tracked);
        ctx->ntracked = cnt;
        ctx->tracked  = newtracked;
        ctx->tracked  = realloc(ctx->tracked, ctx->cap * sizeof(*ctx->tracked));
}

void mt_free(MTContext *ctx) {
        for (int i = 0; i < ctx->ntracked; i++) {
                if (ctx->tracked[i] != NULL) mt_tensor_free(ctx->tracked[i]);
        }
        free(ctx->tracked);
        free(ctx);
}

MTContext *mt_new_context(void) {
        MTContext *ctx = mt_newptr(MTContext, 1);
        ctx->withgrads = CGM_OVERRIDE;
        ctx->ntracked  = 0;
        ctx->cap       = INITIAL_CAP;
        ctx->tracked   = mt_newptr(MTTensor *, INITIAL_CAP);
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

void mt_tensor_enable_grad(MTTensor *t) {
        t->req_grad = 1;
        mt_tensor_zero_grad(t);
}

void mt_tensor_disable_grad(MTTensor *t) {
        t->req_grad = 0;
        if (t->grad != NULL)
                t->grad = NULL;
}

void mt_tensor_backward(MTTensor *t, MTTensor *grad) {
        if (!t->req_grad)
                EXIT_WITH_ERROR("cannot perform backward on tensor not requiring grad");

        if (grad == NULL) {
                if (t->ndims == 0)
                        grad = mt_new_scalar(t->context, 1.0);
                else
                        EXIT_WITH_ERROR("grad must be specified for non scalar tensor");
        }
        t->grad = mt_tensor_add(t->grad, grad);

        for (int i = 0; i < t->ndeps; i++) {
                MTTensor *bwgrad = t->deps[i]->grad_fn(t->deps[i], grad);
                mt_tensor_backward(t->deps[i], bwgrad);
        }
}

void mt_tensor_zero_grad(MTTensor *t) {
        t->grad = mt_new_tensor_full(t->context, 0., t->shape, t->ndims);
}

void mt_tensor_print_debug(MTTensor *t) {
        printf("ndims : %d\n", t->ndims);
        printf("ndeps : %d\n", t->ndeps);
        printf("shape : "), __printarr(t->shape, t->ndims, "%d"), printf("\n");
        printf("data \n");
        printf("  - datalen : %ld\n", t->datalen);
        printf("  - content : "), __printarr(t->data, t->datalen, "%.2f");
        printf("\n");
        printf("\n");
}

int mt_is_tensor_eq(MTTensor *a, MTTensor *b) {
        if (a->ndims != b->ndims) return 0;
        return mt_arrsame(a->data, b->data, a->datalen) &&
               mt_arrsame(a->shape, b->shape, a->ndims);
}
