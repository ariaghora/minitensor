/**
 * Minitensor -- A tensor manipulation library.
 *
 * Copyright (c) 2022, Aria Ghora Prabono <hello@ghora.net>
 * All rights reserved.
 */

#ifndef MINITENSOR_H_
#define MINITENSOR_H_

typedef struct MTTensor  MTTensor;
typedef struct MTContext MTContext;
typedef enum { CGM_REQUIRE_GRAD,
               CGM_NO_REQUIRE_GRAD,
               CGM_OVERRIDE } MtContextGradMode;
typedef enum { DEVICE_CPU,
               DEVICE_GPU } MtDevice;

/**
 * BFunc: the float-float binary function, alias for float(float, float)
 * function
 */
typedef float (*BFunc)(float, float);
/**
 * TensorBFunc: the tensor-tensor binary function, alias for
 * MTTensor*(MTTensor**, MTTensor*) function
 */
typedef MTTensor *(*TensorBFunc)(MTTensor *, MTTensor *);
/**
 * TensorBackwardFunc: the backward function type, alias for
 * MTTensor*(MTTensor**, MTTensor*) function
 */
typedef MTTensor *(*TensorBackwardFunc)(MTTensor **, MTTensor *);

/**
 * A context manages information of underlying allocated tensors it tracks.
 * It handles memory management in users' stead to avoid overly convoluted
 * APIs. Any tensor belongs to exactly one context. Two (or more) tensors must
 * operate under the same context.
 */
struct MTContext {
        /**
         * `withgrads` marks whether the tensors it track require gradients
         * (context-wide), regardless that req_grad is specified individually.
         * It defaults to CGM_OVERRIDE, that respects individual tensor's grad
         * requirement.
         */
        MtContextGradMode withgrads;
        /* Points to the list of tracked tensors */
        MTTensor **tracked;
        /* Count of the tracked tensors */
        int ntracked;
        /* Current max capacity of tracked tensors. It may grow as needed. */
        int cap;
        /* The device where tensor data is allocated: CPU or GPU */
        MtDevice device;
};

struct MTTensor {
        float             *data;
        long               datalen;
        int              **indices;
        int                isleaf;
        int                ndeps;
        int                ndims;
        int                req_grad;
        int               *shape;
        int               *strides;
        MTContext         *context;
        MTTensor         **deps;
        MTTensor          *grad;
        MTTensor          *parent;
        TensorBackwardFunc grad_fn;
};

/* MTTensor main API */
MTTensor  *mt_alloc_empty_tensor(MTContext *ctx);
MTTensor  *mt_new_tensor(MTContext *context, float *data,
                         int *shape, int ndim);
MTTensor  *mt_new_tensor_full(MTContext *context,
                              float val, int *shape,
                              int ndim);
float      mt_tensor_get(MTTensor *t, int *idx, int ndims);
float      mt_tensor_get_v(MTTensor *t);
float      mt_tensor_get_1(MTTensor *t, int i);
float      mt_tensor_get_2(MTTensor *t, int i, int j);
float      mt_tensor_get_3(MTTensor *t, int i, int j, int k);
MTTensor  *mt_new_scalar(MTContext *context, float val);
MTTensor  *mt_tensor_slice(MTContext *ctx, MTTensor *t, int dim,
                           int *index, int indexlen);
MTTensor  *mt_tensor_sum(MTTensor *t, int dim, int keepdims);
void       mt_tensor_free(MTTensor *t);
MTTensor  *mt_tensor_bfunc(MTTensor *a, MTTensor *b, BFunc bfunc);
MTTensor  *mt_tensor_add(MTTensor *a, MTTensor *b);
MTTensor  *mt_tensor_sub(MTTensor *a, MTTensor *b);
MTTensor  *mt_tensor_mul(MTTensor *a, MTTensor *b);
MTTensor  *mt_tensor_div(MTTensor *a, MTTensor *b);
MTTensor  *mt_tensor_reduce(MTTensor *t, int dim, TensorBFunc bfunc,
                            int keepdims);
MTContext *mt_new_context(void);
void       mt_context_push_tensor(MTContext *ctx, MTTensor *t);
void       mt_free(MTContext *ctx);
void       mt_tensor_enable_grad(MTTensor *t);
void       mt_tensor_disable_grad(MTTensor *t);
void       mt_tensor_backward(MTTensor *t, MTTensor *grad);
void       mt_tensor_zero_grad(MTTensor *t);
int        mt_is_tensor_eq(MTTensor *a, MTTensor *b);
void       mt_tensor_print_debug(MTTensor *t);

/* Internal API */
void mt_squeeze_aspects_at_dim(int targetdim, int *shape, int *strides,
                               int **indices, int ndims);
void mt_context_defrag(MTContext *ctx);

/* helper macros */
#define mt_newptr(type, len) ((type *)calloc((len), sizeof(type)))
#define mt_memcpy(to, from, len) (memcpy(to, from, (len) * sizeof(*from)))
#define mt_arrsame(a, b, len) ({                                 \
        int __mt_issame = 1;                                     \
        for (long i = 0; i < len; i++)                           \
                __mt_issame = __mt_issame && ((a[i]) == (b[i])); \
        __mt_issame;                                             \
})
/* array inline expression literal */
#define Arr(type, ...) ((type[]){__VA_ARGS__})
#define EXIT_WITH_ERROR(msg) ({     \
        printf("error: %s\n", msg); \
        exit(1);                    \
})

#endif