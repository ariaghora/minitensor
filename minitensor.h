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
 * UFunc: the float-valued unary function, alias for float(float)
 * function
 */
typedef float (*UFunc)(float);
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

/**
 * A tensor (less rigid definition from mathematical sense) is a container data
 * structure with any arbitrary dimensions. We can say that tensor is the
 * the generalization of scalar (0-dimensinal), vector (1-dimensional) and
 * matrix (2-dimensional).
 *
 * Minitensor tensors operate in row-major. The data is arranged linearly con-
 * tiguous in the memory. Minitensor includes reverse-mode autograd engine
 * which requires the construction of computational graph. A tensor also serves
 * as a node in a computational graph, that has references to its dependents
 * and a reference to its parent.
 */
struct MTTensor {
        /* where the actual numerical data is stored */
        float *data;
        /* Describing the number of elements in `data` */
        long datalen;
        /* Stores multidimensional indices for tensors. If a tensor has 2
         * dimensions, then there will be 2 integer arrays, each array has
         * values from 0, 1, ... until the shape of that dimension minus one. */
        int **indices;
        /* Indicating that this tensor is a leaf (1) or not (0). */
        int isleaf;
        /* Keeps track the number of dependent tensors (when gradient is requ-
         * ired)*/
        int ndeps;
        /* Number of tensor dimensions */
        int ndims;
        /* Indicating whether this tensor requires gradient computation (1) or
         * not (0). */
        int req_grad;
        /* Tracks the shape of a tensor, or the number of elements of every di-
         * mension. */
        int *shape;
        /* The stride, an array with length of `ndims`, each element describing
         * how much "jumps" need to be made to move into the next dimension. */
        int *strides;
        /* The reference to a context */
        MTContext *context;
        /* A list of tensors dependent on this tensor (in computational graph).
         */
        MTTensor **deps;
        MTTensor  *grad;
        MTTensor  *parent;
        /* Function to compute gradient during backward mode autograd*/
        TensorBackwardFunc grad_fn;
};

/**
 *  MTTensor main API
 */
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
MTTensor  *mt_tensor_ufunc(MTTensor *t, UFunc ufunc);
MTTensor  *mt_tensor_add(MTTensor *a, MTTensor *b);
MTTensor  *mt_tensor_sub(MTTensor *a, MTTensor *b);
MTTensor  *mt_tensor_mul(MTTensor *a, MTTensor *b);
MTTensor  *mt_tensor_div(MTTensor *a, MTTensor *b);
MTTensor  *mt_tensor_neg(MTTensor *t);
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

/**
 * Internal API
 */
typedef enum { BC_STATUS_NO_BCAST_REQUIRED,
               BC_STATUS_SKIP_SCALAR_HANDLING,
               BC_STATUS_SUCCESS,
               BC_STATUS_FAILURE } BcastStatus;
typedef struct BcastResult BcastResult;
struct BcastResult {
        MTTensor   *left;
        MTTensor   *right;
        BcastStatus status;
};

/**
 * Broadcast a tensor into one-another's shape. The result struct's left and
 * right attributes will be NULL if there is no broadcasting required or if
 * either left or right is a scalar.
 */
BcastResult mt_broadcast_lr(MTTensor *left, MTTensor *right);

/**
 * remove `targetdim` dimension if it is a singleton dimension. Otherwise,
 * the program will close with error. The shape output determination depends on
 * strides and indices arrays (as arguments). This function modifies shape,
 * strides, and indices arguments.
 */
void mt_squeeze_at_dim(int targetdim, int *shape, int *strides,
                       int **indices, int ndims);

/**
 * Remove NULL elements in the tracked atteribute of ctx and reallocate accordingly
 */
void mt_context_defrag(MTContext *ctx);

/**
 * Access the tensor data with customized indices, shape, strides, and ndims
 * constraints. This is useful for especially to access data of a tensor
 * without necessarily obeying contiguous order. For example, we can pass
 * swapped shape and swapped strides of a tensor into this function (and fix
 * the other variables) to get the tensor data in a transposed order.
 */
float *mt_tensor_get_all_data_constrained(MTTensor *t, int **indices,
                                          int *shape, int *strides, int ndims);

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