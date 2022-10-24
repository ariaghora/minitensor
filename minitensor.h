//
// A tensor manipulation library
//

#ifndef MINITENSOR_H_
#define MINITENSOR_H_

//
// =========================================================================
// MTTensor main API

typedef struct MTTensor  MTTensor;
typedef struct MTContext MTContext;

struct MTContext {
    int        withgrads;
    MTTensor **tracked;
    int        ntracked;
    int        cap;
};

struct MTTensor {
    float     *data;
    long       datalen;
    int      **indices;
    int        isleaf;
    int        ndeps;
    int        ndims;
    int        req_grad;
    int       *shape;
    int       *strides;
    MTContext *context;
    MTTensor **deps;
    MTTensor  *grad;
    MTTensor  *parent;
    MTTensor *(*grad_fn)(MTTensor *self, MTTensor *grad);
};

// BFunc: alias for float(float, float) function
typedef float *(*BFunc)(float, float);
// TensorBFunc: alias for MTTensor*(MTTensor*, MTTensor*) function
typedef MTTensor *(*TensorBFunc)(MTTensor *, MTTensor *);

MTTensor *mt_alloc_empty_tensor(MTContext *ctx);
MTTensor *mt_new_tensor(MTContext *context, float *data,
                        int *shape, int ndim);
MTTensor *mt_new_tensor_full(MTContext *context,
                             float val, int *shape,
                             int ndim);
float     mt_tensor_get(MTTensor *t, int *idx, int ndims);
float     mt_tensor_get_v(MTTensor *t);
float     mt_tensor_get_1(MTTensor *t, int i);
float     mt_tensor_get_2(MTTensor *t, int i, int j);
float     mt_tensor_get_3(MTTensor *t, int i, int j, int k);
MTTensor *mt_new_scalar(MTContext *context, float val);
MTTensor *mt_tensor_slice(MTContext *ctx, MTTensor *t, int dim,
                          int *index, int indexlen);
void      mt_tensor_free(MTTensor *t);

MTTensor *mt_tensor_bfunc(MTTensor *a, MTTensor *b, BFunc bfunc);
MTTensor *mt_tensor_add(MTTensor *a, MTTensor *b);
MTTensor *mt_tensor_sub(MTTensor *a, MTTensor *b);
MTTensor *mt_tensor_mul(MTTensor *a, MTTensor *b);
MTTensor *mt_tensor_div(MTTensor *a, MTTensor *b);
MTTensor *mt_tensor_reduce(MTTensor *t, int dim, TensorBFunc bfunc);
MTTensor *mt_tensor_sum(MTTensor *t, int dim);

MTContext *mt_new_context(void);
void       mt_context_push_tensor(MTContext *ctx, MTTensor *t);
void       mt_context_defrag(MTContext *ctx);
void       mt_context_free(MTContext *ctx);

void mt_tensor_enable_grad(MTTensor *t);
void mt_tensor_disable_grad(MTTensor *t);
void mt_tensor_backward(MTTensor *t, MTTensor *grad);
void mt_tensor_zero_grad(MTTensor *t);

int  mt_is_tensor_eq(MTTensor *a, MTTensor *b);
void assert_true(int boolexp, char *test_desc, char *msg_if_wrong);

//
// API shorthands
#ifndef MINITENSOR_DISABLE_SHORTHANDS_  // {{{

#define new_tensor(ctx, data, shape, ndim) \
    (mt_new_tensor(ctx, data, shape, ndim))
#define new_tensor_full(ctx, val, shape, ndim) \
    (mt_new_tensor_full(ctx, val, shape, ndim))
#define new_scalar(ctx, val) (mt_new_scalar(ctx, val))
#define new_tensor_1(data, shape) (mt_new_tensor(ctx, data, shape, 1))
#define new_tensor_2(data, shape) (mt_new_tensor(ctx, data, shape, 2))
#define new_tensor_3(data, shape) (mt_new_tensor(ctx, data, shape, 3))
#define tensor_get_2(i, j) ()
#define tensor_get_3(i, j, k) ()
#define tensor_set_2(i, j, val) ()
#define tensor_set_3(i, j, k, val) ()
#define tensor_free(t) (mt_tensor_free(t))

#define tensor_add(a, b) (mt_tensor_add(a, b))
#define tensor_sub(a, b) (mt_tensor_sub(a, b))
#define tensor_mul(a, b) (mt_tensor_mul(a, b))
#define tensor_div(a, b) (mt_tensor_div(a, b))
#define tensor_dot(a, b) (mt_tensor_dot(a, b))
#define tensor_sum(t, dim) (mt_tensor_sum(t, dim))

#define new_context() (mt_new_context())
#define tensor_enable_grad(t) (mt_tensor_enable_grad(t))
#define tensor_disable_grad(t) (mt_tensor_disable_grad(t))
#define tensor_backward(t, grad) (mt_tensor_backward(t, grad))
#define tensor_zero_grad(t) (mt_tensor_zero_grad(t))

#define mt_free(ctx) (mt_context_free(ctx))

// Testing helper tool
#define is_tensor_eq(t1, t2) (mt_is_tensor_eq(t1, t2))
#define arrsame(a, b, len) (mt_arrsame(a, b, len))

#endif  // }}} MINITENSOR_DISABLE_SHORTHANDS_

//
// =========================================================================
// The long version of the API, when shorthand is disallowed

// helper macros
#define newptr(type, len) ((type *)calloc((len), sizeof(type)))
#define mt_memcpy(to, from, len) (memcpy(to, from, (len) * sizeof(*from)))
#define EXIT_WITH_ERROR(msg) ({ \
    printf("error: %s\n", msg); \
    exit(1);                    \
})

#define mt_arrsame(a, b, len) ({                         \
    int __mt_issame = 1;                                 \
    for (long i = 0; i < len; i++)                       \
        __mt_issame = __mt_issame && ((a[i]) == (b[i])); \
    __mt_issame;                                         \
})

// array inline expression literal
#define Arr(type, ...) ((type[]){__VA_ARGS__})

#endif