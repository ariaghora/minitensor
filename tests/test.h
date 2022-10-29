typedef struct {
        int npass;
        int nfail;
} Test;

#define new_test() ((Test){.nfail = 0, .npass = 0})
#define mt_assert_true(test, boolexp, test_desc, msg_if_wrong) ({                  \
        if (!(boolexp)) {                                                          \
                printf("\033[31m[FAIL]\033[0m %s: %s\n", test_desc, msg_if_wrong); \
                t->nfail++;                                                        \
        } else {                                                                   \
                printf("\033[32m[PASS]\033[0m %s\n", test_desc);                   \
                t->npass++;                                                        \
        }                                                                          \
})

/* testing tensor core functionality */
void run_tensor_creation_tests(Test *);
void run_tensor_slice_tests(Test *);
void run_tensor_access_tests(Test *);
void run_context_tests(Test *);

/* testing math functionality **/
void run_tensor_addition_tests(Test *);
void run_tensor_subtraction_tests(Test *);
void run_tensor_sum_tests(Test *t);

/* testing autograd engine **/
void run_simple_autograd_tests(Test *);
void run_autograd_backward_tests(Test *);
void run_tensor_reduce_tests(Test *t);