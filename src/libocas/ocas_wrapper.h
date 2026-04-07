/*
 * ocas_wrapper.h – context structs and high-level wrappers around the
 * svm_ocas_solver / msvm_ocas_solver callback-based API.
 *
 * Data layout contract (matches NumPy C-order [nData, nDim]):
 *   X[i, j]  lives at  X_ptr[i*nDim + j]
 *   LIBOCAS_INDEX(j, i, nDim) = i*nDim + j   ← same offset
 * so a C-contiguous [nData, nDim] array can be passed directly.
 *
 * Binary SVM  : labels data_y in {-1, +1}
 * Multi-class : labels data_y in {1, 2, ..., nY}  (1-indexed, as required by libocas)
 */

#ifndef OCAS_WRAPPER_H
#define OCAS_WRAPPER_H

#include <stdint.h>
#include "libocas.h"

/* ── Binary SVM context ─────────────────────────────────────────────────── */
typedef struct {
    double   *X;        /* [nData × nDim] C-order input features           */
    uint32_t  nDim;     /* number of input features                         */
    uint32_t  nData;    /* number of training examples                      */
    double   *data_y;   /* [nData] labels, must be {-1, +1}                 */
    double    X0;       /* 1.0 → fit intercept; 0.0 → no intercept          */

    /* Weight vector (maintained across callbacks) */
    double   *W;        /* [nDim] current weights                           */
    double   *oldW;     /* [nDim] weights at previous iterate               */
    double    W0;       /* current intercept                                */
    double    oldW0;    /* intercept at previous iterate                    */

    /* Cutting-plane buffer (pre-allocated to BufSize columns) */
    double   *new_a;    /* [nDim]         scratch vector for new cut        */
    double   *full_A;   /* [nDim × BufSize] column-major buffer             */
    double   *A0;       /* [BufSize]      intercept terms for each cut      */
} bin_ctx_t;


/* ── Multi-class SVM context ────────────────────────────────────────────── */
typedef struct {
    double   *X;        /* [nData × nDim] C-order input features           */
    uint32_t  nDim;     /* number of input features                         */
    uint32_t  nData;    /* number of training examples                      */
    uint32_t  nY;       /* number of classes                                */
    double   *data_y;   /* [nData] 1-indexed integer class labels           */

    /* Weight matrix: W[j + y*nDim] = weight for feature j, class y        */
    double   *W;        /* [nDim × nY] col-major  (current)                */
    double   *oldW;     /* [nDim × nY] col-major  (previous)               */

    /* Cutting-plane buffer */
    double   *new_a;    /* [nDim × nY] scratch for new cut                 */
    double   *full_A;   /* [nDim*nY × BufSize] col-major buffer            */
} msvm_ctx_t;


/* ── Entry-point functions (implemented in ocas_wrapper.c) ─────────────── */

ocas_return_value_T train_binary_ocas(
    bin_ctx_t *ctx,
    double     C,
    double     TolRel,
    double     TolAbs,
    double     QPBound,
    double     MaxTime,
    uint32_t   BufSize,
    uint8_t    Method
);

ocas_return_value_T train_msvm_ocas(
    msvm_ctx_t *ctx,
    double      C,
    double      TolRel,
    double      TolAbs,
    double      QPBound,
    double      MaxTime,
    uint32_t    BufSize,
    uint8_t     Method
);

#endif /* OCAS_WRAPPER_H */
