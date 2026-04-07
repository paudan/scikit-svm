/*
 * ocas_wrapper.c – callback implementations and high-level training
 * wrappers for svm_ocas_solver (binary) and msvm_ocas_solver (multi-class).
 *
 * All callbacks operate on the appropriate context struct passed via the
 * void *user_data pointer, so no global variables are needed.
 */

#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <stdio.h>
#include <stdint.h>

#include "libocas.h"
#include "ocas_wrapper.h"

/* ── helpers ────────────────────────────────────────────────────────────── */

/* Pair (value, data) used by the sort callback. */
typedef struct { double v; double d; } vd_pair_t;

static int _cmp_asc(const void *a, const void *b)
{
    double va = ((const vd_pair_t *)a)->v;
    double vb = ((const vd_pair_t *)b)->v;
    if (va < vb) return -1;
    if (va > vb) return  1;
    return 0;
}

/* Sort value[] ascending, permuting data[] consistently.
 * Matches the original libocas qsort_data() sort order, which is ascending. */
static int _qsort_data(double *value, double *data, uint32_t size)
{
    uint32_t i;
    vd_pair_t *pairs = (vd_pair_t *)malloc(size * sizeof(vd_pair_t));
    if (!pairs) return -1;
    for (i = 0; i < size; i++) { pairs[i].v = value[i]; pairs[i].d = data[i]; }
    qsort(pairs, size, sizeof(vd_pair_t), _cmp_asc);
    for (i = 0; i < size; i++) { value[i] = pairs[i].v; data[i] = pairs[i].d; }
    free(pairs);
    return 0;
}

static void _print_null(ocas_return_value_T val) { (void)val; }


/* ═══════════════════════════════════════════════════════════════════════════
 * Binary SVM callbacks
 * ═══════════════════════════════════════════════════════════════════════════
 *
 * LIBOCAS_INDEX(ROW, COL, NUM_ROWS) = COL*NUM_ROWS + ROW
 *   full_A  is  [nDim × BufSize]  col-major → full_A[j + i*nDim]
 *   X       is  [nData × nDim]  C-order    → X[row + col*nDim] via that macro
 *                                           = X_np[row, col]
 */

/* compute_W:
 *   oldW = W
 *   W    = full_A[:,0:nSel] @ alpha[0:nSel]
 *   sq_norm_W = ||W||² + W0²
 *   dp_WoldW  = W·oldW + W0·oldW0
 */
static void _bin_compute_W(double *sq_norm_W, double *dp_WoldW,
                           double *alpha, uint32_t nSel, void *user_data)
{
    bin_ctx_t *ctx = (bin_ctx_t *)user_data;
    uint32_t i, j;
    double snw, dpw;

    memcpy(ctx->oldW, ctx->W, ctx->nDim * sizeof(double));
    memset(ctx->W,    0,      ctx->nDim * sizeof(double));
    ctx->oldW0 = ctx->W0;
    ctx->W0    = 0.0;

    for (i = 0; i < nSel; i++) {
        if (alpha[i] > 0.0) {
            for (j = 0; j < ctx->nDim; j++)
                ctx->W[j] += alpha[i] * ctx->full_A[LIBOCAS_INDEX(j, i, ctx->nDim)];
            ctx->W0 += ctx->A0[i] * alpha[i];
        }
    }

    snw = ctx->W0 * ctx->W0;
    dpw = ctx->W0 * ctx->oldW0;
    for (j = 0; j < ctx->nDim; j++) {
        snw += ctx->W[j] * ctx->W[j];
        dpw += ctx->W[j] * ctx->oldW[j];
    }
    *sq_norm_W = snw;
    *dp_WoldW  = dpw;
}

/* update_W:
 *   W  = (1-t)*oldW  + t*W
 *   W0 = (1-t)*oldW0 + t*W0
 *   returns ||W||² + W0²
 */
static double _bin_update_W(double t, void *user_data)
{
    bin_ctx_t *ctx = (bin_ctx_t *)user_data;
    uint32_t j;
    double snw;

    ctx->W0 = ctx->oldW0 * (1.0 - t) + t * ctx->W0;
    snw = ctx->W0 * ctx->W0;
    for (j = 0; j < ctx->nDim; j++) {
        ctx->W[j] = ctx->oldW[j] * (1.0 - t) + t * ctx->W[j];
        snw += ctx->W[j] * ctx->W[j];
    }
    return snw;
}

/* add_new_cut (dense):
 *   new_a   = sum_{i in new_cut} X[new_cut[i], :]
 *   A0[nSel]= X0 * sum_{i in new_cut} y[new_cut[i]]
 *   full_A[:, nSel] = new_a
 *   new_col_H[nSel] = ||new_a||² + A0[nSel]²
 *   new_col_H[i]    = full_A[:,i]·new_a + A0[i]*A0[nSel],  i < nSel
 */
static int _bin_add_new_cut(double *new_col_H, uint32_t *new_cut,
                             uint32_t cut_length, uint32_t nSel,
                             void *user_data)
{
    bin_ctx_t *ctx = (bin_ctx_t *)user_data;
    uint32_t i, j;
    double sq_norm_a;

    memset(ctx->new_a, 0, ctx->nDim * sizeof(double));
    ctx->A0[nSel] = 0.0;

    for (i = 0; i < cut_length; i++) {
        uint32_t idx = new_cut[i];
        for (j = 0; j < ctx->nDim; j++)
            ctx->new_a[j] += ctx->X[LIBOCAS_INDEX(j, idx, ctx->nDim)];
        ctx->A0[nSel] += ctx->X0 * ctx->data_y[idx];
    }

    sq_norm_a = ctx->A0[nSel] * ctx->A0[nSel];
    for (j = 0; j < ctx->nDim; j++) {
        sq_norm_a += ctx->new_a[j] * ctx->new_a[j];
        ctx->full_A[LIBOCAS_INDEX(j, nSel, ctx->nDim)] = ctx->new_a[j];
    }

    new_col_H[nSel] = sq_norm_a;
    for (i = 0; i < nSel; i++) {
        double tmp = ctx->A0[nSel] * ctx->A0[i];
        for (j = 0; j < ctx->nDim; j++)
            tmp += ctx->new_a[j] * ctx->full_A[LIBOCAS_INDEX(j, i, ctx->nDim)];
        new_col_H[i] = tmp;
    }
    return 0;
}

/* compute_output:
 *   output[i] = y[i]*X0*W0 + W·X[i,:]
 */
static int _bin_compute_output(double *output, void *user_data)
{
    bin_ctx_t *ctx = (bin_ctx_t *)user_data;
    uint32_t i, j;
    double tmp;

    for (i = 0; i < ctx->nData; i++) {
        tmp = ctx->data_y[i] * ctx->X0 * ctx->W0;
        for (j = 0; j < ctx->nDim; j++)
            tmp += ctx->W[j] * ctx->X[LIBOCAS_INDEX(j, i, ctx->nDim)];
        output[i] = tmp;
    }
    return 0;
}


/* ═══════════════════════════════════════════════════════════════════════════
 * Multi-class SVM callbacks
 * ═══════════════════════════════════════════════════════════════════════════
 *
 * W[j + y*nDim]  weight for feature j, class y
 * full_A: [nDim*nY × BufSize] col-major
 * output: [nY × nData] col-major  → output[y + i*nY]
 */

/* compute_W:
 *   oldW = W
 *   W    = full_A[:,0:nSel] @ alpha[0:nSel]
 */
static void _msvm_compute_W(double *sq_norm_W, double *dp_WoldW,
                             double *alpha, uint32_t nSel, void *user_data)
{
    msvm_ctx_t *ctx = (msvm_ctx_t *)user_data;
    uint32_t i, j;
    uint32_t total = ctx->nDim * ctx->nY;
    double snw = 0.0, dpw = 0.0;

    memcpy(ctx->oldW, ctx->W, total * sizeof(double));
    memset(ctx->W,    0,      total * sizeof(double));

    for (i = 0; i < nSel; i++) {
        if (alpha[i] > 0.0) {
            for (j = 0; j < total; j++)
                ctx->W[j] += alpha[i] * ctx->full_A[LIBOCAS_INDEX(j, i, total)];
        }
    }

    for (j = 0; j < total; j++) {
        snw += ctx->W[j] * ctx->W[j];
        dpw += ctx->W[j] * ctx->oldW[j];
    }
    *sq_norm_W = snw;
    *dp_WoldW  = dpw;
}

/* update_W:
 *   W = (1-t)*oldW + t*W
 *   returns ||W||²
 */
static double _msvm_update_W(double t, void *user_data)
{
    msvm_ctx_t *ctx = (msvm_ctx_t *)user_data;
    uint32_t j;
    double snw = 0.0;
    uint32_t total = ctx->nDim * ctx->nY;

    for (j = 0; j < total; j++) {
        ctx->W[j] = ctx->oldW[j] * (1.0 - t) + t * ctx->W[j];
        snw += ctx->W[j] * ctx->W[j];
    }
    return snw;
}

/* add_new_cut (dense):
 *   new_cut[i] = 0-indexed predicted class for sample i
 *   data_y[i]  = 1-indexed true class
 *
 *   new_a[:,y_true]  += X[i, :]   if new_cut[i] != y_true
 *   new_a[:,y_pred]  -= X[i, :]   if new_cut[i] != y_true
 */
static int _msvm_add_new_cut(double *new_col_H, uint32_t *new_cut,
                              uint32_t nSel, void *user_data)
{
    msvm_ctx_t *ctx = (msvm_ctx_t *)user_data;
    uint32_t i, j;
    uint32_t total = ctx->nDim * ctx->nY;
    double sq_norm_a;

    memset(ctx->new_a, 0, total * sizeof(double));

    for (i = 0; i < ctx->nData; i++) {
        uint32_t y_true = (uint32_t)(ctx->data_y[i] - 1);  /* 0-indexed true class */
        uint32_t y_pred = new_cut[i];                       /* 0-indexed predicted  */
        if (y_pred != y_true) {
            for (j = 0; j < ctx->nDim; j++) {
                ctx->new_a[LIBOCAS_INDEX(j, y_true, ctx->nDim)] +=
                    ctx->X[LIBOCAS_INDEX(j, i, ctx->nDim)];
                ctx->new_a[LIBOCAS_INDEX(j, y_pred, ctx->nDim)] -=
                    ctx->X[LIBOCAS_INDEX(j, i, ctx->nDim)];
            }
        }
    }

    sq_norm_a = 0.0;
    for (j = 0; j < total; j++) {
        sq_norm_a += ctx->new_a[j] * ctx->new_a[j];
        ctx->full_A[LIBOCAS_INDEX(j, nSel, total)] = ctx->new_a[j];
    }

    new_col_H[nSel] = sq_norm_a;
    for (i = 0; i < nSel; i++) {
        double tmp = 0.0;
        for (j = 0; j < total; j++)
            tmp += ctx->new_a[j] * ctx->full_A[LIBOCAS_INDEX(j, i, total)];
        new_col_H[i] = tmp;
    }
    return 0;
}

/* compute_output:
 *   output[y + i*nY] = W[y*nDim : (y+1)*nDim] · X[i, :]
 */
static int _msvm_compute_output(double *output, void *user_data)
{
    msvm_ctx_t *ctx = (msvm_ctx_t *)user_data;
    uint32_t i, j, y;

    for (i = 0; i < ctx->nData; i++) {
        for (y = 0; y < ctx->nY; y++) {
            double tmp = 0.0;
            for (j = 0; j < ctx->nDim; j++)
                tmp += ctx->W[LIBOCAS_INDEX(j, y, ctx->nDim)]
                     * ctx->X[LIBOCAS_INDEX(j, i, ctx->nDim)];
            output[LIBOCAS_INDEX(y, i, ctx->nY)] = tmp;
        }
    }
    return 0;
}


/* ═══════════════════════════════════════════════════════════════════════════
 * High-level training entry points
 * ═══════════════════════════════════════════════════════════════════════════ */

ocas_return_value_T train_binary_ocas(
    bin_ctx_t *ctx,
    double C, double TolRel, double TolAbs,
    double QPBound, double MaxTime,
    uint32_t BufSize, uint8_t Method)
{
    return svm_ocas_solver(
        C, ctx->nData, TolRel, TolAbs, QPBound, MaxTime, BufSize, Method,
        _bin_compute_W,
        _bin_update_W,
        _bin_add_new_cut,
        _bin_compute_output,
        _qsort_data,
        _print_null,
        (void *)ctx);
}

ocas_return_value_T train_msvm_ocas(
    msvm_ctx_t *ctx,
    double C, double TolRel, double TolAbs,
    double QPBound, double MaxTime,
    uint32_t BufSize, uint8_t Method)
{
    return msvm_ocas_solver(
        C, ctx->data_y, ctx->nY, ctx->nData,
        TolRel, TolAbs, QPBound, MaxTime, BufSize, Method,
        _msvm_compute_W,
        _msvm_update_W,
        _msvm_add_new_cut,
        _msvm_compute_output,
        _qsort_data,
        _print_null,
        (void *)ctx);
}
