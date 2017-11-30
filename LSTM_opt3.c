#include <cstddef>
#include <iostream>
#include <mkl.h>
#include <omp.h>
#include <stdlib.h>
#include <math.h>
#include <string.h>
#define max(a,b) ((a) > (b) ? (a) : (b))

#include <stdio.h>
#include <stdlib.h>
#include "mkl.h"
#include<sys/time.h>
#include <stdbool.h>
#include "helper_string.h"

//#define hDebug  //  
#define hDebug  printf  
//#define pGEMM  //  
//#define pGEMM  printf  

int loops = 100;
int batch_size = 64;
int time_step = 10;
int input_dim = 150;
int hid = 1024;

void  LSTM_forward(int batch_size, int time_step, int input_dim, int hid, 
                      const float* w_x, const float* w_h, const float* b, const float* x, const float* h_0, const float* c_0, 
               /*out*/float *o_t, float *f_t, float *i_t, float* c_wave_t, //hid * batch_size
                      float* c_t, float* h,//time_Step * hid * batch_size
            /*global*/const float** A, const float** B, float** C, float* x_temp){
    ////global
    //const float** A = (const float**)mkl_malloc(4 * time_step * sizeof (float*), 64);
    //const float** B = (const float**)mkl_malloc(4 * time_step * sizeof (float*), 64);
    //float** C = (float**)mkl_malloc(4 * time_step * sizeof (float*), 64);
    //float* x_temp = (float*)mkl_malloc(time_step * 4 * batch_size * hid * sizeof (float), 64);
    
    memset(x_temp, 0, sizeof(float) * (time_step * 4 * batch_size * hid));
    int i,j,p;
    // w_x * x
    MKL_INT m[1]; 
    MKL_INT n[1]; 
    MKL_INT k[1]; 
    
    MKL_INT lda[1]; 
    MKL_INT ldb[1]; 
    MKL_INT ldc[1]; 
    
    CBLAS_TRANSPOSE transA[1]; 
    CBLAS_TRANSPOSE transB[1]; 
    
    float alpha[1]; 
    float beta[1]; 
    MKL_INT size_per_grp[1]; 

    m[0] = hid;
    k[0] = input_dim;
    n[0] = batch_size;
    
    lda[0] = k[0]; 
    ldb[0] = n[0]; 
    ldc[0] = n[0]; 
    
    transB[0] = CblasNoTrans; 
    transA[0] = CblasNoTrans; 
    
    alpha[0] = 1.0; 
    if (b == NULL) {
        beta[0] = 0.0;
    }
    else {
        beta[0] = 1.0;
        #pragma omp parallel for 
        for (i = 0; i < time_step; i++) { 
            for (j = 0; j < hid; j++) { 
                for (p = 0; p < batch_size; p++) { 
                    size_t offset0 = i * batch_size * hid + j * batch_size + p; 
                    size_t offset1 = (i + time_step) * batch_size * hid + j * batch_size + p; 
                    size_t offset2 = (i + 2 * time_step) * batch_size * hid + j * batch_size + p; 
                    size_t offset3 = (i + 3 * time_step) * batch_size * hid + j * batch_size + p; 
        
                    x_temp[offset0] = b[j]; 
                    x_temp[offset1] = b[j + hid]; 
                    x_temp[offset2] = b[j + 2 * hid]; 
                    x_temp[offset3] = b[j + 3 * hid]; 
                } 
            } 
        } 
    }
    size_per_grp[0] = 4 * time_step;

    if (NULL == A || NULL == B || NULL == C || NULL == x_temp) {
        printf( "\n ERROR: malloc global buffers failed \n\n");
        return;
    }
    #pragma omp parallel for 
    for (i = 0; i < time_step; i++) { 
        A[i] = w_x;                                       // w_ix
        A[i + time_step] = w_x + input_dim * hid;         // w_fx
        A[i + 2 * time_step] = w_x + 2 * input_dim * hid; // w_cx 
        A[i + 3 * time_step] = w_x + 3 * input_dim * hid; // w_ox 
    
        B[i] = x + i * k[0] * n[0]; 
        B[i + time_step] = B[i]; 
        B[i + 2 * time_step] = B[i]; 
        B[i + 3 * time_step] = B[i]; 
    
        C[i] = x_temp + i * m[0] * n[0]; 
        C[i + time_step] = x_temp + (i + time_step) * m[0] * n[0]; 
        C[i + 2 * time_step] = x_temp + (i + 2 * time_step) * m[0] * n[0]; 
        C[i + 3 * time_step] = x_temp + (i + 3 * time_step) * m[0] * n[0]; 
    } 
    cblas_sgemm_batch(CblasRowMajor, transA, transB, m, n, k, alpha, A, lda, B, ldb, beta, C, ldc, 1, size_per_grp); 

    // loop on step
    m[0] = hid;
    k[0] = hid;
    n[0] = batch_size;
    
    beta[0] = 1.0;

    lda[0] = k[0]; 
    ldb[0] = n[0]; 
    ldc[0] = n[0]; 
    size_per_grp[0] = 4;
    
    A[0] = w_h;                //w_ih
    A[1] = w_h + hid * hid;    //w_fh
    A[2] = w_h + 2 * hid * hid;//w_ch
    A[3] = w_h + 3 * hid * hid;//w_oh
    
    B[0] = h_0;
    B[1] = h_0;
    B[2] = h_0;
    B[3] = h_0;

    size_t mn = m[0] * n[0];
    #pragma omp parallel for
    for (j = 0; j < mn; j++) {
        c_t[j] = c_0[j];
    }

    float* c_t_ptr = NULL;
    float* f_t_ptr = NULL;
    float* i_t_ptr = NULL;
    float* c_wave_t_ptr = NULL;
    float* o_t_ptr = NULL;
    for (i = 0; i < time_step; i++) {
        // f,i,c_wave,o
        C[0] = x_temp + i * m[0] * n[0];
        C[1] = x_temp + (i + time_step) * m[0] * n[0];
        C[2] = x_temp + (i + 2 * time_step) * m[0] * n[0];
        C[3] = x_temp + (i + 3 * time_step) * m[0] * n[0];

        cblas_sgemm_batch(CblasRowMajor, transA, transB, m, n, k, alpha, A, lda, B, ldb, beta, C, ldc, 1, size_per_grp);

        // sigmoid for f,i,o, tanh for c_wave
        c_t_ptr = c_t + i * mn;
        f_t_ptr = f_t + i * mn;
        i_t_ptr = i_t + i * mn;
        o_t_ptr = o_t + i * mn;
        c_t_ptr = c_t + i * mn;
        c_wave_t_ptr = c_wave_t + i * mn;
        #pragma omp parallel for
        for (j = 0; j < mn; j++) {
            float exp_i = exp((float)(C[0][j]));
            float exp_f = exp((float)(C[1][j]));
            c_wave_t_ptr[j] = tanh((float)(C[2][j]));
            float exp_o = exp((float)(C[3][j]));
            f_t_ptr[j] = exp_f / ((float)1.0 + exp_f);        
            i_t_ptr[j] = exp_i / ((float)1.0 + exp_i);
            o_t_ptr[j] = exp_o / ((float)1.0 + exp_o);
        }
        //c
        const float* c_tm1 = NULL;
        if(i == 0) 
            c_tm1 = c_0;
        else
            c_tm1 = c_t_ptr - mn;
        #pragma omp parallel for 
        for (j = 0; j < mn; j++) { 
            c_t_ptr[j] = (float)((float)(f_t_ptr[j]) * (float)(c_tm1[j]) + (float)(i_t_ptr[j]) * (float)(c_wave_t_ptr[j])); 
        }
        float* y_ptr = NULL;
        y_ptr = h + i * mn;
        //h:all time_step
        #pragma omp parallel for
        for (j = 0; j < mn; j++) {
            y_ptr[j] = (float)(o_t_ptr[j]) * tanh((float)(c_t_ptr[j]));
        }
        // update
        B[0] = y_ptr;
        B[1] = B[0];
        B[2] = B[0];
        B[3] = B[0];
    }
    //mkl_free(A);
    //mkl_free(B);
    //mkl_free(C);
    //mkl_free(x_temp);
}

void  LSTM_backward(int batch_size, int time_step, int input_dim, int hid, 
                    const float* w_x, const float* w_h, const float* b, const float* x, const float* h_0, const float* c_0,//same with forward input
                    float *ho, float *hf, float *hi, float* hc, float* c, float* h,//forward output: time_step * hid * batch_size
                    float* grad_last,//last gradient
             /*out*/float* dwix, float* dwfx, float* dwcx, float* dwox,
                    float* dwih, float* dwfh, float* dwch, float* dwoh, 
                    float* dbi, float* dbf, float* dbc, float* dbo, 
                    float* dx,
                    const float** A, const float** B, float** C,  float* dh, float* dc, float* dh_next, float* dc_next, float* dhf, float* dhi, float* dhc, float* dho, float* x_temp){
    //global
    //const float** A = (const float**)mkl_malloc(4 * time_step * sizeof (float*), 64);
    //const float** B = (const float**)mkl_malloc(4 * time_step * sizeof (float*), 64);
    //float** C = (float**)mkl_malloc(4 * time_step * sizeof (float*), 64);
    //memset(dwfx, 0, sizeof(float) * hid * input_dim);
    //memset(dwix, 0, sizeof(float) * hid * input_dim);
    //memset(dwcx, 0, sizeof(float) * hid * input_dim);
    //memset(dwox, 0, sizeof(float) * hid * input_dim);
    //memset(dwfh, 0, sizeof(float) * hid * hid);
    //memset(dwih, 0, sizeof(float) * hid * hid);
    //memset(dwch, 0, sizeof(float) * hid * hid);
    //memset(dwoh, 0, sizeof(float) * hid * hid);
    //memset(dbf, 0, sizeof(float) * hid);
    //memset(dbi, 0, sizeof(float) * hid);
    //memset(dbc, 0, sizeof(float) * hid);
    //memset(dbo, 0, sizeof(float) * hid);
    
    int i,j,p;
    MKL_INT m[1]; 
    MKL_INT n[1]; 
    MKL_INT k[1]; 
    
    MKL_INT lda[1]; 
    MKL_INT ldb[1]; 
    MKL_INT ldc[1]; 
    
    CBLAS_TRANSPOSE transA[1]; 
    CBLAS_TRANSPOSE transB[1]; 
    
    float alpha[1]; 
    float beta[1]; 
    MKL_INT size_per_grp[1]; 

    //from last timestep
    memset(dh_next, 0, sizeof(float) * hid * batch_size);
    memset(dc_next, 0, sizeof(float) * hid * batch_size);
    
    //cache: hf hi hc ho c, c=[c_0, all c_t]i
    //calculate all gf, gi, gc_wave, go
    // loop on step
    m[0] = hid;
    k[0] = hid;
    n[0] = batch_size;
    
    beta[0] = 0.0;

    lda[0] = m[0]; 
    ldb[0] = n[0]; 
    ldc[0] = n[0]; 
    transA[0] = CblasTrans; 
    transB[0] = CblasNoTrans; 
    
    alpha[0] = 1.0; 
    size_per_grp[0] = 4;
    
    A[0] = w_h;                //w_ih
    A[1] = w_h + hid * hid;    //w_fh
    A[2] = w_h + 2 * hid * hid;//w_ch
    A[3] = w_h + 3 * hid * hid;//w_oh
    
    size_t bh = batch_size * hid;
    size_t tbi = batch_size * input_dim * time_step;
    size_t ib = input_dim * batch_size;
    size_t hh = hid * hid;
    C[0] = x_temp;
    C[1] = x_temp + bh;
    C[2] = x_temp + bh * 2;
    C[3] = x_temp + bh * 3;
    float* c_ptr = NULL;
    float* hf_ptr = NULL;
    float* hi_ptr = NULL;
    float* hc_ptr = NULL;
    float* ho_ptr = NULL;
    float* dhf_ptr = NULL;
    float* dhi_ptr = NULL;
    float* dhc_ptr = NULL;
    float* dho_ptr = NULL;

    for(i = time_step - 1; i >= 0; i--) {
        int kk = i * bh;
        c_ptr = c + kk;
        hf_ptr = hf + kk;
        hi_ptr = hi + kk;
        hc_ptr = hc + kk;
        ho_ptr = ho + kk;
        dhf_ptr = dhf + kk;
        dhi_ptr = dhi + kk;
        dhc_ptr = dhc + kk;
        dho_ptr = dho + kk;
        
        const float *c_old;
        if(i != 0)
            c_old = c_ptr - bh;
        else
            c_old = c_0;
        if(i == time_step - 1) {
            #pragma omp parallel for
            for(j = 0; j < bh; j++ ) {
                float tanh_c = tanh(c_ptr[j]);
                //dh[j] = 1.0 + dh_next[j];
                dh[j] = grad_last[j] + dh_next[j];
                dho_ptr[j] = ho_ptr[j] * (1.0 - ho_ptr[j]) * tanh_c * dh[j];
                dc[j] = ho_ptr[j] * dh[j] * (1.0 - tanh_c * tanh_c) + dc_next[j];
                dhf_ptr[j] = hf_ptr[j] * (1.0 - hf_ptr[j]) * c_old[j] * dc[j];
                dhi_ptr[j] = hi_ptr[j] * (1.0 - hi_ptr[j]) * hc_ptr[j] * dc[j];
                dhc_ptr[j] = (1.0 - hc_ptr[j] * hc_ptr[j]) * hi_ptr[j] * dc[j];
            }
        }
        else {
            #pragma omp parallel for
            for(j = 0; j < bh; j++ ) {
                float tanh_c = tanh(c_ptr[j]);
                dh[j] = dh_next[j];
                dho_ptr[j] = ho_ptr[j] * (1.0 - ho_ptr[j]) * tanh_c * dh[j];
                dc[j] = ho_ptr[j] * dh[j] * (1.0 - tanh_c * tanh_c) + dc_next[j];
                dhf_ptr[j] = hf_ptr[j] * (1.0 - hf_ptr[j]) * c_old[j] * dc[j];
                dhi_ptr[j] = hi_ptr[j] * (1.0 - hi_ptr[j]) * hc_ptr[j] * dc[j];
                dhc_ptr[j] = (1.0 - hc_ptr[j] * hc_ptr[j]) * hi_ptr[j] * dc[j];
            }
        }
        B[0] = dhi_ptr;
        B[1] = dhf_ptr;
        B[2] = dhc_ptr;
        B[3] = dho_ptr;
        cblas_sgemm_batch(CblasRowMajor, transA, transB, m, n, k, alpha, A, lda, B, ldb, beta, C, ldc, 1, size_per_grp);
    
        //calculate dbf, dbi, dbc, dbo
        #pragma omp parallel for
        for(j = 0; j < bh; j++ ) {
            dh_next[j] = C[0][j] + C[1][j] + C[2][j] + C[3][j];
            dc_next[j] = hf_ptr[j] * dc[j];
        }
        #pragma omp parallel for
        for(j = 0; j < hid; j++) {
            for(p = 0; p < batch_size; p++) {
                int index = j * batch_size + p;
                dbf[j] += dhf_ptr[index];
                dbi[j] += dhi_ptr[index];
                dbc[j] += dhc_ptr[index];
                dbo[j] += dho_ptr[index];
            }
        } 
    }
    //calculate dwfx, dwix, dwcx, dwox
    m[0] = hid;
    k[0] = batch_size;
    n[0] = input_dim;

    lda[0] = k[0];
    ldb[0] = k[0];
    ldc[0] = n[0];
    transA[0] = CblasNoTrans;
    transB[0] = CblasTrans;
    
    size_per_grp[0] = 4 * time_step;
    for (i = 0; i < time_step; i++) {
        A[i] = dhf + i * bh;                 
        A[i + time_step] = dhi + i * bh;    
        A[i + 2 * time_step] = dhc + i * bh; 
        A[i + 3 * time_step] = dho + i * bh; 
    
        B[i] = x + i * input_dim * batch_size; 
        B[i + time_step] = B[i]; 
        B[i + 2 * time_step] = B[i]; 
        B[i + 3 * time_step] = B[i]; 
    
        C[i] = x_temp + i * hid * input_dim;
        C[i + time_step] = x_temp + (i + time_step) * hid * input_dim; 
        C[i + 2 * time_step] = x_temp + (i + 2 * time_step) * hid * input_dim; 
        C[i + 3 * time_step] = x_temp + (i + 3 * time_step) * hid * input_dim; 
    }
    cblas_sgemm_batch(CblasRowMajor, transA, transB, m, n, k, alpha, A, lda, B, ldb, beta, C, ldc, 1, size_per_grp);
    #pragma omp parallel for
    for(i = 0; i < hid * input_dim; i++) {
        for(j = 0; j < time_step; j++) {
            dwfx[i] += C[j][i];
            dwix[i] += C[j + time_step][i];
            dwcx[i] += C[j + 2 * time_step][i];
            dwox[i] += C[j + 3 * time_step][i];
        }
    }
    //calculate dwfh, dwih, dwch, dwoh
    m[0] = hid;
    k[0] = batch_size;
    n[0] = hid;

    lda[0] = k[0];
    ldb[0] = k[0];
    ldc[0] = n[0];
    transA[0] = CblasNoTrans;
    transB[0] = CblasTrans;
    
    size_per_grp[0] = 4 * time_step;
    for (i = 0; i < time_step; i++) {
        A[i] = dhf + i * bh;                 
        A[i + time_step] = dhi + i * bh;    
        A[i + 2 * time_step] = dhc + i * bh; 
        A[i + 3 * time_step] = dho + i * bh; 
   
        if(i == 0) {
            B[i] = h_0; 
            B[i + time_step] = B[i]; 
            B[i + 2 * time_step] = B[i]; 
            B[i + 3 * time_step] = B[i]; 
        }    
        else {
            B[i] = h + (i - 1) * bh; 
            B[i + time_step] = B[i]; 
            B[i + 2 * time_step] = B[i]; 
            B[i + 3 * time_step] = B[i]; 
        } 
        C[i] = x_temp + i * hh;
        C[i + time_step] = x_temp + (i + time_step) * hh; 
        C[i + 2 * time_step] = x_temp + (i + 2 * time_step) * hh; 
        C[i + 3 * time_step] = x_temp + (i + 3 * time_step) * hh; 
    } 
    cblas_sgemm_batch(CblasRowMajor, transA, transB, m, n, k, alpha, A, lda, B, ldb, beta, C, ldc, 1, size_per_grp);
    #pragma omp parallel for
    for(i = 0; i < hid * hid; i++) {
        for(j = 0; j < time_step; j++) {
            dwfh[i] += C[j][i];
            dwih[i] += C[j + time_step][i];
            dwch[i] += C[j + 2 * time_step][i];
            dwoh[i] += C[j + 3 * time_step][i];
        }
    }

    //calculate dx
    m[0] = input_dim;
    k[0] = hid;
    n[0] = batch_size;
    
    lda[0] = m[0]; 
    ldb[0] = n[0]; 
    ldc[0] = n[0]; 
    transA[0] = CblasTrans;
    transB[0] = CblasNoTrans;
    size_per_grp[0] = 4 * time_step;
    for (i = 0; i < time_step; i++) { 
        A[i] = w_x;                                       // w_ix
        A[i + time_step] = w_x + input_dim * hid;         // w_fx
        A[i + 2 * time_step] = w_x + 2 * input_dim * hid; // w_cx 
        A[i + 3 * time_step] = w_x + 3 * input_dim * hid; // w_ox 
    
        B[i] = dhi + i * bh; 
        B[i + time_step] = dhf + i * bh; 
        B[i + 2 * time_step] = dhc + i * bh; 
        B[i + 3 * time_step] = dho + i * bh; 
    
        C[i] = x_temp + i * ib;
        C[i + time_step] = x_temp + (i + time_step) * ib; 
        C[i + 2 * time_step] = x_temp + (i + 2 * time_step)* ib; 
        C[i + 3 * time_step] = x_temp + (i + 3 * time_step)* ib; 
    }
    cblas_sgemm_batch(CblasRowMajor, transA, transB, m, n, k, alpha, A, lda, B, ldb, beta, C, ldc, 1, size_per_grp); 
    for(i = 0; i < tbi; i++) {
        dx[i] = x_temp[i] + x_temp[i + tbi] + x_temp[i + 2 * tbi] + x_temp[i + 3 * tbi];
    }
}
void parseCmdLine(int argc, char **argv)
{
        if(checkCmdLineFlag( argc, (const char**) argv, "help") || (argc == 1))
        {
                printf("--help:\t\t\t print this menu\n");
                printf("--batch_size=[int]:\t\t number of kerenl execution times (default: 1000)\n");
                printf("--time_step=[int]:\t\t height of W \n");
                printf("--input_dim=[int]:\t\t width of W \n");
                printf("--hid=[int]:\t\t width of X \n");
                exit(0);
        }

        if(checkCmdLineFlag( argc, (const char**) argv, "batch_size"))
        {
                batch_size = getCmdLineArgumentInt( argc, (const char**) argv, "batch_size");
        }

        if (checkCmdLineFlag( argc, (const char**) argv, "time_step"))
        {
                time_step = getCmdLineArgumentInt( argc, (const char**) argv, "time_step");
        }

        if (checkCmdLineFlag( argc, (const char**) argv, "input_dim"))
        {
                input_dim = getCmdLineArgumentInt( argc, (const char**) argv, "input_dim");
        }

        if (checkCmdLineFlag( argc, (const char**) argv, "hid"))
        {
                hid = getCmdLineArgumentInt( argc, (const char**) argv, "hid");
        }
}

void main(int argc, char** argv) {
    printf("-----\n");
    srand(45678);
    int i,j;
    parseCmdLine(argc, argv);
    printf("time_step=%d\n", time_step);
    printf("input_dim=%d\n", input_dim);
    printf("hid=%d\n", hid);
    printf("batch_size=%d\n", batch_size);
    
    //global forward
    const float** A = (const float**)mkl_malloc(4 * time_step * sizeof (float*), 64);
    const float** B = (const float**)mkl_malloc(4 * time_step * sizeof (float*), 64);
    float** C = (float**)mkl_malloc(4 * time_step * sizeof (float*), 64);
    float* x_temp = (float*)mkl_malloc(time_step * 4 * batch_size * hid * sizeof (float), 64);
    //global backward
    
    float* w_x;
    float* w_h;
    float* b;
    float* x;
    float* h_0;
    float* c_0;
    float* grad_last;
    //forward input
    w_x = (float*)mkl_malloc(4 * hid * input_dim * sizeof (float), 64);
    w_h = (float*)mkl_malloc(4 * hid * hid * sizeof (float), 64);
    b = (float*)mkl_malloc(4 * hid * sizeof (float), 64);
    //b = NULL;
    x = (float*)mkl_malloc(time_step * input_dim * batch_size * sizeof (float), 64);
    h_0 = (float*)mkl_malloc(hid * batch_size * sizeof (float), 64);
    c_0 = (float*)mkl_malloc(hid * batch_size * sizeof (float), 64);
    grad_last = (float*)mkl_malloc(hid * batch_size * sizeof (float), 64);
    for (i = 0; i < 4 * hid * input_dim; i++) {
        w_x[i] = ((float)rand()/(float)RAND_MAX) - ((float)rand()/(float)RAND_MAX);
    }
    for (i = 0; i < 4 * hid * hid; i++) {
        w_h[i] = ((float)rand()/(float)RAND_MAX) - ((float)rand()/(float)RAND_MAX);
    }
    for (i = 0; i < 4 * hid; i++) {
        b[i] = ((float)rand()/(float)RAND_MAX);
    }
    for (i = 0; i < time_step * input_dim * batch_size; i++) {
        x[i] = ((float)rand()/(float)RAND_MAX * 2.0f - 1.0f);
    }
    for (i = 0; i < hid * batch_size; i++) {
        h_0[i] = ((float)rand()/(float)RAND_MAX);
    }
    for (i = 0; i < hid * batch_size; i++) {
        c_0[i] = ((float)rand()/(float)RAND_MAX);
    }
    for (i = 0; i < hid * batch_size; i++) {
        grad_last[i] = 1.0;
    }
    //output
    float* dall = (float*)mkl_malloc( (hid * input_dim * 4 + hid * hid * 4 + hid * 4 + time_step * input_dim * batch_size) * sizeof (float), 64);
    memset(dall, 0, sizeof(float) * (hid * input_dim * 4 + hid * hid * 4 + hid * 4 + time_step * input_dim * batch_size));

    //share from forward
    float* hf = (float*)mkl_malloc(time_step * batch_size * hid * sizeof (float), 64);
    float* hi = (float*)mkl_malloc(time_step * batch_size * hid * sizeof (float), 64);
    float* hc = (float*)mkl_malloc(time_step * batch_size * hid * sizeof (float), 64);
    float* ho = (float*)mkl_malloc(time_step * batch_size * hid * sizeof (float), 64);
    float* c = (float*)mkl_malloc(time_step * batch_size * hid * sizeof (float), 64);

    //forward output
    float* h = (float*)mkl_malloc(time_step * hid * batch_size * sizeof (float), 64);
    memset(h, 0, sizeof(float) * time_step * hid * batch_size);

    //backward tmp memory
    float *dh = (float*)mkl_malloc(hid * batch_size * sizeof (float), 64); 
    float *dc = (float*)mkl_malloc(hid * batch_size * sizeof (float), 64);
    float *dh_next = (float*)mkl_malloc(hid * batch_size * sizeof (float), 64); 
    float *dc_next = (float*)mkl_malloc(hid * batch_size * sizeof (float), 64);
    float *dhf = (float*)mkl_malloc(time_step * hid * batch_size * sizeof (float), 64); 
    float *dhi = (float*)mkl_malloc(time_step * hid * batch_size * sizeof (float), 64); 
    float *dhc = (float*)mkl_malloc(time_step * hid * batch_size * sizeof (float), 64); 
    float *dho = (float*)mkl_malloc(time_step * hid * batch_size * sizeof (float), 64); 
    int max_size = max(4 * time_step * hid * batch_size, 4 * time_step * hid * input_dim);
    max_size = max(max_size, 4 * time_step * hid * hid);
    max_size = max(max_size, 4 * time_step * input_dim * batch_size);
    printf("max_size=%d\n", max_size);
    float *temp = (float*)mkl_malloc(max_size * sizeof (float), 64); 

    LSTM_forward(batch_size, time_step, input_dim, hid,
                         w_x, w_h, b, x, h_0, c_0,
                         ho, hf, hi, hc, //hid * batch_size
                         c, h,//time_Step * hid * batch_size
                         A, B, C, x_temp);
    LSTM_backward(batch_size, time_step, input_dim, hid, 
                    w_x, w_h, b, x, h_0, c_0,//same with forward input
                    ho, hf, hi, hc, c, h,//forward output: time_step * hid * batch_size
                    grad_last,//gz: (H,N)
                    dall + hid * input_dim * 2,
                    dall,
                    dall + hid * input_dim,
                    dall + hid * input_dim * 3,//dwxf,i,c,o

                    dall + hid * input_dim * 4 + hid * hid * 2, 
                    dall + hid * input_dim * 4, 
                    dall + hid * input_dim * 4 + hid * hid, 
                    dall + hid * input_dim * 4 + hid * hid * 3,//dwhf,i,c,o

                    dall + hid * input_dim * 4 + hid * hid * 4 + hid * 2,
                    dall + hid * input_dim * 4 + hid * hid * 4, 
                    dall + hid * input_dim * 4 + hid * hid * 4 + hid, 
                    dall + hid * input_dim * 4 + hid * hid * 4 + hid * 3,//dbf,i,c,o 
                    dall + hid * input_dim * 4 + hid * hid * 4 + hid * 4,//dx*/
                    A, B, C, dh, dc, dh_next, dc_next, dhf, dhi, dhc, dho, temp);
                    //icfo    
    struct timeval tic, toc, tic_all, toc_all;
    gettimeofday(&tic_all, NULL);
    gettimeofday(&tic, NULL);
    for(i = 0; i < loops; i++) {
        LSTM_forward(batch_size, time_step, input_dim, hid,
                         w_x, w_h, b, x, h_0, c_0,
                         ho, hf, hi, hc, //hid * batch_size
                         c, h, //time_Step * hid * batch_size
                         A, B, C, x_temp);
    }
    gettimeofday(&toc, NULL);
    float interval = (toc.tv_sec-tic.tv_sec)*1000 + (float)(toc.tv_usec-tic.tv_usec)/1000;
    printf("forward samples/s: %f, time/ms:%f\n", batch_size*loops  / (interval / 1000), interval);
    
    gettimeofday(&tic, NULL);
    for(i = 0; i < loops; i++) {
        LSTM_backward(batch_size, time_step, input_dim, hid, 
                    w_x, w_h, b, x, h_0, c_0,//same with forward input
                    ho, hf, hi, hc, c, h,//forward output: time_step * hid * batch_size
                    grad_last,//gz: (H,N)
                    dall + hid * input_dim * 2,
                    dall,
                    dall + hid * input_dim,
                    dall + hid * input_dim * 3,//dwxf,i,c,o

                    dall + hid * input_dim * 4 + hid * hid * 2, 
                    dall + hid * input_dim * 4, 
                    dall + hid * input_dim * 4 + hid * hid, 
                    dall + hid * input_dim * 4 + hid * hid * 3,//dwhf,i,c,o

                    dall + hid * input_dim * 4 + hid * hid * 4 + hid * 2,
                    dall + hid * input_dim * 4 + hid * hid * 4, 
                    dall + hid * input_dim * 4 + hid * hid * 4 + hid, 
                    dall + hid * input_dim * 4 + hid * hid * 4 + hid * 3,//dbf,i,c,o 
                    dall + hid * input_dim * 4 + hid * hid * 4 + hid * 4,//dx
                    A, B, C, dh, dc, dh_next, dc_next, dhf, dhi, dhc, dho, temp);
                    //icfo
    }
    gettimeofday(&toc, NULL);
    interval = (toc.tv_sec-tic.tv_sec)*1000 + (float)(toc.tv_usec-tic.tv_usec)/1000;
    printf("backward samples/s: %f, time/ms:%f\n", batch_size*loops  / (interval / 1000), interval);
    gettimeofday(&toc_all, NULL);
    interval = (toc_all.tv_sec-tic_all.tv_sec)*1000 + (float)(toc_all.tv_usec-tic_all.tv_usec)/1000;
    printf("all samples/s: %f, time/ms:%f\n", batch_size*loops  / (interval / 1000), interval);
    mkl_free(w_x);
    mkl_free(w_h);
    mkl_free(x);
    mkl_free(b);
    mkl_free(h_0);
    mkl_free(c_0);
    mkl_free(grad_last);

    mkl_free(hf);
    mkl_free(hi);
    mkl_free(hc);
    mkl_free(ho);
    mkl_free(h);
    mkl_free(c);
    mkl_free(dall);

    mkl_free(A);
    mkl_free(B);
    mkl_free(C);
    mkl_free(x_temp);

    mkl_free(dh);
    mkl_free(dc);  
    mkl_free(dh_next);
    mkl_free(dc_next);
    mkl_free(dhf);
    mkl_free(dhi);
    mkl_free(dhc);
    mkl_free(dho);
    mkl_free(temp);
}
