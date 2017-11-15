#include <stdio.h>
#include <stdlib.h>
#include "mkl.h"
#include<sys/time.h>
#include <omp.h>
#include <stdbool.h>
#include "helper_string.h"
#include "math.h"

int loops = 100;
int batch_size = 64;
int time_step = 10;
int input_dim = 150;
int hid = 1024;

//global forward
const float** AF;
const float** BF;
float** CF;
float* x_temp;

//global backward
const float** A;
const float** B;
float** C;

void  LSTM_batch_gemm(int batch_size, int time_step, int input_dim, int hid, 
                      const float* w_x, const float* w_h, const float* b, const float* x, const float* h_0, const float* c_0, 
               /*out*/float *o_t, float *f_t, float *i_t, float* c_wave_t, //hid * batch_size
                      float* c_t, float* h){//time_Step * hid * batch_size
    ////global forward
    //const float** AF = (const float**)mkl_malloc(4 * time_step * sizeof (float*), 64);
    //const float** BF = (const float**)mkl_malloc(4 * time_step * sizeof (float*), 64);
    //float** CF = (float**)mkl_malloc(4 * time_step * sizeof (float*), 64);
    //float* x_temp = (float*)mkl_malloc(time_step * 4 * batch_size * hid * sizeof (float), 64);
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
            for (j = 0; j < batch_size; j++) { 
                for (p = 0; p < hid; p++) { 
                    size_t offset0 = i * batch_size * hid + j * hid + p; 
                    size_t offset1 = (i + time_step) * batch_size * hid + j * hid + p; 
                    size_t offset2 = (i + 2 * time_step) * batch_size * hid + j * hid + p; 
                    size_t offset3 = (i + 3 * time_step) * batch_size * hid + j * hid + p; 
        
                    x_temp[offset0] = b[p]; 
                    x_temp[offset1] = b[p + hid]; 
                    x_temp[offset2] = b[p + 2 * hid]; 
                    x_temp[offset2] = b[p + 3 * hid]; 
                } 
            } 
        } 
    }
    size_per_grp[0] = 4 * time_step;

    if (NULL == AF || NULL == BF || NULL == CF || NULL == x_temp) {
        printf( "\n ERROR: malloc global buffers failed \n\n");
        return;
    }
    #pragma omp parallel for 
    for (i = 0; i < time_step; i++) { 
        AF[i] = w_x;                                       // w_fx
        AF[i + time_step] = w_x + input_dim * hid;         // w_ix
        AF[i + 2 * time_step] = w_x + 2 * input_dim * hid; // w_cx 
        AF[i + 3 * time_step] = w_x + 3 * input_dim * hid; // w_ox 
    
        BF[i] = x + i * k[0] * n[0]; 
        BF[i + time_step] = BF[i]; 
        BF[i + 2 * time_step] = BF[i]; 
        BF[i + 3 * time_step] = BF[i]; 
    
        CF[i] = x_temp + i * m[0] * n[0]; 
        CF[i + time_step] = x_temp + (i + time_step) * m[0] * n[0]; 
        CF[i + 2 * time_step] = x_temp + (i + 2 * time_step) * m[0] * n[0]; 
        CF[i + 3 * time_step] = x_temp + (i + 3 * time_step) * m[0] * n[0]; 
    } 
    cblas_sgemm_batch(CblasRowMajor, transA, transB, m, n, k, alpha, AF, lda, BF, ldb, beta, CF, ldc, 1, size_per_grp); 

    // loop on step
    m[0] = hid;
    k[0] = hid;
    n[0] = batch_size;
    
    beta[0] = 1.0;

    lda[0] = k[0]; 
    ldb[0] = n[0]; 
    ldc[0] = n[0]; 
    size_per_grp[0] = 4;
    
    AF[0] = w_h;                //w_fh
    AF[1] = w_h + hid * hid;    //w_ih
    AF[2] = w_h + 2 * hid * hid;//w_ch
    AF[3] = w_h + 3 * hid * hid;//w_oh
    
    BF[0] = h_0;
    BF[1] = h_0;
    BF[2] = h_0;
    BF[3] = h_0;

    size_t mn = m[0] * n[0];
    #pragma omp parallel for
    for (j = 0; j < mn; j++) {
        c_t[j] = c_0[j];
    }

    for (i = 0; i < time_step; i++) {
        // f,i,c_wave,o
        CF[0] = x_temp + i * m[0] * n[0];
        CF[1] = x_temp + (i + time_step) * m[0] * n[0];
        CF[2] = x_temp + (i + 2 * time_step) * m[0] * n[0];
        CF[3] = x_temp + (i + 3 * time_step) * m[0] * n[0];

        cblas_sgemm_batch(CblasRowMajor, transA, transB, m, n, k, alpha, AF, lda, BF, ldb, beta, CF, ldc, 1, size_per_grp);

        // sigmoid for f,i,o, tanh for c_wave
        #pragma omp parallel for
        for (j = 0; j < mn; j++) {
            int id1 = i * mn + j;
            float exp_f = exp((float)(CF[0][j]));
            float exp_i = exp((float)(CF[1][j]));
            c_wave_t[id1] = tanh((float)(CF[2][j]));
            float exp_o = exp((float)(CF[3][j]));
            f_t[id1] = exp_f / ((float)1.0 + exp_f);        
            i_t[id1] = exp_i / ((float)1.0 + exp_i);
            o_t[id1] = exp_o / ((float)1.0 + exp_o);
        }
        //c
        float c_tm1;
        #pragma omp parallel for 
        for (j = 0; j < mn; j++) { 
            int id2 = i * mn + j;
            if(i == 0) 
                c_tm1 = c_0[j];
            else
                c_tm1 = c_t[id2 - mn];
            c_t[id2] = (float)((float)(f_t[id2]) * (float)(c_tm1) + (float)(i_t[id2]) * (float)(c_wave_t[id2])); 
        }
        //h:all time_step
        float* y_ptr = NULL;
        y_ptr = h + i * batch_size * hid;
        #pragma omp parallel for
        for (j = 0; j < mn; j++) {
            int id3 = i * mn + j;
            y_ptr[j] = (float)(o_t[id3]) * tanh((float)(c_t[id3]));
        }
        // update
        BF[0] = y_ptr;
        BF[1] = BF[0];
        BF[2] = BF[0];
        BF[3] = BF[0];
    }
    //mkl_free(AF);
    //mkl_free(BF);
    //mkl_free(CF);
    //mkl_free(x_temp);
}

void  LSTM_backward(int batch_size, int time_step, int input_dim, int hid, 
                    const float* w_x, const float* w_h, const float* b, const float* x, const float* h_0, const float* c_0,//same with forward input
                    float *ho, float *hf, float *hi, float* hc, float* c, float* h,//forward output: time_step * hid * batch_size
             /*out*/float* dwfx, float* dwix, float* dwcx, float* dwox,
                    float* dwfh, float* dwih, float* dwch, float* dwoh, 
                    float* dbf, float* dbi, float* dbc, float* dbo, 
                    float* dx){
    ////global backward
    //const float** A = (const float**)mkl_malloc(4 * time_step * sizeof (float*), 64);
    //const float** B = (const float**)mkl_malloc(4 * time_step * sizeof (float*), 64);
    //float** C = (float**)mkl_malloc(4 * time_step * sizeof (float*), 64);
    
    int i,j;
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
    float *dh = (float*)mkl_malloc(hid * batch_size * sizeof (float), 64); 
    float *dc = (float*)mkl_malloc(hid * batch_size * sizeof (float), 64);
    float *dh_next = (float*)mkl_malloc(hid * batch_size * sizeof (float), 64); 
    float *dc_next = (float*)mkl_malloc(hid * batch_size * sizeof (float), 64);
    memset(dh_next, 0, sizeof(float) * hid * batch_size);
    memset(dc_next, 0, sizeof(float) * hid * batch_size);
    //temp mem
    float *dhf = (float*)mkl_malloc(time_step * hid * batch_size * sizeof (float), 64); 
    float *dhi = (float*)mkl_malloc(time_step * hid * batch_size * sizeof (float), 64); 
    float *dhc = (float*)mkl_malloc(time_step * hid * batch_size * sizeof (float), 64); 
    float *dho = (float*)mkl_malloc(time_step * hid * batch_size * sizeof (float), 64); 
    
    float *dhhf = (float*)mkl_malloc(hid * batch_size * sizeof (float), 64); 
    float *dhhi = (float*)mkl_malloc(hid * batch_size * sizeof (float), 64); 
    float *dhhc = (float*)mkl_malloc(hid * batch_size * sizeof (float), 64); 
    float *dhho = (float*)mkl_malloc(hid * batch_size * sizeof (float), 64); 
    
    float *dwfx_allt = (float*)mkl_malloc(time_step * hid * input_dim * sizeof (float), 64);
    float *dwix_allt = (float*)mkl_malloc(time_step * hid * input_dim * sizeof (float), 64);
    float *dwcx_allt = (float*)mkl_malloc(time_step * hid * input_dim * sizeof (float), 64);
    float *dwox_allt = (float*)mkl_malloc(time_step * hid * input_dim * sizeof (float), 64);
    
    float *dwfh_allt = (float*)mkl_malloc(time_step * hid * hid * sizeof (float), 64);
    float *dwih_allt = (float*)mkl_malloc(time_step * hid * hid * sizeof (float), 64);
    float *dwch_allt = (float*)mkl_malloc(time_step * hid * hid * sizeof (float), 64);
    float *dwoh_allt = (float*)mkl_malloc(time_step * hid * hid * sizeof (float), 64);

    float *dxf = (float*)mkl_malloc(time_step * input_dim * batch_size * sizeof (float), 64);
    float *dxi = (float*)mkl_malloc(time_step * input_dim * batch_size * sizeof (float), 64);
    float *dxc = (float*)mkl_malloc(time_step * input_dim * batch_size * sizeof (float), 64);
    float *dxo = (float*)mkl_malloc(time_step * input_dim * batch_size * sizeof (float), 64);
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
    
    A[0] = w_h;                //w_fh
    A[1] = w_h + hid * hid;    //w_ih
    A[2] = w_h + 2 * hid * hid;//w_ch
    A[3] = w_h + 3 * hid * hid;//w_oh
    
    C[0] = dhhf;
    C[1] = dhhi;
    C[2] = dhhc;
    C[3] = dhho;
    size_t bh = batch_size * hid;
    size_t tbi = batch_size * input_dim * time_step;
    size_t ib = input_dim * batch_size;
    size_t hh = hid * hid;
    for(i = time_step - 1; i >= 0; i--) {
        int kk = i * bh;
        #pragma omp parallel for 
        for(j = 0; j < bh; j++ ) {
            int index = kk + j;
            float c_old;
            if(i != 0) 
                c_old = c[index - bh];
            else
                c_old = c_0[j];
            float tanh_c = tanh(c[index]);
            if(i == time_step - 1) 
                dh[j] = 1.0 + dh_next[j];
            else
                dh[j] = dh_next[j];
            dho[index] = ho[index] * (1.0 - ho[index]) * tanh_c * dh[j];
            dc[j] = ho[index] * dh[j] * (1.0 - tanh_c * tanh_c) + dc_next[j];
            dhf[index] = hf[index] * (1.0 - hf[index]) * c_old * dc[j];
            dhi[index] = hi[index] * (1.0 - hi[index]) * hc[index] * dc[j];
            dhc[index] = (1.0 - hc[index] * hc[index]) * hi[index] * dc[j];
            //printf("dhf[%d]=%f\n", index, dhf[index]);
        }
        B[0] = dhf + kk;
        B[1] = dhi + kk;
        B[2] = dhc + kk;
        B[3] = dho + kk;
        cblas_sgemm_batch(CblasRowMajor, transA, transB, m, n, k, alpha, A, lda, B, ldb, beta, C, ldc, 1, size_per_grp);
    
        //calculate dbf, dbi, dbc, dbo
        #pragma omp parallel for 
        for(j = 0; j < bh; j++ ) {
            int index = kk + j;
            dh_next[j] = dhhf[j] + dhhi[j] + dhhc[j] + dhho[j];
            dc_next[j] = hf[index] * dc[j];
        }
        #pragma omp parallel for 
        for(j = 0; j < hid; j++) {
            for(int p = 0; p < batch_size; p++) {
                int index = kk + j * batch_size + p;
                dbf[j] += dhf[index];
                dbi[j] += dhi[index];
                dbc[j] += dhc[index];
                dbo[j] += dho[index];
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
    //#pragma omp parallel for 
    for (i = 0; i < time_step; i++) {
        A[i] = dhf + i * bh;                 
        A[i + time_step] = dhi + i * bh;    
        A[i + 2 * time_step] = dhc + i * bh; 
        A[i + 3 * time_step] = dho + i * bh; 
    
        B[i] = x + i * input_dim * batch_size; 
        B[i + time_step] = B[i]; 
        B[i + 2 * time_step] = B[i]; 
        B[i + 3 * time_step] = B[i]; 
    
        C[i] = dwfx_allt + i * hid * input_dim;
        C[i + time_step] = dwix_allt + i * hid * input_dim; 
        C[i + 2 * time_step] = dwcx_allt + i * hid * input_dim; 
        C[i + 3 * time_step] = dwox_allt + i * hid * input_dim; 
    } 
    cblas_sgemm_batch(CblasRowMajor, transA, transB, m, n, k, alpha, A, lda, B, ldb, beta, C, ldc, 1, size_per_grp);
    //#pragma omp parallel for 
    for(i = 0; i < hid * input_dim; i++) {
        for(j = 0; j < time_step; j++) {
            int index = j * hid * input_dim + i;
            dwfx[i] += dwfx_allt[index];
            dwix[i] += dwix_allt[index];
            dwcx[i] += dwcx_allt[index];
            dwox[i] += dwox_allt[index];
            //printf("dwfx_allt[%d]=%f\n", index, dwfx_allt[index]);
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
    //#pragma omp parallel for 
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
        C[i] = dwfh_allt + i * hh;
        C[i + time_step] = dwih_allt + i * hh; 
        C[i + 2 * time_step] = dwch_allt + i * hh; 
        C[i + 3 * time_step] = dwoh_allt + i * hh; 
    } 
    cblas_sgemm_batch(CblasRowMajor, transA, transB, m, n, k, alpha, A, lda, B, ldb, beta, C, ldc, 1, size_per_grp);
    //#pragma omp parallel for 
    for(i = 0; i < hid * hid; i++) {
        for(j = 0; j < time_step; j++) {
            int index = j * hid * hid + i;
            dwfh[i] += dwfh_allt[index];
            dwih[i] += dwih_allt[index];
            dwch[i] += dwch_allt[index];
            dwoh[i] += dwoh_allt[index];
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
    //#pragma omp parallel for 
    for (i = 0; i < time_step; i++) { 
        A[i] = w_x;                                       // w_fx
        A[i + time_step] = w_x + input_dim * hid;         // w_ix
        A[i + 2 * time_step] = w_x + 2 * input_dim * hid; // w_cx 
        A[i + 3 * time_step] = w_x + 3 * input_dim * hid; // w_ox 
    
        B[i] = dhf + i * bh; 
        B[i + time_step] = dhi + i * bh; 
        B[i + 2 * time_step] = dhc + i * bh; 
        B[i + 3 * time_step] = dho + i * bh; 
    
        C[i] = dxf + i * ib;
        C[i + time_step] = dxi + i * ib; 
        C[i + 2 * time_step] = dxc + i * ib; 
        C[i + 3 * time_step] = dxo + i * ib; 
    } 
    //printf("dhf[0]=%f\n", dhf[0]);
    //printf("dxf=%f\n", dxf[0]);
    cblas_sgemm_batch(CblasRowMajor, transA, transB, m, n, k, alpha, A, lda, B, ldb, beta, C, ldc, 1, size_per_grp); 
    #pragma omp parallel for 
    for(i = 0; i < tbi; i++) {
        dx[i] = dxf[i] + dxi[i] + dxc[i] + dxo[i];
    }
    mkl_free(dh); 
    mkl_free(dc); 
    mkl_free(dh_next); 
    mkl_free(dc_next); 
    mkl_free(dhf); 
    mkl_free(dhi); 
    mkl_free(dhc); 
    mkl_free(dho); 
    mkl_free(dhhf); 
    mkl_free(dhhi); 
    mkl_free(dhhc); 
    mkl_free(dhho); 
    mkl_free(dwfx_allt); 
    mkl_free(dwix_allt); 
    mkl_free(dwcx_allt); 
    mkl_free(dwox_allt); 
    mkl_free(dwfh_allt); 
    mkl_free(dwih_allt);
    mkl_free(dwch_allt);
    mkl_free(dwoh_allt);
    mkl_free(dxf);
    mkl_free(dxi);
    mkl_free(dxc);
    mkl_free(dxo);
    //mkl_free(A);
    //mkl_free(B);
    //mkl_free(C);
}
//share global memory
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
    srand(45678);
    int i,j;
    parseCmdLine(argc, argv);
    printf("time_step=%d\n", time_step);
    printf("input_dim=%d\n", input_dim);
    printf("hid=%d\n", hid);
    printf("batch_size=%d\n", batch_size);
    
    //global forward
    AF = (const float**)mkl_malloc(4 * time_step * sizeof (float*), 64);
    BF = (const float**)mkl_malloc(4 * time_step * sizeof (float*), 64);
    CF = (float**)mkl_malloc(4 * time_step * sizeof (float*), 64);
    x_temp = (float*)mkl_malloc(time_step * 4 * batch_size * hid * sizeof (float), 64);
    //global backward
    A = (const float**)mkl_malloc(4 * time_step * sizeof (float*), 64);
    B = (const float**)mkl_malloc(4 * time_step * sizeof (float*), 64);
    C = (float**)mkl_malloc(4 * time_step * sizeof (float*), 64);
    
    float* w_x;
    float* w_h;
    float* b;
    float* x;
    float* h_0;
    float* c_0;
    //forward input
    w_x = (float*)mkl_malloc(4 * hid * input_dim * sizeof (float), 64);
    w_h = (float*)mkl_malloc(4 * hid * hid * sizeof (float), 64);
    b = (float*)mkl_malloc(4 * hid * sizeof (float), 64);
    x = (float*)mkl_malloc(time_step * input_dim * batch_size * sizeof (float), 64);
    h_0 = (float*)mkl_malloc(hid * batch_size * sizeof (float), 64);
    c_0 = (float*)mkl_malloc(hid * batch_size * sizeof (float), 64);
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
    //output
    float* dall = (float*)mkl_malloc( (hid * input_dim * 4 + hid * hid * 4 + hid * 4 + time_step * input_dim * batch_size) * sizeof (float), 64);
    memset(dall, 0, sizeof(float) * hid * input_dim * 4 + hid * hid * 4 + hid * 4 + time_step * input_dim * batch_size);

    //share from forward
    float* hf = (float*)mkl_malloc(time_step * batch_size * hid * sizeof (float), 64);
    float* hi = (float*)mkl_malloc(time_step * batch_size * hid * sizeof (float), 64);
    float* hc = (float*)mkl_malloc(time_step * batch_size * hid * sizeof (float), 64);
    float* ho = (float*)mkl_malloc(time_step * batch_size * hid * sizeof (float), 64);
    float* c = (float*)mkl_malloc(time_step * batch_size * hid * sizeof (float), 64);

    //forward output
    float* h = (float*)mkl_malloc(time_step * hid * batch_size * sizeof (float), 64);
    memset(h, 0, sizeof(float) * time_step * hid * batch_size);

    LSTM_batch_gemm(batch_size, time_step, input_dim, hid,
                         w_x, w_h, b, x, h_0, c_0,
                         ho, hf, hi, hc, //hid * batch_size
                         c, h);//time_Step * hid * batch_size
    LSTM_backward(batch_size, time_step, input_dim, hid, 
                    w_x, w_h, b, x, h_0, c_0,//same with forward input
                    ho, hf, hi, hc, c, h,//forward output: time_step * hid * batch_size
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
                    dall + hid * input_dim * 4 + hid * hid * 4 + hid * 4);//dx*/
                    //icfo    
    struct timeval tic, toc;
    gettimeofday(&tic, NULL);
    for(i = 0; i < loops; i++) {
        LSTM_batch_gemm(batch_size, time_step, input_dim, hid,
                         w_x, w_h, b, x, h_0, c_0,
                         ho, hf, hi, hc, //hid * batch_size
                         c, h);//time_Step * hid * batch_size
    }
    gettimeofday(&toc, NULL);
    float interval = (toc.tv_sec-tic.tv_sec)*1000 + (float)(toc.tv_usec-tic.tv_usec)/1000;
    printf("forward samples/s: %f, time/ms:%f\n", batch_size*loops  / (interval / 1000), interval);
    
    gettimeofday(&tic, NULL);
    for(i = 0; i < loops; i++) {
        LSTM_backward(batch_size, time_step, input_dim, hid, 
                    w_x, w_h, b, x, h_0, c_0,//same with forward input
                    ho, hf, hi, hc, c, h,//forward output: time_step * hid * batch_size
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
                    dall + hid * input_dim * 4 + hid * hid * 4 + hid * 4);//dx
                    //icfo
    }
    gettimeofday(&toc, NULL);
    interval = (toc.tv_sec-tic.tv_sec)*1000 + (float)(toc.tv_usec-tic.tv_usec)/1000;
    printf("backward samples/s: %f, time/ms:%f\n", batch_size*loops  / (interval / 1000), interval);
    mkl_free(w_x);
    mkl_free(w_h);
    mkl_free(x);
    mkl_free(h_0);
    mkl_free(c_0);
    mkl_free(b);

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
    mkl_free(AF);
    mkl_free(BF);
    mkl_free(CF);
    mkl_free(x_temp);
}

