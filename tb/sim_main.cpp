#include "Vtensor_core.h"
#include "verilated.h"
#include <cstdio>
#include <cmath>
#include <string>

static const int M = 4, N = 4, K = 4;
static const int DATA_W = 8, ACC_W = 32;

// Flat index helpers
static inline int AI(int i, int k) { return i*K+k; }
static inline int BI(int k, int j) { return k*N+j; }
static inline int CI(int i, int j) { return i*N+j; }
static inline int DI(int i, int j) { return i*N+j; }

static Vtensor_core* dut;
static uint64_t tick_count = 0;

static void tick() {
    dut->clk = 0; dut->eval();
    dut->clk = 1; dut->eval();
    tick_count++;
}

static void reset() {
    dut->rst_n = 0;
    dut->start = 0;
    for (int i = 0; i < M*K; i++) dut->a_mat[i] = 0;
    for (int i = 0; i < K*N; i++) dut->b_mat[i] = 0;
    for (int i = 0; i < M*N; i++) dut->c_mat[i] = 0;
    for (int i = 0; i < 4; i++) tick();
    dut->rst_n = 1;
    for (int i = 0; i < 2; i++) tick();
}

// Run one MMA and return pass/fail
static bool run_and_check(
    const char* name,
    int8_t  a[M][K],
    int8_t  b[K][N],
    int32_t c[M][N])
{
    // Load matrices
    for (int i = 0; i < M; i++)
        for (int k = 0; k < K; k++)
            dut->a_mat[AI(i,k)] = (uint32_t)(int32_t)a[i][k];
    for (int k = 0; k < K; k++)
        for (int j = 0; j < N; j++)
            dut->b_mat[BI(k,j)] = (uint32_t)(int32_t)b[k][j];
    for (int i = 0; i < M; i++)
        for (int j = 0; j < N; j++)
            dut->c_mat[CI(i,j)] = (uint32_t)c[i][j];

    // Compute reference
    int32_t expected[M][N];
    for (int i = 0; i < M; i++)
        for (int j = 0; j < N; j++) {
            expected[i][j] = c[i][j];
            for (int k = 0; k < K; k++)
                expected[i][j] += (int32_t)a[i][k] * (int32_t)b[k][j];
        }

    // Pulse start
    tick();
    dut->start = 1; tick();
    dut->start = 0;

    // Wait for done (timeout after 200 cycles)
    int timeout = 200;
    while (!dut->done && timeout-- > 0) tick();
    if (timeout <= 0) { printf("FAIL  %s  (TIMEOUT)\n", name); return false; }

    // done fired; d_mat updates this same clock edge (nonblocking → read after extra tick)
    tick();

    // Check
    int errors = 0;
    for (int i = 0; i < M; i++)
        for (int j = 0; j < N; j++) {
            int32_t got = (int32_t)dut->d_mat[DI(i,j)];
            if (got != expected[i][j]) {
                printf("  MISMATCH [%d][%d]: got %d  expected %d\n",
                       i, j, got, expected[i][j]);
                errors++;
            }
        }

    if (errors == 0) printf("PASS  %s\n", name);
    else             printf("FAIL  %s  (%d errors)\n", name, errors);
    return errors == 0;
}

static void print_result(const char* label) {
    printf("%s:\n", label);
    for (int i = 0; i < M; i++) {
        printf("  [");
        for (int j = 0; j < N; j++)
            printf(" %6d", (int32_t)dut->d_mat[DI(i,j)]);
        printf(" ]\n");
    }
}

int main(int argc, char** argv) {
    Verilated::commandArgs(argc, argv);
    dut = new Vtensor_core;

    printf("=== Tensor Core Testbench (Verilator) ===\n");
    printf("Config: M=%d K=%d N=%d  DATA_W=%d ACC_W=%d\n", M,K,N,DATA_W,ACC_W);

    reset();
    int pass = 0, total = 0;

    // Test 1: Identity x Identity
    {
        int8_t  a[M][K] = {{1,0,0,0},{0,1,0,0},{0,0,1,0},{0,0,0,1}};
        int8_t  b[K][N] = {{1,0,0,0},{0,1,0,0},{0,0,1,0},{0,0,0,1}};
        int32_t c[M][N] = {};
        pass += run_and_check("Identity x Identity + 0", a, b, c); total++;
    }

    // Test 2: All-ones
    {
        int8_t  a[M][K], b[K][N]; int32_t c[M][N] = {};
        for (int i=0;i<M;i++) for (int k=0;k<K;k++) a[i][k]=1;
        for (int k=0;k<K;k++) for (int j=0;j<N;j++) b[k][j]=1;
        pass += run_and_check("Ones x Ones (expect 4 everywhere)", a, b, c); total++;
        print_result("Ones result");
    }

    // Test 3: Arbitrary + non-zero C
    {
        int8_t a[M][K] = {{1,2,3,4},{5,6,7,8},{-1,-2,-3,-4},{2,-3,1,-2}};
        int8_t b[K][N] = {{1,0,-1,2},{3,-1,0,1},{2,1,-2,0},{-1,2,1,3}};
        int32_t c[M][N] = {{100,0,0,0},{0,200,0,0},{0,0,50,0},{10,20,30,40}};
        pass += run_and_check("Arbitrary A*B + C", a, b, c); total++;
    }

    // Test 4: Negative values
    {
        int8_t a[M][K] = {{-1,-2,-3,-4},{-5,-6,-7,-8},{1,2,3,4},{0,0,0,0}};
        int8_t b[K][N] = {{-1,-2,-3,-4},{-5,-6,-7,-8},{1,2,3,4},{0,0,0,0}};
        int32_t c[M][N] = {};
        pass += run_and_check("Negative values", a, b, c); total++;
    }

    // Test 5: Back-to-back #1
    {
        int8_t a[M][K]={{2,0,0,0},{0,2,0,0},{0,0,2,0},{0,0,0,2}};
        int8_t b[K][N]={{3,0,0,0},{0,3,0,0},{0,0,3,0},{0,0,0,3}};
        int32_t c[M][N]={};
        pass += run_and_check("Back-to-back #1: 2I x 3I = 6I", a, b, c); total++;
    }

    // Test 6: Back-to-back #2
    {
        int8_t  a[M][K], b[K][N]; int32_t c[M][N]={};
        for (int i=0;i<M;i++) for (int k=0;k<K;k++) a[i][k]=i+1;
        for (int k=0;k<K;k++) for (int j=0;j<N;j++) b[k][j]=j+1;
        pass += run_and_check("Back-to-back #2: outer product", a, b, c); total++;
    }

    printf("\n=== %d/%d tests passed ===\n", pass, total);
    dut->final();
    delete dut;
    return (pass == total) ? 0 : 1;
}
