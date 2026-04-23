`timescale 1ns/1ps

// Index helpers (macros to keep code readable)
// A[i][k] stored flat as a_mat[i*K+k], etc.
`define AI(i,k) ((i)*K+(k))
`define BI(k,j) ((k)*N+(j))
`define CI(i,j) ((i)*N+(j))
`define DI(i,j) ((i)*N+(j))

module tensor_core_tb;

    localparam int M      = 4;
    localparam int N      = 4;
    localparam int K      = 4;
    localparam int DATA_W = 8;
    localparam int ACC_W  = 32;
    localparam int CLK_P  = 10;

    logic                          clk, rst_n, start, done, busy;
    logic signed [DATA_W-1:0]      a_mat [M*K];
    logic signed [DATA_W-1:0]      b_mat [K*N];
    logic signed [ACC_W-1:0]       c_mat [M*N];
    logic signed [ACC_W-1:0]       d_mat [M*N];

    tensor_core #(.M(M),.N(N),.K(K),.DATA_W(DATA_W),.ACC_W(ACC_W)) dut (.*);

    initial clk = 0;
    always #(CLK_P/2) clk = ~clk;

    // ── Reference model ───────────────────────────────────────────────────────
    logic signed [ACC_W-1:0] expected [M*N];

    task automatic compute_expected();
        integer i, j, k;
        for (i = 0; i < M; i++)
            for (j = 0; j < N; j++) begin
                expected[`CI(i,j)] = c_mat[`CI(i,j)];
                for (k = 0; k < K; k++)
                    expected[`CI(i,j)] +=
                        ACC_W'(signed'(a_mat[`AI(i,k)])) *
                        ACC_W'(signed'(b_mat[`BI(k,j)]));
            end
    endtask

    task automatic run_and_check(input string name);
        integer i, j, errors;
        compute_expected();

        @(posedge clk); #1;
        start = 1;
        @(posedge clk); #1;
        start = 0;

        // done fires in the OUTPUT state; d_mat updates that same clock edge
        // (nonblocking), so it's readable one delta after the NEXT clock edge.
        wait (done === 1'b1);
        @(posedge clk); #1;

        errors = 0;
        for (i = 0; i < M; i++)
            for (j = 0; j < N; j++)
                if (d_mat[`DI(i,j)] !== expected[`CI(i,j)]) begin
                    $display("  MISMATCH [%0d][%0d]: got %0d  expected %0d",
                             i, j,
                             $signed(d_mat[`DI(i,j)]),
                             $signed(expected[`CI(i,j)]));
                    errors++;
                end
        if (errors == 0) $display("PASS  %s", name);
        else             $display("FAIL  %s  (%0d errors)", name, errors);
    endtask

    task automatic zero_c();
        integer ii;
        for (ii = 0; ii < M*N; ii++) c_mat[ii] = '0;
    endtask

    task automatic print_result(input string label);
        integer i, j;
        $display("%s:", label);
        for (i = 0; i < M; i++) begin
            $write("  [");
            for (j = 0; j < N; j++)
                $write(" %6d", $signed(d_mat[`DI(i,j)]));
            $display(" ]");
        end
    endtask

    // ── Test 1: Identity × Identity ──────────────────────────────────────────
    task automatic test_identity();
        integer i, k, j;
        for (i = 0; i < M; i++)
            for (k = 0; k < K; k++)
                a_mat[`AI(i,k)] = (i == k) ? 1 : 0;
        for (k = 0; k < K; k++)
            for (j = 0; j < N; j++)
                b_mat[`BI(k,j)] = (k == j) ? 1 : 0;
        zero_c();
        run_and_check("Identity x Identity + 0");
    endtask

    // ── Test 2: All-ones ──────────────────────────────────────────────────────
    task automatic test_ones();
        integer ii;
        for (ii = 0; ii < M*K; ii++) a_mat[ii] = 1;
        for (ii = 0; ii < K*N; ii++) b_mat[ii] = 1;
        zero_c();
        run_and_check("Ones x Ones (expect K=4 everywhere)");
    endtask

    // ── Test 3: Arbitrary + non-zero C ───────────────────────────────────────
    task automatic test_arbitrary();
        // A
        a_mat[`AI(0,0)]=1;  a_mat[`AI(0,1)]=2;  a_mat[`AI(0,2)]=3;  a_mat[`AI(0,3)]=4;
        a_mat[`AI(1,0)]=5;  a_mat[`AI(1,1)]=6;  a_mat[`AI(1,2)]=7;  a_mat[`AI(1,3)]=8;
        a_mat[`AI(2,0)]=-1; a_mat[`AI(2,1)]=-2; a_mat[`AI(2,2)]=-3; a_mat[`AI(2,3)]=-4;
        a_mat[`AI(3,0)]=2;  a_mat[`AI(3,1)]=-3; a_mat[`AI(3,2)]=1;  a_mat[`AI(3,3)]=-2;
        // B
        b_mat[`BI(0,0)]=1;  b_mat[`BI(0,1)]=0;  b_mat[`BI(0,2)]=-1; b_mat[`BI(0,3)]=2;
        b_mat[`BI(1,0)]=3;  b_mat[`BI(1,1)]=-1; b_mat[`BI(1,2)]=0;  b_mat[`BI(1,3)]=1;
        b_mat[`BI(2,0)]=2;  b_mat[`BI(2,1)]=1;  b_mat[`BI(2,2)]=-2; b_mat[`BI(2,3)]=0;
        b_mat[`BI(3,0)]=-1; b_mat[`BI(3,1)]=2;  b_mat[`BI(3,2)]=1;  b_mat[`BI(3,3)]=3;
        // C
        c_mat[`CI(0,0)]=100; c_mat[`CI(0,1)]=0;   c_mat[`CI(0,2)]=0;  c_mat[`CI(0,3)]=0;
        c_mat[`CI(1,0)]=0;   c_mat[`CI(1,1)]=200; c_mat[`CI(1,2)]=0;  c_mat[`CI(1,3)]=0;
        c_mat[`CI(2,0)]=0;   c_mat[`CI(2,1)]=0;   c_mat[`CI(2,2)]=50; c_mat[`CI(2,3)]=0;
        c_mat[`CI(3,0)]=10;  c_mat[`CI(3,1)]=20;  c_mat[`CI(3,2)]=30; c_mat[`CI(3,3)]=40;
        run_and_check("Arbitrary A*B + C");
    endtask

    // ── Test 4: Negative values ───────────────────────────────────────────────
    task automatic test_negative();
        a_mat[`AI(0,0)]=-1; a_mat[`AI(0,1)]=-2; a_mat[`AI(0,2)]=-3; a_mat[`AI(0,3)]=-4;
        a_mat[`AI(1,0)]=-5; a_mat[`AI(1,1)]=-6; a_mat[`AI(1,2)]=-7; a_mat[`AI(1,3)]=-8;
        a_mat[`AI(2,0)]=1;  a_mat[`AI(2,1)]=2;  a_mat[`AI(2,2)]=3;  a_mat[`AI(2,3)]=4;
        a_mat[`AI(3,0)]=0;  a_mat[`AI(3,1)]=0;  a_mat[`AI(3,2)]=0;  a_mat[`AI(3,3)]=0;
        b_mat[`BI(0,0)]=-1; b_mat[`BI(0,1)]=-2; b_mat[`BI(0,2)]=-3; b_mat[`BI(0,3)]=-4;
        b_mat[`BI(1,0)]=-5; b_mat[`BI(1,1)]=-6; b_mat[`BI(1,2)]=-7; b_mat[`BI(1,3)]=-8;
        b_mat[`BI(2,0)]=1;  b_mat[`BI(2,1)]=2;  b_mat[`BI(2,2)]=3;  b_mat[`BI(2,3)]=4;
        b_mat[`BI(3,0)]=0;  b_mat[`BI(3,1)]=0;  b_mat[`BI(3,2)]=0;  b_mat[`BI(3,3)]=0;
        zero_c();
        run_and_check("Negative values");
    endtask

    // ── Test 5: Back-to-back ──────────────────────────────────────────────────
    task automatic test_back_to_back();
        integer i, k, j;
        for (i = 0; i < M; i++)
            for (k = 0; k < K; k++) begin
                a_mat[`AI(i,k)] = (i == k) ? 2 : 0;
                b_mat[`BI(i,k)] = (i == k) ? 3 : 0;
            end
        zero_c();
        run_and_check("Back-to-back #1: 2I x 3I = 6I");

        for (i = 0; i < M; i++)
            for (k = 0; k < K; k++)
                a_mat[`AI(i,k)] = i + 1;
        for (k = 0; k < K; k++)
            for (j = 0; j < N; j++)
                b_mat[`BI(k,j)] = j + 1;
        zero_c();
        run_and_check("Back-to-back #2: outer product");
    endtask

    // ── Main ──────────────────────────────────────────────────────────────────
    initial begin
        integer ii;
        $display("=== Tensor Core Testbench ===");
        $display("Config: M=%0d K=%0d N=%0d  DATA_W=%0d ACC_W=%0d",
                 M, K, N, DATA_W, ACC_W);

        rst_n = 0; start = 0;
        for (ii = 0; ii < M*K; ii++) a_mat[ii] = '0;
        for (ii = 0; ii < K*N; ii++) b_mat[ii] = '0;
        for (ii = 0; ii < M*N; ii++) c_mat[ii] = '0;
        repeat (4) @(posedge clk);
        rst_n = 1;
        repeat (2) @(posedge clk);

        test_identity();
        test_ones();
        print_result("Ones result");
        test_arbitrary();
        test_negative();
        test_back_to_back();

        $display("=== All tests complete ===");
        $finish;
    end

    initial begin #500000; $display("TIMEOUT"); $finish; end

    initial begin
        $dumpfile("sim/tensor_core.vcd");
        $dumpvars(0, tensor_core_tb);
    end

endmodule
