// Tensor Core: D = A*B + C  (M×K * K×N + M×N → M×N)
//
// All matrix ports are flat 1-D unpacked arrays (iverilog-safe).
// Index convention: A[i][k] = a_mat[i*K+k], B[k][j] = b_mat[k*N+j],
//                   C[i][j] = c_mat[i*N+j], D[i][j] = d_mat[i*N+j]
//
// Protocol: pulse start for one cycle with matrices stable.
//           done pulses one cycle; d_mat is valid the SAME cycle as done.

module tensor_core #(
    parameter int M      = 4,
    parameter int N      = 4,
    parameter int K      = 4,
    parameter int DATA_W = 8,
    parameter int ACC_W  = 32
)(
    input  logic                          clk,
    input  logic                          rst_n,
    input  logic                          start,

    input  logic signed [DATA_W-1:0]      a_mat [M*K],
    input  logic signed [DATA_W-1:0]      b_mat [K*N],
    input  logic signed [ACC_W-1:0]       c_mat [M*N],

    output logic signed [ACC_W-1:0]       d_mat [M*N],
    output logic                          done,
    output logic                          busy
);

    // Each element travels through: shift-reg (i or j stages) + PE pipeline
    // (j horizontal or i vertical stages). PE[M-1][N-1] accumulates k=K-1
    // at cycle (K-1) + (M-1) + (N-1) + 1 = K+M+N-2, so need K+M+N-1 cycles.
    localparam int DRAIN  = K + M + N - 1;
    localparam int CNT_W  = $clog2(DRAIN + 2);
    localparam int KPTR_W = $clog2(K + 2);

    // ── Input buffers ─────────────────────────────────────────────────────────
    logic signed [DATA_W-1:0] a_buf [M*K];
    logic signed [DATA_W-1:0] b_buf [K*N];
    logic signed [ACC_W-1:0]  c_buf [M*N];

    // ── FSM ───────────────────────────────────────────────────────────────────
    typedef enum logic [1:0] {
        IDLE    = 2'b00,
        CLEAR   = 2'b01,
        COMPUTE = 2'b10,
        OUTPUT  = 2'b11
    } state_t;

    state_t             state;
    logic [CNT_W-1:0]   cycle_cnt;
    logic [KPTR_W-1:0]  k_ptr;
    logic               sa_en, sa_clear;

    // ── Systolic array I/O ────────────────────────────────────────────────────
    logic signed [DATA_W-1:0] a_row [M];   // column k of A (one element per row)
    logic signed [DATA_W-1:0] b_col [N];   // row k of B    (one element per col)
    logic signed [ACC_W-1:0]  sa_result [M*N];

    systolic_array #(.M(M),.N(N),.K(K),.DATA_W(DATA_W),.ACC_W(ACC_W)) u_sa (
        .clk    (clk), .rst_n (rst_n),
        .en     (sa_en), .clear (sa_clear),
        .a_row  (a_row), .b_col (b_col),
        .result (sa_result)
    );

    // ── Stream mux ────────────────────────────────────────────────────────────
    integer fi, fj;
    always_comb begin
        for (fi = 0; fi < M; fi++)
            a_row[fi] = (k_ptr < KPTR_W'(K)) ? a_buf[fi*K + k_ptr] : '0;
        for (fj = 0; fj < N; fj++)
            b_col[fj] = (k_ptr < KPTR_W'(K)) ? b_buf[k_ptr*N + fj] : '0;
    end

    // ── FSM + output registers ────────────────────────────────────────────────
    integer ii, jj, kk;

    always_ff @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            state     <= IDLE;
            cycle_cnt <= '0;
            k_ptr     <= '0;
            sa_en     <= 1'b0;
            sa_clear  <= 1'b0;
            done      <= 1'b0;
            busy      <= 1'b0;
            for (ii = 0; ii < M*K; ii++) a_buf[ii] <= '0;
            for (ii = 0; ii < K*N; ii++) b_buf[ii] <= '0;
            for (ii = 0; ii < M*N; ii++) c_buf[ii] <= '0;
            for (ii = 0; ii < M*N; ii++) d_mat[ii] <= '0;
        end else begin
            done     <= 1'b0;
            sa_clear <= 1'b0;

            case (state)
                IDLE: begin
                    if (start) begin
                        for (ii = 0; ii < M*K; ii++) a_buf[ii] <= a_mat[ii];
                        for (ii = 0; ii < K*N; ii++) b_buf[ii] <= b_mat[ii];
                        for (ii = 0; ii < M*N; ii++) c_buf[ii] <= c_mat[ii];
                        sa_clear  <= 1'b1;
                        sa_en     <= 1'b0;
                        cycle_cnt <= '0;
                        k_ptr     <= '0;
                        busy      <= 1'b1;
                        state     <= CLEAR;
                    end
                end

                CLEAR: begin
                    sa_en <= 1'b1;
                    k_ptr <= '0;
                    state <= COMPUTE;
                end

                COMPUTE: begin
                    cycle_cnt <= cycle_cnt + 1;
                    if (k_ptr < KPTR_W'(K))
                        k_ptr <= k_ptr + 1;
                    if (cycle_cnt == CNT_W'(DRAIN - 1)) begin
                        sa_en <= 1'b0;
                        state <= OUTPUT;
                    end
                end

                OUTPUT: begin
                    for (ii = 0; ii < M*N; ii++)
                        d_mat[ii] <= sa_result[ii] + c_buf[ii];
                    done  <= 1'b1;
                    busy  <= 1'b0;
                    state <= IDLE;
                end

                default: state <= IDLE;
            endcase
        end
    end

endmodule
