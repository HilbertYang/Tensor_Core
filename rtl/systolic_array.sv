// Output-stationary systolic array.
// Flat 1-D unpacked arrays (iverilog-safe):
//   a_row[i]    = A row i element for current k
//   b_col[j]    = B col j element for current k
//   result[i*N+j] = accumulated A*B at (i,j)

module systolic_array #(
    parameter int M      = 4,
    parameter int N      = 4,
    parameter int K      = 4,
    parameter int DATA_W = 8,
    parameter int ACC_W  = 32
)(
    input  logic                          clk,
    input  logic                          rst_n,
    input  logic                          en,
    input  logic                          clear,
    input  logic signed [DATA_W-1:0]      a_row [M],    // 1-D: element per row
    input  logic signed [DATA_W-1:0]      b_col [N],    // 1-D: element per col
    output logic signed [ACC_W-1:0]       result [M*N]  // 1-D flat: result[i*N+j]
);

    // Diagonal skew FIFOs.  a_sr[i*M+d] = shift-reg stage d for row i.
    logic signed [DATA_W-1:0] a_sr [M*M];
    logic signed [DATA_W-1:0] b_sr [N*N];

    integer ii, jj, dd;
    always_ff @(posedge clk or negedge rst_n) begin
        if (!rst_n || clear) begin
            for (ii = 0; ii < M*M; ii++) a_sr[ii] <= '0;
            for (jj = 0; jj < N*N; jj++) b_sr[jj] <= '0;
        end else begin
            for (ii = 0; ii < M; ii++) begin
                a_sr[ii*M + 0] <= en ? a_row[ii] : '0;
                for (dd = 1; dd < M; dd++)
                    a_sr[ii*M + dd] <= a_sr[ii*M + dd - 1];
            end
            for (jj = 0; jj < N; jj++) begin
                b_sr[jj*N + 0] <= en ? b_col[jj] : '0;
                for (dd = 1; dd < N; dd++)
                    b_sr[jj*N + dd] <= b_sr[jj*N + dd - 1];
            end
        end
    end

    // PE interconnect: a_wire[i*(N+1)+j], b_wire[i*N+j] (with extra boundary)
    wire signed [DATA_W-1:0] a_wire [M*(N+1)];
    wire signed [DATA_W-1:0] b_wire [(M+1)*N];

    // Skewed inputs: row i delayed by i stages
    genvar gi, gj;
    for (gi = 0; gi < M; gi++) begin : g_ain
        assign a_wire[gi*(N+1) + 0] = a_sr[gi*M + gi];
    end
    for (gj = 0; gj < N; gj++) begin : g_bin
        assign b_wire[0*N + gj] = b_sr[gj*N + gj];
    end

    // PE grid
    wire signed [ACC_W-1:0] pe_acc [M*N];

    for (gi = 0; gi < M; gi++) begin : g_row
        for (gj = 0; gj < N; gj++) begin : g_col
            pe #(.DATA_W(DATA_W), .ACC_W(ACC_W)) u_pe (
                .clk   (clk),   .rst_n (rst_n),
                .en    (en),    .clear (clear),
                .a_in  (a_wire[gi*(N+1) + gj]),
                .b_in  (b_wire[gi*N     + gj]),
                .a_out (a_wire[gi*(N+1) + gj + 1]),
                .b_out (b_wire[(gi+1)*N + gj]),
                .acc   (pe_acc[gi*N + gj])
            );
            assign result[gi*N + gj] = pe_acc[gi*N + gj];
        end
    end

endmodule
