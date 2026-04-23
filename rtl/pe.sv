// MAC Processing Element — unchanged, scalar ports work fine in iverilog
module pe #(
    parameter int DATA_W = 8,
    parameter int ACC_W  = 32
)(
    input  logic                     clk,
    input  logic                     rst_n,
    input  logic                     en,
    input  logic                     clear,
    input  logic signed [DATA_W-1:0] a_in,
    input  logic signed [DATA_W-1:0] b_in,
    output logic signed [DATA_W-1:0] a_out,
    output logic signed [DATA_W-1:0] b_out,
    output logic signed [ACC_W-1:0]  acc
);
    always_ff @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            acc   <= '0;
            a_out <= '0;
            b_out <= '0;
        end else if (clear) begin
            acc   <= '0;
            a_out <= '0;
            b_out <= '0;
        end else if (en) begin
            acc   <= acc + ACC_W'(signed'(a_in)) * ACC_W'(signed'(b_in));
            a_out <= a_in;
            b_out <= b_in;
        end
    end
endmodule
