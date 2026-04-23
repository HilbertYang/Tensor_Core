CXX      = g++
CXXFLAGS = -std=c++17 -O2

TOP      = tensor_core
RTL_SRCS = rtl/pe.sv rtl/systolic_array.sv rtl/tensor_core.sv
TB_SRC   = tb/sim_main.cpp
SIM_DIR  = sim/verilator
SIM_BIN  = sim/tensor_core_sim

# ── Verilator simulation ──────────────────────────────────────────────────────
sim: $(SIM_BIN)

$(SIM_BIN): $(RTL_SRCS) $(TB_SRC)
	mkdir -p sim
	verilator --cc --exe --build --sv \
	    -Wall --Wno-WIDTHTRUNC --Wno-WIDTHEXPAND \
	    --Wno-UNUSEDSIGNAL --Wno-UNUSEDPARAM \
	    --top-module $(TOP) \
	    --Mdir $(SIM_DIR) \
	    -CFLAGS "$(CXXFLAGS)" \
	    $(RTL_SRCS) $(TB_SRC)
	cp $(SIM_DIR)/V$(TOP) $(SIM_BIN)

run: $(SIM_BIN)
	./$(SIM_BIN)

# ── iverilog (reference, limited SV support) ──────────────────────────────────
iverilog_sim:
	mkdir -p sim
	iverilog -g2012 -o sim/iverilog_sim \
	    $(RTL_SRCS) tb/tensor_core_tb.sv
	./sim/iverilog_sim

clean:
	rm -rf sim

.PHONY: sim run iverilog_sim clean
