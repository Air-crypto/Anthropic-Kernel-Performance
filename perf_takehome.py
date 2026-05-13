"""
# Anthropic's Original Performance Engineering Take-home (Release version)

Copyright Anthropic PBC 2026. Permission is granted to modify and use, but not
to publish or redistribute your solutions so it's hard to find spoilers.

# Task

- Optimize the kernel (in KernelBuilder.build_kernel) as much as possible in the
  available time, as measured by test_kernel_cycles on a frozen separate copy
  of the simulator.

Validate your results using `python tests/submission_tests.py` without modifying
anything in the tests/ folder.

We recommend you look through problem.py next.
"""

import random
import unittest
from collections import defaultdict

from problem import (
    Engine,
    DebugInfo,
    SLOT_LIMITS,
    VLEN,
    N_CORES,
    SCRATCH_SIZE,
    Machine,
    Tree,
    Input,
    HASH_STAGES,
    reference_kernel,
    build_mem_image,
    reference_kernel2,
)

def _vec_range(base: int, length: int = VLEN) -> range:
    return range(base, base + length)


def _slot_rw(engine: str, slot: tuple) -> tuple[list[int], list[int]]:
    """Extract which registers are read/written by an instruction"""
    reads: list[int] = []
    writes: list[int] = []

    if engine == "alu":
        _op, dest, a1, a2 = slot
        reads = [a1, a2]
        writes = [dest]

    elif engine == "valu":
        op = slot[0]
        if op == "vbroadcast":
            _, dest, src = slot
            reads = [src]
            writes = list(_vec_range(dest))
        elif op == "multiply_add":
            _, dest, a, b, c = slot
            reads = list(_vec_range(a)) + list(_vec_range(b)) + list(_vec_range(c))
            writes = list(_vec_range(dest))
        else:
            _, dest, a1, a2 = slot
            reads = list(_vec_range(a1)) + list(_vec_range(a2))
            writes = list(_vec_range(dest))

    elif engine == "load":
        op = slot[0]
        if op == "load":
            _, dest, addr = slot
            reads = [addr]
            writes = [dest]
        elif op == "vload":
            _, dest, addr = slot
            reads = [addr]
            writes = list(_vec_range(dest))
        elif op == "const":
            _, dest, _val = slot
            writes = [dest]
        elif op == "load_offset":
            _, dest, addr, _lane = slot
            reads = [addr]
            writes = [dest]
        else:
            raise NotImplementedError(f"Unknown load op {slot}")

    elif engine == "store":
        op = slot[0]
        if op == "store":
            _, addr, src = slot
            reads = [addr, src]
        elif op == "vstore":
            _, addr, src = slot
            reads = [addr] + list(_vec_range(src))
        else:
            raise NotImplementedError(f"Unknown store op {slot}")

    elif engine == "flow":
        op = slot[0]
        if op == "select":
            _, dest, cond, a, b = slot
            reads = [cond, a, b]
            writes = [dest]
        elif op == "add_imm":
            _, dest, a, _imm = slot
            reads = [a]
            writes = [dest]
        elif op == "vselect":
            _, dest, cond, a, b = slot
            reads = list(_vec_range(cond)) + list(_vec_range(a)) + list(_vec_range(b))
            writes = list(_vec_range(dest))
        elif op in (
            "halt", "pause", "trace_write", "jump", "jump_indirect",
            "cond_jump", "cond_jump_rel", "coreid"
        ):
            pass
        else:
            pass

    elif engine == "debug":
        pass

    else:
        raise NotImplementedError(f"Unknown engine {engine}")

    return reads, writes


def _schedule_slots(slots: list[tuple[str, tuple]]) -> list[dict[str, list[tuple]]]:
    """Pack instructions into VLIW bundles respecting deps and slot limits"""
    cycles: list[dict[str, list[tuple]]] = []
    usage: list[dict[str, int]] = []

    # track when each register becomes ready after being written
    readyT = defaultdict(int)
    lastW = defaultdict(lambda: -1)
    lastR = defaultdict(lambda: -1)

    def ensure_cycle(c: int) -> None:
        while len(cycles) <= c:
            cycles.append({})
            usage.append(defaultdict(int))

    def find_cycle(engine: str, earliest: int) -> int:
        """Find first cycle that has engine capacity and respects deps"""
        limit = SLOT_LIMITS.get(engine, 1)
        c = earliest
        while True:
            ensure_cycle(c)
            if usage[c][engine] < limit:
                return c
            c += 1

    for engine, slot in slots:
        rd, wr = _slot_rw(engine, slot)

        # respect RAW and WAW hazards
        earliest = 0
        for a in rd:
            earliest = max(earliest, readyT[a])
        for a in wr:
            earliest = max(earliest, lastW[a] + 1, lastR[a])

        c = find_cycle(engine, earliest)
        ensure_cycle(c)
        cycles[c].setdefault(engine, []).append(slot)
        usage[c][engine] += 1

        # update hazard tracking
        for a in rd:
            if lastR[a] < c:
                lastR[a] = c
        for a in wr:
            lastW[a] = c
            readyT[a] = c + 1

    return [bundle for bundle in cycles if bundle]


class KernelBuilder:
    # memory layout constants
    MEM_HEADER_SIZE = 7
    
    def __init__(self):
        self.instrs: list[dict[str, list[tuple]]] = []
        self.scratch: dict[str, int] = {}
        self.scratch_debug: dict[int, tuple[str, int]] = {}
        self.sptr = 0
        # cache const/vconst addresses to avoid redundant allocations
        self.const_map: dict[int, int] = {}
        self.vconst_map: dict[int, int] = {}

    def debug_info(self):
        return DebugInfo(scratch_map=self.scratch_debug)

    def alloc_scratch(self, name: str | None = None, length: int = 1) -> int:
        addr = self.sptr
        if name is not None:
            self.scratch[name] = addr
            self.scratch_debug[addr] = (name, length)
        self.sptr += length
        assert self.sptr <= SCRATCH_SIZE, "Out of scratch space"
        return addr

    def alloc_vec(self, name: str | None = None) -> int:
        return self.alloc_scratch(name, VLEN)

    def build_kernel(
        self,
        forest_height: int,
        n_nodes: int,
        batch_size: int,
        rounds: int,
        group_size: int = 16,
        round_tile: int = 13,
        CUT: int = 3,
    ):
        self.instrs = []

        instruction_list: list[tuple[str, tuple]] = []

        # temp regs for address computation
        addr_tmp1  = self.alloc_scratch("tmp_addr")
        addr_tmp2 = self.alloc_scratch("tmp_addr2")

        # memory layout: [header][forest values][indices][values]
        mem_forest_base = self.MEM_HEADER_SIZE
        mem_indices_base = self.MEM_HEADER_SIZE + n_nodes
        mem_values_base = self.MEM_HEADER_SIZE + n_nodes + batch_size

        # pointers to main data structures
        ptr_forest = self.alloc_scratch("forest_values_p")
        ptr_indices = self.alloc_scratch("inp_indices_p")
        ptr_values = self.alloc_scratch("inp_values_p")

        # commonly used scalar constants
        scalar_consts: dict[int, int] = {}
        base_consts = [0, 1, 2, 3, 4, 7, VLEN]
        for val in base_consts:
            scalar_consts[val] = self.alloc_scratch(f"c_{val}")
            self.const_map[val] = scalar_consts[val]

        def get_or_alloc_const(val: int) -> int:
            if val not in scalar_consts:
                scalar_consts[val] = self.alloc_scratch(f"c_{val}")
                self.const_map[val] = scalar_consts[val]
            return scalar_consts[val]

        # vector broadcasts of common constants
        vector_consts: dict[int, int] = {}
        base_vec_vals = [1, 2, 3, 4, 7]
        for val in base_vec_vals:
            vector_consts[val] = self.alloc_vec(f"v_{val}")
            self.vconst_map[val] = vector_consts[val]

        def get_or_alloc_vconst(val: int) -> int:
            if val not in vector_consts:
                vector_consts[val] = self.alloc_vec(f"v_{val}")
                self.vconst_map[val] = vector_consts[val]
            return vector_consts[val]

        # broadcast forest base pointer to all lanes
        vec_forest_ptr = self.alloc_vec("v_forest_p")

        # preload tree nodes up to depth CUT for faster access
        effective_cut = min(CUT, forest_height)
        num_preload = max(1, (1 << (effective_cut + 1)) - 1)

        preloaded_nodes_scalar: list[int] = []
        preloaded_nodes_vec: list[int] = []
        for idx in range(num_preload):
            preloaded_nodes_scalar.append(self.alloc_scratch(f"node_{idx}"))
            preloaded_nodes_vec.append(self.alloc_vec(f"v_node_{idx}"))
            get_or_alloc_const(idx)

        # optimize hash stages - detect multiply_add opportunities
        hash_const1_vecs: list[int] = []
        hash_const3_vecs: list[int | None] = []
        hash_mul_vecs: list[int | None] = []

        for (op1, val1, op2, op3, val3) in HASH_STAGES:
            get_or_alloc_const(val1)
            get_or_alloc_vconst(val1)
            hash_const1_vecs.append(get_or_alloc_vconst(val1))

            # detect pattern: (x + A) + (x << B) = x * (1 + 2^B) + A
            can_fuse = (op1 == "+") and (op2 == "+") and (op3 == "<<")
            if can_fuse:
                fused_multiplier = 1 + (1 << val3)
                get_or_alloc_const(fused_multiplier)
                get_or_alloc_vconst(fused_multiplier)
                hash_mul_vecs.append(get_or_alloc_vconst(fused_multiplier))
                hash_const3_vecs.append(None)
            else:
                get_or_alloc_const(val3)
                get_or_alloc_vconst(val3)
                hash_mul_vecs.append(None)
                hash_const3_vecs.append(get_or_alloc_vconst(val3))

        assert batch_size % VLEN == 0
        num_vec_blocks = batch_size // VLEN
        
        # scratch space for indices and values (working set)
        scratch_idx_base = self.alloc_scratch("idx_scratch", batch_size)
        scratch_val_base = self.alloc_scratch("val_scratch", batch_size)

        load_offset_reg = self.alloc_scratch("offset")

        # figure out max group size based on available scratch
        regs_per_context = 5  # each context needs 5 vector registers
        context_scratch_cost = regs_per_context * VLEN
        scratch_headroom = 64
        max_concurrent_groups = max(1, (SCRATCH_SIZE - self.sptr - scratch_headroom) // context_scratch_cost)
        actual_group_size = min(group_size, max_concurrent_groups)

        # allocate temp registers for each batch element being processed
        context_regs = []
        for g_idx in range(actual_group_size):
            context_regs.append(
                {
                    "node": self.alloc_vec(f"ctx{g_idx}_node"),
                    "tmp1": self.alloc_vec(f"ctx{g_idx}_tmp1"),
                    "tmp2": self.alloc_vec(f"ctx{g_idx}_tmp2"),
                    "tmp3": self.alloc_vec(f"ctx{g_idx}_tmp3"),
                    "tmp4": self.alloc_vec(f"ctx{g_idx}_tmp4"),
                }
            )

        # load base pointers
        instruction_list.append(("load", ("const", ptr_forest, mem_forest_base)))
        instruction_list.append(("load", ("const", ptr_indices, mem_indices_base)))
        instruction_list.append(("load", ("const", ptr_values, mem_values_base)))

        # load remaining constants (skip 0-4 and pointers)
        small_const_range = range(5)
        for val, addr in scalar_consts.items():
            if val in small_const_range:
                continue
            is_pointer = addr in (ptr_forest, ptr_indices, ptr_values)
            if not is_pointer:
                instruction_list.append(("load", ("const", addr, val)))

        # build small constants via ALU to save load slots
        reg_zero = scalar_consts[0]
        reg_one = scalar_consts[1]
        reg_two = scalar_consts[2]
        reg_three = scalar_consts[3]
        reg_four = scalar_consts[4]
        instruction_list.append(("flow", ("add_imm", reg_one, reg_zero, 1))) 
        instruction_list.append(("alu",  ("+", reg_two, reg_one, reg_one))) 
        instruction_list.append(("alu",  ("+", reg_three, reg_two, reg_one))) 
        instruction_list.append(("alu",  ("+", reg_four, reg_two, reg_two))) 

        # broadcast constants to vector registers
        instruction_list.append(("valu", ("vbroadcast", vec_forest_ptr, ptr_forest)))
        for val, vec_addr in vector_consts.items():
            instruction_list.append(("valu", ("vbroadcast", vec_addr, scalar_consts[val])))

        # preload tree nodes for faster lookup (cut optimization)
        for node_id in range(num_preload):
            # alternate between temp address registers
            use_first_temp = (node_id % 2 == 0)
            temp_reg = addr_tmp1 if use_first_temp else addr_tmp2
            instruction_list.append(("alu",  ("+", temp_reg, ptr_forest, scalar_consts[node_id])))
            instruction_list.append(("load", ("load", preloaded_nodes_scalar[node_id], temp_reg)))
            instruction_list.append(("valu", ("vbroadcast", preloaded_nodes_vec[node_id], preloaded_nodes_scalar[node_id])))

        # vectorized loads for all batch elements
        stride_const = scalar_consts[VLEN]
        for blk_idx in range(num_vec_blocks):
            instruction_list.append(("alu",  ("+", addr_tmp1, ptr_indices, load_offset_reg)))
            instruction_list.append(("load", ("vload", scratch_idx_base + blk_idx * VLEN, addr_tmp1)))
            instruction_list.append(("alu",  ("+", addr_tmp1, ptr_values, load_offset_reg)))
            instruction_list.append(("load", ("vload", scratch_val_base + blk_idx * VLEN, addr_tmp1)))
            instruction_list.append(("alu",  ("+", load_offset_reg, load_offset_reg, stride_const)))

        # cache frequently used vector constants
        vec_one = vector_consts[1]
        vec_two = vector_consts[2]
        vec_three = vector_consts[3]
        vec_four = vector_consts[4]
        vec_seven = vector_consts[7]
        scalar_one = scalar_consts[1]

        def xor_node_into_hash(val_vec: int, node_vec: int) -> None:
            """XOR node value into hash (lane by lane since we need scalar xor)"""
            for lane_id in range(VLEN):
                instruction_list.append(("alu", ("^", val_vec + lane_id, val_vec + lane_id, node_vec + lane_id)))

        # group batches together, tile rounds for better locality
        for group_offset in range(0, num_vec_blocks, actual_group_size):
            for round_offset in range(0, rounds, round_tile):
                round_limit = min(rounds, round_offset + round_tile)

                for group_idx in range(actual_group_size):
                    block_idx = group_offset + group_idx
                    # skip if we've exceeded batch size
                    if block_idx >= num_vec_blocks:
                        break

                    ctx = context_regs[group_idx]
                    idx_vec_base = scratch_idx_base + block_idx * VLEN
                    val_vec_base = scratch_val_base + block_idx * VLEN

                    for round_idx in range(round_offset, round_limit):
                        tree_level = round_idx % (forest_height + 1)

                        # level 0: just XOR with root node
                        if tree_level == 0:
                            xor_node_into_hash(val_vec_base, preloaded_nodes_vec[0])

                        # level 1: select between nodes 1 and 2 based on LSB
                        elif tree_level == 1 and effective_cut >= 1:
                            instruction_list.append(("valu", ("&", ctx["tmp1"], idx_vec_base, vec_one)))
                            instruction_list.append(("flow", ("vselect", ctx["node"], ctx["tmp1"], preloaded_nodes_vec[1], preloaded_nodes_vec[2])))
                            xor_node_into_hash(val_vec_base, ctx["node"])

                        # level 2: 4-way select using bits [0:1]
                        elif tree_level == 2 and effective_cut >= 2:
                            instruction_list.append(("valu", ("-", ctx["tmp1"], idx_vec_base, vec_three)))
                            instruction_list.append(("valu", ("&", ctx["tmp2"], ctx["tmp1"], vec_one)))
                            instruction_list.append(("valu", ("&", ctx["node"], ctx["tmp1"], vec_two)))

                            instruction_list.append(("flow", ("vselect", ctx["tmp1"], ctx["tmp2"], preloaded_nodes_vec[4], preloaded_nodes_vec[3])))
                            instruction_list.append(("flow", ("vselect", ctx["tmp2"], ctx["tmp2"], preloaded_nodes_vec[6], preloaded_nodes_vec[5])))
                            instruction_list.append(("flow", ("vselect", ctx["node"], ctx["node"], ctx["tmp2"], ctx["tmp1"])))
                            xor_node_into_hash(val_vec_base, ctx["node"])

                        # level 3: 8-way select using bits [0:2]
                        elif tree_level == 3 and effective_cut >= 3:
                            instruction_list.append(("valu", ("-", ctx["tmp1"], idx_vec_base, vec_seven)))
                            instruction_list.append(("valu", ("&", ctx["tmp2"], ctx["tmp1"], vec_one)))
                            instruction_list.append(("valu", ("&", ctx["tmp3"], ctx["tmp1"], vec_two)))
                            instruction_list.append(("valu", ("&", ctx["tmp4"], ctx["tmp1"], vec_four)))

                            # nested selects to build 8-way mux
                            instruction_list.append(("flow", ("vselect", ctx["node"], ctx["tmp2"], preloaded_nodes_vec[8],  preloaded_nodes_vec[7])))
                            instruction_list.append(("flow", ("vselect", ctx["tmp1"], ctx["tmp2"], preloaded_nodes_vec[10], preloaded_nodes_vec[9])))
                            instruction_list.append(("flow", ("vselect", ctx["tmp1"], ctx["tmp3"], ctx["tmp1"], ctx["node"])))

                            instruction_list.append(("flow", ("vselect", ctx["node"], ctx["tmp2"], preloaded_nodes_vec[12], preloaded_nodes_vec[11])))
                            instruction_list.append(("flow", ("vselect", ctx["tmp2"], ctx["tmp2"], preloaded_nodes_vec[14], preloaded_nodes_vec[13])))
                            instruction_list.append(("flow", ("vselect", ctx["node"], ctx["tmp3"], ctx["tmp2"], ctx["node"])))

                            instruction_list.append(("flow", ("vselect", ctx["node"], ctx["tmp4"], ctx["node"], ctx["tmp1"])))
                            xor_node_into_hash(val_vec_base, ctx["node"])

                        # deeper levels: load from memory (can't preload everything)
                        else:
                            for lane_id in range(VLEN):
                                instruction_list.append(("alu", ("+", ctx["tmp1"] + lane_id, vec_forest_ptr + lane_id, idx_vec_base + lane_id)))
                            for lane_id in range(VLEN):
                                instruction_list.append(("load", ("load", ctx["node"] + lane_id, ctx["tmp1"] + lane_id)))
                            xor_node_into_hash(val_vec_base, ctx["node"])

                        # apply hash function (use multiply_add when possible)
                        for hash_idx, (op1, _val1, op2, op3, _val3) in enumerate(HASH_STAGES):
                            fused_mul_vec = hash_mul_vecs[hash_idx]
                            # check if we can use fused multiply-add
                            if fused_mul_vec is not None:
                                instruction_list.append(("valu", ("multiply_add", val_vec_base, val_vec_base, fused_mul_vec, hash_const1_vecs[hash_idx])))
                            else:
                                # unfused version
                                instruction_list.append(("valu", (op1, ctx["tmp1"], val_vec_base, hash_const1_vecs[hash_idx])))
                                instruction_list.append(("valu", (op3, ctx["tmp2"], val_vec_base, hash_const3_vecs[hash_idx])))
                                instruction_list.append(("valu", (op2, val_vec_base, ctx["tmp1"], ctx["tmp2"])))

                        # update index for next level
                        at_leaf_level = (tree_level == forest_height)
                        if at_leaf_level:
                            # at leaves, reset to 0
                            instruction_list.append(("valu", ("^", idx_vec_base, idx_vec_base, idx_vec_base)))
                        else:
                            # navigate: idx = (idx * 2) + (hash & 1) + 1
                            for lane_id in range(VLEN):
                                instruction_list.append(("alu", ("&", ctx["tmp1"] + lane_id, val_vec_base + lane_id, scalar_one)))
                                instruction_list.append(("alu", ("+", ctx["node"] + lane_id, ctx["tmp1"] + lane_id, scalar_one)))
                            instruction_list.append(("valu", ("multiply_add", idx_vec_base, idx_vec_base, vec_two, ctx["node"])))

        # write back updated values
        instruction_list.append(("flow", ("add_imm", addr_tmp1, ptr_values, 0)))
        for blk_idx in range(num_vec_blocks):
            instruction_list.append(("store", ("vstore", addr_tmp1, scratch_val_base + blk_idx * VLEN)))
            is_not_last = (blk_idx != num_vec_blocks - 1)
            if is_not_last:
                instruction_list.append(("alu", ("+", addr_tmp1, addr_tmp1, stride_const)))

        # write back updated indices
        instruction_list.append(("flow", ("add_imm", addr_tmp1, ptr_indices, 0)))
        for blk_idx in range(num_vec_blocks):
            instruction_list.append(("store", ("vstore", addr_tmp1, scratch_idx_base + blk_idx * VLEN)))
            is_not_last = (blk_idx != num_vec_blocks - 1)
            if is_not_last:
                instruction_list.append(("alu", ("+", addr_tmp1, addr_tmp1, stride_const)))

        # pack instructions into VLIW bundles
        scheduled_bundles = _schedule_slots(instruction_list)
        self.instrs.append({"flow": [("pause",)]})
        self.instrs.extend(scheduled_bundles)
        self.instrs.append({"flow": [("pause",)]})

BASELINE = 147734

def do_kernel_test(
    forest_height: int,
    rounds: int,
    batch_size: int,
    seed: int = 123,
    trace: bool = False,
    prints: bool = False,
):
    print(f"{forest_height=}, {rounds=}, {batch_size=}")
    random.seed(seed)
    forest = Tree.generate(forest_height)
    inp = Input.generate(forest, batch_size, rounds)
    mem = build_mem_image(forest, inp)

    kb = KernelBuilder()
    kb.build_kernel(forest.height, len(forest.values), len(inp.indices), rounds)
    # print(kb.instrs)

    value_trace = {}
    machine = Machine(
        mem,
        kb.instrs,
        kb.debug_info(),
        n_cores=N_CORES,
        value_trace=value_trace,
        trace=trace,
    )
    machine.prints = prints
    for i, ref_mem in enumerate(reference_kernel2(mem, value_trace)):
        machine.run()
        inp_values_p = ref_mem[6]
        if prints:
            print(machine.mem[inp_values_p : inp_values_p + len(inp.values)])
            print(ref_mem[inp_values_p : inp_values_p + len(inp.values)])
        assert (
            machine.mem[inp_values_p : inp_values_p + len(inp.values)]
            == ref_mem[inp_values_p : inp_values_p + len(inp.values)]
        ), f"Incorrect result on round {i}"
        inp_indices_p = ref_mem[5]
        if prints:
            print(machine.mem[inp_indices_p : inp_indices_p + len(inp.indices)])
            print(ref_mem[inp_indices_p : inp_indices_p + len(inp.indices)])
        # Updating these in memory isn't required, but you can enable this check for debugging
        # assert machine.mem[inp_indices_p:inp_indices_p+len(inp.indices)] == ref_mem[inp_indices_p:inp_indices_p+len(inp.indices)]

    print("CYCLES: ", machine.cycle)
    print("Speedup over baseline: ", BASELINE / machine.cycle)
    return machine.cycle


class Tests(unittest.TestCase):
    def test_ref_kernels(self):
        """
        Test the reference kernels against each other
        """
        random.seed(123)
        for i in range(10):
            f = Tree.generate(4)
            inp = Input.generate(f, 10, 6)
            mem = build_mem_image(f, inp)
            reference_kernel(f, inp)
            for _ in reference_kernel2(mem, {}):
                pass
            assert inp.indices == mem[mem[5] : mem[5] + len(inp.indices)]
            assert inp.values == mem[mem[6] : mem[6] + len(inp.values)]

    def test_kernel_trace(self):
        # Full-scale example for performance testing
        do_kernel_test(10, 16, 256, trace=True, prints=False)

    # Passing this test is not required for submission, see submission_tests.py for the actual correctness test
    # You can uncomment this if you think it might help you debug
    # def test_kernel_correctness(self):
    #     for batch in range(1, 3):
    #         for forest_height in range(3):
    #             do_kernel_test(
    #                 forest_height + 2, forest_height + 4, batch * 16 * VLEN * N_CORES
    #             )

    def test_kernel_cycles(self):
        do_kernel_test(10, 16, 256)


# To run all the tests:
#    python perf_takehome.py
# To run a specific test:
#    python perf_takehome.py Tests.test_kernel_cycles
# To view a hot-reloading trace of all the instructions:  **Recommended debug loop**
# NOTE: The trace hot-reloading only works in Chrome. In the worst case if things aren't working, drag trace.json onto https://ui.perfetto.dev/
#    python perf_takehome.py Tests.test_kernel_trace
# Then run `python watch_trace.py` in another tab, it'll open a browser tab, then click "Open Perfetto"
# You can then keep that open and re-run the test to see a new trace.

# To run the proper checks to see which thresholds you pass:
#    python tests/submission_tests.py

if __name__ == "__main__":
    unittest.main()