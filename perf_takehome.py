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

from problem import (
    HASH_STAGES,
    N_CORES,
    SCRATCH_SIZE,
    VLEN,
    DebugInfo,
    Engine,
    Input,
    Machine,
    Tree,
    build_mem_image,
    reference_kernel,
    reference_kernel2,
)
from utils import InstructionRecord, add_to_record, get_addresses


class KernelBuilder:
    def __init__(self):
        self.instrs = []
        self.scratch = {}
        self.scratch_debug = {}
        self.scratch_ptr = 0
        self.const_map = {}
        self.const_vec_map = {}

    def debug_info(self):
        return DebugInfo(scratch_map=self.scratch_debug)

    def build(self, slots: list[tuple[Engine, tuple]], vliw: bool = True):
        instructions_book: list[InstructionRecord] = []
        current = InstructionRecord()

        for engine, slot in slots:
            srcs, dsts = get_addresses(engine, slot)

            # if the current instruciton has any instruciton which writes to X, we cannot have another
            # instruction which takes that X value, as it updates live.
            # `and not current.has_any_src(dst)` does not matter as the instruction will not update that value.

            # if engine == "debug":
            #     # instructions_book.append(current)
            #     # current = InstructionRecord()

            #     # current.instruction = {"debug": [slot]}  # pyright: ignore[reportAttributeAccessIssue]

            #     # instructions_book.append(current)
            #     # current = InstructionRecord()

            #     continue

            # # add to previous
            # if len(instructions_book) > 0:
            #     prev = instructions_book[-1]

            #     # conflicts
            #     if not prev.has_any_dst(srcs):
            #         if add_to_record(prev, engine, slot, srcs, dsts):
            #             continue

            # add to current
            if add_to_record(current, engine, slot, srcs, dsts):
                continue

            # new record
            instructions_book.append(current)
            current = InstructionRecord()

            add_to_record(current, engine, slot, srcs, dsts)

        if current.instruction:
            instructions_book.append(current)

        # return orig
        return [record.instruction for record in instructions_book]

    def add(self, engine, slot):
        self.instrs.append({engine: [slot]})

    def alloc_scratch(self, name=None, length=1):
        addr = self.scratch_ptr
        if name is not None:
            self.scratch[name] = addr
            self.scratch_debug[addr] = (name, length)
        self.scratch_ptr += length
        assert self.scratch_ptr <= SCRATCH_SIZE, "Out of scratch space"
        return addr

    def scratch_const(self, val, name=None):
        if val not in self.const_map:
            addr = self.alloc_scratch(name)
            self.add("load", ("const", addr, val))
            self.const_map[val] = addr
        return self.const_map[val]

    def clone_scalar_to_vec(self, src_addr, name=None):
        dest = self.alloc_scratch(name, VLEN)
        self.add("valu", ("vbroadcast", dest, src_addr))
        return dest

    def scratch_const_vec(self, val, name=None):
        if val not in self.const_vec_map:
            scalar_addr = self.scratch_const(val)
            self.const_vec_map[val] = self.clone_scalar_to_vec(scalar_addr, name)
        return self.const_vec_map[val]

    def build_hash(self, val_hash_addr, tmp1, tmp2, round, i, parallel=False):
        slots = []

        engine = "valu" if parallel else "alu"
        const_fn = self.scratch_const_vec if parallel else self.scratch_const
        for hi, (op1, val1, op2, op3, val3) in enumerate(HASH_STAGES):
            slots.append((engine, (op1, tmp1, val_hash_addr, const_fn(val1))))
            slots.append((engine, (op3, tmp2, val_hash_addr, const_fn(val3))))
            slots.append((engine, (op2, val_hash_addr, tmp1, tmp2)))

            slots.append(
                ("debug", ("compare", val_hash_addr, (round, i, "hash_stage", hi)))
            )

        return slots

    def build_kernel(
        self, forest_height: int, n_nodes: int, batch_size: int, rounds: int
    ):
        NUM_STREAMS = 3

        # per-stream scratch registers
        tmp1s = [self.alloc_scratch(f"tmp1_{s}", VLEN) for s in range(NUM_STREAMS)]
        tmp2s = [self.alloc_scratch(f"tmp2_{s}", VLEN) for s in range(NUM_STREAMS)]
        tmp3s = [self.alloc_scratch(f"tmp3_{s}", VLEN) for s in range(NUM_STREAMS)]

        init_vars = [
            "rounds",
            "n_nodes",
            "batch_size",
            "forest_height",
            "forest_values_p",
            "inp_indices_p",
            "inp_values_p",
        ]

        for v in init_vars:
            self.alloc_scratch(v, 1)

        for i, v in enumerate(init_vars):
            self.add("load", ("const", tmp1s[0], i))
            self.add("load", ("load", self.scratch[v], tmp1s[0]))

        n_nodes_v = self.clone_scalar_to_vec(self.scratch["n_nodes"], "n_nodes_v")
        forest_values_p_v = self.clone_scalar_to_vec(
            self.scratch["forest_values_p"], "forest_values_p_v"
        )
        inp_indices_p_v = self.clone_scalar_to_vec(
            self.scratch["inp_indices_p"], "inp_indices_p_v"
        )
        inp_values_p_v = self.clone_scalar_to_vec(
            self.scratch["inp_values_p"], "inp_values_p_v"
        )

        zero_const = self.scratch_const_vec(0)
        one_const = self.scratch_const_vec(1)
        two_const = self.scratch_const_vec(2)

        self.add("flow", ("pause",))
        self.add("debug", ("comment", "Starting loop"))

        body = []

        tmp_idxs = [self.alloc_scratch(f"tmp_idx_{s}", VLEN) for s in range(NUM_STREAMS)]
        tmp_vals = [self.alloc_scratch(f"tmp_val_{s}", VLEN) for s in range(NUM_STREAMS)]
        tmp_node_vals = [self.alloc_scratch(f"tmp_node_val_{s}", VLEN) for s in range(NUM_STREAMS)]
        tmp_addrs = [self.alloc_scratch(f"tmp_addr_{s}", VLEN) for s in range(NUM_STREAMS)]
        tmp_indices_ps = [self.alloc_scratch(f"tmp_indices_p_{s}", VLEN) for s in range(NUM_STREAMS)]
        tmp_values_ps = [self.alloc_scratch(f"tmp_values_p_{s}", VLEN) for s in range(NUM_STREAMS)]

        for round in range(rounds):
            for i in range(0, batch_size, VLEN * NUM_STREAMS):
                n_streams = min(NUM_STREAMS, (batch_size - i) // VLEN)
                streams = range(n_streams)

                # compute address pointers for all streams
                for s in streams:
                    ic = self.scratch_const_vec(i + s * VLEN)
                    body.append(("valu", ("+", tmp_indices_ps[s], inp_indices_p_v, ic)))
                for s in streams:
                    ic = self.scratch_const_vec(i + s * VLEN)
                    body.append(("valu", ("+", tmp_values_ps[s], inp_values_p_v, ic)))

                # load indices
                for s in streams:
                    body.append(("load", ("vload", tmp_idxs[s], tmp_indices_ps[s])))
                    body.append(("debug", ("compare", tmp_idxs[s], (round, i + s * VLEN, "idx"))))

                # load values
                for s in streams:
                    body.append(("load", ("vload", tmp_vals[s], tmp_values_ps[s])))
                    body.append(("debug", ("compare", tmp_vals[s], (round, i + s * VLEN, "val"))))

                # node_val = mem[forest_values_p + idx]
                for s in streams:
                    body.append(("valu", ("+", tmp_addrs[s], forest_values_p_v, tmp_idxs[s])))
                for lane in range(VLEN):
                    for s in streams:
                        body.append(("load", ("load_offset", tmp_node_vals[s], tmp_addrs[s], lane)))
                for s in streams:
                    body.append(("debug", ("compare", tmp_node_vals[s], (round, i + s * VLEN, "node_val"))))

                # val = myhash(val ^ node_val)
                for s in streams:
                    body.append(("valu", ("^", tmp_vals[s], tmp_vals[s], tmp_node_vals[s])))
                for hi, (op1, val1, op2, op3, val3) in enumerate(HASH_STAGES):
                    for s in streams:
                        body.append(("valu", (op1, tmp1s[s], tmp_vals[s], self.scratch_const_vec(val1))))
                    for s in streams:
                        body.append(("valu", (op3, tmp2s[s], tmp_vals[s], self.scratch_const_vec(val3))))
                    for s in streams:
                        body.append(("valu", (op2, tmp_vals[s], tmp1s[s], tmp2s[s])))
                    for s in streams:
                        body.append(("debug", ("compare", tmp_vals[s], (round, i + s * VLEN, "hash_stage", hi))))
                for s in streams:
                    body.append(("debug", ("compare", tmp_vals[s], (round, i + s * VLEN, "hashed_val"))))

                # idx = 2*idx + (1 if val % 2 == 0 else 2)
                for s in streams:
                    body.append(("valu", ("%", tmp1s[s], tmp_vals[s], two_const)))
                for s in streams:
                    body.append(("valu", ("==", tmp1s[s], tmp1s[s], zero_const)))
                for s in streams:
                    body.append(("flow", ("vselect", tmp3s[s], tmp1s[s], one_const, two_const)))
                for s in streams:
                    body.append(("valu", ("*", tmp_idxs[s], tmp_idxs[s], two_const)))
                for s in streams:
                    body.append(("valu", ("+", tmp_idxs[s], tmp_idxs[s], tmp3s[s])))
                for s in streams:
                    body.append(("debug", ("compare", tmp_idxs[s], (round, i + s * VLEN, "next_idx"))))

                # idx = 0 if idx >= n_nodes else idx
                for s in streams:
                    body.append(("valu", ("<", tmp1s[s], tmp_idxs[s], n_nodes_v)))
                for s in streams:
                    body.append(("flow", ("vselect", tmp_idxs[s], tmp1s[s], tmp_idxs[s], zero_const)))
                for s in streams:
                    body.append(("debug", ("compare", tmp_idxs[s], (round, i + s * VLEN, "wrapped_idx"))))

                # store results
                for s in streams:
                    body.append(("store", ("vstore", tmp_indices_ps[s], tmp_idxs[s])))
                for s in streams:
                    body.append(("store", ("vstore", tmp_values_ps[s], tmp_vals[s])))

        body_instrs = self.build(body)
        self.instrs.extend(body_instrs)
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
