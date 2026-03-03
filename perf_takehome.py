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
        """
        Like reference_kernel2 but building actual instructions.
        Scalar implementation using only scalar ALU and load/store.
        """
        tmp1 = self.alloc_scratch("tmp1", VLEN)
        tmp2 = self.alloc_scratch("tmp2", VLEN)
        tmp3 = self.alloc_scratch("tmp3", VLEN)
        # Scratch space addresses
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
            self.add("load", ("const", tmp1, i))
            self.add("load", ("load", self.scratch[v], tmp1))

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

        # Pause instructions are matched up with yield statements in the reference
        # kernel to let you debug at intermediate steps. The testing harness in this
        # file requires these match up to the reference kernel's yields, but the
        # submission harness ignores them.
        self.add("flow", ("pause",))
        # Any debug engine instruction is ignored by the submission simulator
        self.add("debug", ("comment", "Starting loop"))

        body = []  # array of slots

        # Scalar scratch registers
        tmp_idx = self.alloc_scratch("tmp_idx", VLEN)
        tmp_val = self.alloc_scratch("tmp_val", VLEN)
        tmp_node_val = self.alloc_scratch("tmp_node_val", VLEN)
        tmp_addr = self.alloc_scratch("tmp_addr", VLEN)

        tmp_indices_p = self.alloc_scratch("tmp_indices_p", VLEN)
        tmp_values_p = self.alloc_scratch("tmp_values_p", VLEN)

        for round in range(rounds):
            for i in range(0, batch_size, VLEN):
                i_const = self.scratch_const_vec(i)

                # inp_indices_i = inp_indices_addr_consts[i]
                # inp_values_i = inp_values_addr_consts[i]

                body.append(
                    (
                        "valu",
                        ("+", tmp_indices_p, inp_indices_p_v, i_const),
                    )
                )
                body.append(("valu", ("+", tmp_values_p, inp_values_p_v, i_const)))

                # idx = mem[inp_indices_p + i]
                body.append(("load", ("vload", tmp_idx, tmp_indices_p)))
                body.append(("debug", ("compare", tmp_idx, (round, i, "idx"))))

                # val = mem[inp_values_p + i]
                body.append(("load", ("vload", tmp_val, tmp_values_p)))
                body.append(("debug", ("compare", tmp_val, (round, i, "val"))))

                # node_val = mem[forest_values_p + idx]
                body.append(("valu", ("+", tmp_addr, forest_values_p_v, tmp_idx)))
                for lane in range(VLEN):
                    body.append(("load", ("load_offset", tmp_node_val, tmp_addr, lane)))
                body.append(
                    ("debug", ("compare", tmp_node_val, (round, i, "node_val")))
                )

                # val = myhash(val ^ node_val)
                body.append(("valu", ("^", tmp_val, tmp_val, tmp_node_val)))
                body.extend(
                    self.build_hash(tmp_val, tmp1, tmp2, round, i, parallel=True)
                )
                body.append(("debug", ("compare", tmp_val, (round, i, "hashed_val"))))

                # idx = 2*idx + (1 if val % 2 == 0 else 2)
                body.append(("valu", ("%", tmp1, tmp_val, two_const)))
                body.append(("valu", ("==", tmp1, tmp1, zero_const)))
                body.append(("flow", ("vselect", tmp3, tmp1, one_const, two_const)))
                body.append(("valu", ("*", tmp_idx, tmp_idx, two_const)))
                body.append(("valu", ("+", tmp_idx, tmp_idx, tmp3)))
                body.append(("debug", ("compare", tmp_idx, (round, i, "next_idx"))))

                # idx = 0 if idx >= n_nodes else idx
                body.append(("valu", ("<", tmp1, tmp_idx, n_nodes_v)))
                body.append(("flow", ("vselect", tmp_idx, tmp1, tmp_idx, zero_const)))
                body.append(("debug", ("compare", tmp_idx, (round, i, "wrapped_idx"))))

                # mem[inp_indices_p + i] = idx
                body.append(("store", ("vstore", tmp_indices_p, tmp_idx)))

                # mem[inp_values_p + i] = val
                body.append(("store", ("vstore", tmp_values_p, tmp_val)))

        body_instrs = self.build(body)
        self.instrs.extend(body_instrs)
        # Required to match with the yield in reference_kernel2
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
