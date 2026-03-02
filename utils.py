from typing import Any, List, Tuple

from problem import SLOT_LIMITS, VLEN, Engine, Instruction


class InstructionRecord:
    instruction: Instruction
    srcs: set[int]
    dsts: set[int]
    full: bool

    def __init__(self):
        self.instruction = {"alu": [], "flow": [], "load": [], "store": []}
        self.srcs = set()
        self.dsts = set()
        self.full = False

    def is_engine_full(self, engine: Engine) -> bool:
        if engine not in self.instruction:
            return False

        self.full = self.full or len(self.instruction[engine]) >= SLOT_LIMITS[engine]

        return self.full

    def has_any_src(self, srcs: list[int]) -> bool:
        return bool(self.srcs.intersection(srcs))

    def has_any_dst(self, dsts: list[int]) -> bool:
        return bool(self.dsts.intersection(dsts))

    def add_src(self, src: int):
        self.srcs.add(src)

    def add_dst(self, dst: int):
        self.dsts.add(dst)

    def add_srcs(self, srcs: list[int]):
        self.srcs.update(srcs)

    def add_dsts(self, dsts: list[int]):
        self.dsts.update(dsts)


def _vrange(base: int) -> List[int]:
    return [base + i for i in range(VLEN)]


def add_to_record(
    record: InstructionRecord,
    engine: Engine,
    slot: tuple,
    src: list[int],
    dst: list[int],
):
    if record.has_any_dst(src) or record.is_engine_full(engine):
        return False

    if engine not in record.instruction:
        record.instruction[engine] = []
    record.instruction[engine].append(slot)

    record.add_dsts(dst)
    record.add_srcs(src)

    return True


def get_addresses(
    engine: str,
    params: Tuple[Any, ...],
) -> Tuple[List[int], List[int]]:
    engine = engine.lower()

    try:
        if engine == "alu":
            # alu(op, dest, a1, a2)
            op, dest, a1, a2 = params
            return [a1, a2], [dest]

        elif engine == "valu":
            match params:
                case ("vbroadcast", dest, src):
                    return [src], _vrange(dest)

                case ("multiply_add", dest, a, b, c):
                    return _vrange(a) + _vrange(b) + _vrange(c), _vrange(dest)

                case (op, dest, a1, a2):
                    # generic vector ALU op
                    return _vrange(a1) + _vrange(a2), _vrange(dest)

                case _:
                    raise NotImplementedError(f"Unknown valu op {params}")

        elif engine == "load":
            match params:
                case ("load", dest, addr):
                    # reads scratch[addr], writes scratch[dest]
                    return [addr], [dest]

                case ("load_offset", dest, addr, offset):
                    # reads scratch[addr + offset], writes scratch[dest + offset]
                    return [addr + offset], [dest + offset]

                case ("vload", dest, addr):
                    # reads scratch[addr], writes scratch[dest : dest+vlen]
                    return [addr], _vrange(dest)

                case ("const", dest, val):
                    return [], [dest]

                case _:
                    raise NotImplementedError(f"Unknown load op {params}")

        elif engine == "store":
            match params:
                case ("store", addr, src):
                    # reads scratch[addr] and scratch[src], writes no scratch
                    return [addr, src], []

                case ("vstore", addr, src):
                    # reads scratch[addr] and scratch[src : src+vlen], writes no scratch
                    return [addr] + _vrange(src), []

                case _:
                    raise NotImplementedError(f"Unknown store op {params}")

        elif engine == "flow":
            match params:
                case ("select", dest, cond, a, b):
                    return [cond, a, b], [dest]

                case ("add_imm", dest, a, imm):
                    return [a], [dest]

                case ("vselect", dest, cond, a, b):
                    return _vrange(cond) + _vrange(a) + _vrange(b), _vrange(dest)

                case ("halt",):
                    return [], []

                case ("pause",):
                    return [], []

                case ("trace_write", val):
                    return [val], []

                case ("cond_jump", cond, addr):
                    # addr is a literal PC target
                    return [cond], []

                case ("cond_jump_rel", cond, offset):
                    # offset is immediate
                    return [cond], []

                case ("jump", addr):
                    # addr is a literal PC target
                    return [], []

                case ("jump_indirect", addr):
                    return [addr], []

                case ("coreid", dest):
                    return [], [dest]

                case _:
                    raise NotImplementedError(f"Unknown flow op {params}")

        elif engine == "debug":
            match params:
                case ("compare", loc, _key):
                    # reads one scalar scratch location
                    return [loc], []

                case ("vcompare", loc, _keys):
                    # reads a full vector window at loc
                    return _vrange(loc), []

                case ("comment", _msg):
                    return [], []

                case _:
                    raise NotImplementedError(f"Unknown debug op {params}")

        else:
            raise NotImplementedError(f"Unknown engine {engine}")

    except:
        return [], []
