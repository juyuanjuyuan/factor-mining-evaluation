"""Expression-tree primitives for paper-style genetic programming."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Iterable, Mapping

import numpy as np


DEFAULT_TERMINALS = ("c", "o", "h", "l", "vol", "vwap", "pct(c, 1)")
DEFAULT_WINDOWS = (2, 3, 5, 10, 20, 40, 60)
DEFAULT_EXPONENTS = (0.5, 1.5, 2.0)


@dataclass(frozen=True)
class FunctionSpec:
    name: str
    arity: int
    parameter_kind: str | None = None


FUNCTION_SPECS: tuple[FunctionSpec, ...] = (
    FunctionSpec("add", 2),
    FunctionSpec("sub", 2),
    FunctionSpec("mul", 2),
    FunctionSpec("div", 2),
    FunctionSpec("abs", 1),
    FunctionSpec("sqrt", 1),
    FunctionSpec("log", 1),
    FunctionSpec("inv", 1),
    FunctionSpec("neg", 1),
    FunctionSpec("rank_cs", 1),
    FunctionSpec("delay", 1, "window"),
    FunctionSpec("delta", 1, "window"),
    FunctionSpec("ts_corr", 2, "window"),
    FunctionSpec("ts_cov", 2, "window"),
    FunctionSpec("scale_cs", 1),
    FunctionSpec("signed_power", 1, "exponent"),
    FunctionSpec("decay_linear", 1, "window"),
    FunctionSpec("ts_min", 1, "window"),
    FunctionSpec("ts_max", 1, "window"),
    FunctionSpec("ts_argmin", 1, "window"),
    FunctionSpec("ts_argmax", 1, "window"),
    FunctionSpec("ts_rank", 1, "window"),
    FunctionSpec("ts_sum", 1, "window"),
    FunctionSpec("ts_product", 1, "window"),
    FunctionSpec("ts_std", 1, "window"),
)

FUNCTION_BY_NAME = {spec.name: spec for spec in FUNCTION_SPECS}


@dataclass(frozen=True)
class ExpressionTree:
    """Immutable GP node serializable without relying on Python pickle."""

    kind: str
    value: str | float
    children: tuple["ExpressionTree", ...] = ()
    parameter: float | int | None = None

    def __post_init__(self) -> None:
        if self.kind not in {"function", "terminal", "constant"}:
            raise ValueError(f"Unknown expression-tree node kind: {self.kind}")
        if self.kind == "function":
            spec = FUNCTION_BY_NAME.get(str(self.value))
            if spec is None:
                raise ValueError(f"Unknown GP function: {self.value}")
            if len(self.children) != spec.arity:
                raise ValueError(
                    f"Function {self.value} requires {spec.arity} children"
                )
            if spec.parameter_kind is None and self.parameter is not None:
                raise ValueError(f"Function {self.value} does not accept a parameter")
            if spec.parameter_kind is not None and self.parameter is None:
                raise ValueError(f"Function {self.value} requires a parameter")
        elif self.children or self.parameter is not None:
            raise ValueError("Terminal and constant nodes cannot have children/parameters")

    @property
    def node_count(self) -> int:
        return 1 + sum(child.node_count for child in self.children)

    @property
    def depth(self) -> int:
        return 0 if not self.children else 1 + max(child.depth for child in self.children)

    @property
    def has_data_terminal(self) -> bool:
        if self.kind == "terminal":
            return True
        return any(child.has_data_terminal for child in self.children)

    def paths(self) -> tuple[tuple[int, ...], ...]:
        result: list[tuple[int, ...]] = [()]
        for index, child in enumerate(self.children):
            result.extend((index, *path) for path in child.paths())
        return tuple(result)

    def subtree(self, path: tuple[int, ...]) -> "ExpressionTree":
        node = self
        for index in path:
            node = node.children[index]
        return node

    def replace(
        self,
        path: tuple[int, ...],
        replacement: "ExpressionTree",
    ) -> "ExpressionTree":
        if not path:
            return replacement
        index = path[0]
        children = list(self.children)
        children[index] = children[index].replace(path[1:], replacement)
        return ExpressionTree(
            kind=self.kind,
            value=self.value,
            children=tuple(children),
            parameter=self.parameter,
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "kind": self.kind,
            "value": self.value,
            "parameter": self.parameter,
            "children": [child.to_dict() for child in self.children],
        }

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> "ExpressionTree":
        return cls(
            kind=str(payload["kind"]),
            value=payload["value"],
            parameter=payload.get("parameter"),
            children=tuple(cls.from_dict(child) for child in payload.get("children", [])),
        )

    def to_expression(self) -> str:
        if self.kind == "terminal":
            return str(self.value)
        if self.kind == "constant":
            value = float(self.value)
            if np.isclose(value, 0.0):
                value = 0.0
            return f"{value:.12g}"

        name = str(self.value)
        args = [child.to_expression() for child in self.children]
        left = f"({args[0]})"
        if name == "add":
            return f"({left} + ({args[1]}))"
        if name == "sub":
            return f"({left} - ({args[1]}))"
        if name == "mul":
            return f"({left} * ({args[1]}))"
        if name == "div":
            right = f"({args[1]})"
            denominator = f"where(abs({right}) > 1e-6, {right}, 1.0)"
            return f"({left} / {denominator})"
        if name == "abs":
            return f"abs({left})"
        if name == "sqrt":
            return f"sqrt(abs({left}))"
        if name == "log":
            return f"log(abs({left}) + 1e-6)"
        if name == "inv":
            denominator = f"where(abs({left}) > 1e-6, {left}, 1.0)"
            return f"(1.0 / {denominator})"
        if name == "neg":
            return f"(-{left})"
        if name == "rank_cs":
            return f"rank_cs({left})"
        if name == "scale_cs":
            return f"scale_cs({left}, 1.0)"
        if name == "signed_power":
            return f"signed_power({left}, {float(self.parameter):.12g})"
        if name in {"delay", "delta", "decay_linear", "ts_min", "ts_max", "ts_argmin", "ts_argmax", "ts_rank", "ts_sum", "ts_product", "ts_std"}:
            return f"{name}({left}, {int(self.parameter)})"
        if name in {"ts_corr", "ts_cov"}:
            return f"{name}({left}, ({args[1]}), {int(self.parameter)})"
        raise ValueError(f"No expression renderer for GP function: {name}")


def _random_parameter(
    rng: np.random.Generator,
    kind: str | None,
    *,
    windows: tuple[int, ...],
    exponents: tuple[float, ...],
) -> int | float | None:
    if kind == "window":
        return int(rng.choice(windows))
    if kind == "exponent":
        return float(rng.choice(exponents))
    return None


def random_leaf(
    rng: np.random.Generator,
    *,
    terminals: tuple[str, ...] = DEFAULT_TERMINALS,
    constant_range: tuple[float, float] | None = (-1.0, 1.0),
) -> ExpressionTree:
    use_constant = constant_range is not None and bool(rng.random() < 0.2)
    if use_constant:
        lower, upper = constant_range
        return ExpressionTree("constant", float(rng.uniform(lower, upper)))
    return ExpressionTree("terminal", str(rng.choice(terminals)))


def random_tree(
    rng: np.random.Generator,
    *,
    target_depth: int,
    grow: bool,
    terminals: tuple[str, ...] = DEFAULT_TERMINALS,
    functions: tuple[FunctionSpec, ...] = FUNCTION_SPECS,
    windows: tuple[int, ...] = DEFAULT_WINDOWS,
    exponents: tuple[float, ...] = DEFAULT_EXPONENTS,
    constant_range: tuple[float, float] | None = (-1.0, 1.0),
    current_depth: int = 0,
) -> ExpressionTree:
    if target_depth < 0:
        raise ValueError("target_depth cannot be negative")
    if current_depth >= target_depth or (
        grow and current_depth > 0 and rng.random() < 0.35
    ):
        return random_leaf(
            rng,
            terminals=terminals,
            constant_range=constant_range,
        )
    spec = functions[int(rng.integers(0, len(functions)))]
    children = tuple(
        random_tree(
            rng,
            target_depth=target_depth,
            grow=grow,
            terminals=terminals,
            functions=functions,
            windows=windows,
            exponents=exponents,
            constant_range=constant_range,
            current_depth=current_depth + 1,
        )
        for _ in range(spec.arity)
    )
    return ExpressionTree(
        "function",
        spec.name,
        children=children,
        parameter=_random_parameter(
            rng,
            spec.parameter_kind,
            windows=windows,
            exponents=exponents,
        ),
    )


def valid_tree(tree: ExpressionTree, *, max_depth: int, max_nodes: int) -> bool:
    return (
        tree.has_data_terminal
        and tree.depth <= max_depth
        and tree.node_count <= max_nodes
    )


def _is_constant(tree: ExpressionTree, value: float) -> bool:
    return tree.kind == "constant" and float(tree.value) == value


def _fold_constant_function(
    name: str,
    children: tuple[ExpressionTree, ...],
) -> ExpressionTree | None:
    """Fold only scalar operations whose protected semantics match the engine."""

    if not children or any(child.kind != "constant" for child in children):
        return None
    values = tuple(float(child.value) for child in children)
    try:
        if name == "add":
            value = values[0] + values[1]
        elif name == "sub":
            value = values[0] - values[1]
        elif name == "mul":
            value = values[0] * values[1]
        elif name == "div":
            denominator = values[1] if abs(values[1]) > 1e-6 else 1.0
            value = values[0] / denominator
        elif name == "abs":
            value = abs(values[0])
        elif name == "sqrt":
            value = float(np.sqrt(abs(values[0])))
        elif name == "log":
            value = float(np.log(abs(values[0]) + 1e-6))
        elif name == "inv":
            value = 1.0 / values[0] if abs(values[0]) > 1e-6 else 1.0
        elif name == "neg":
            value = -values[0]
        else:
            return None
    except (ArithmeticError, FloatingPointError, ValueError):
        return None
    return ExpressionTree("constant", value) if np.isfinite(value) else None


def canonicalize_tree(tree: ExpressionTree) -> ExpressionTree:
    """Apply conservative rewrites that preserve the engine's NaN semantics."""

    if tree.kind == "constant":
        value = float(tree.value)
        return ExpressionTree("constant", 0.0 if value == 0.0 else value)
    if tree.kind == "terminal":
        return tree

    name = str(tree.value)
    children = tuple(canonicalize_tree(child) for child in tree.children)
    folded = _fold_constant_function(name, children)
    if folded is not None:
        return folded

    if name == "neg" and children[0].kind == "function" and children[0].value == "neg":
        return children[0].children[0]
    if name == "add":
        if _is_constant(children[0], 0.0):
            return children[1]
        if _is_constant(children[1], 0.0):
            return children[0]
    if name == "sub" and _is_constant(children[1], 0.0):
        return children[0]
    if name == "mul":
        if _is_constant(children[0], 1.0):
            return children[1]
        if _is_constant(children[1], 1.0):
            return children[0]
    if name == "div" and _is_constant(children[1], 1.0):
        return children[0]
    if name == "rank_cs" and children[0].kind == "function" and children[0].value == "rank_cs":
        return children[0]

    if name in {"add", "mul"}:
        children = tuple(sorted(children, key=lambda child: child.to_expression()))
    return ExpressionTree("function", name, children, tree.parameter)


def _paths_matching(
    tree: ExpressionTree,
    predicate,
) -> tuple[tuple[int, ...], ...]:
    return tuple(path for path in tree.paths() if predicate(tree.subtree(path)))


def mutate_window(
    tree: ExpressionTree,
    rng: np.random.Generator,
    *,
    windows: tuple[int, ...] = DEFAULT_WINDOWS,
) -> ExpressionTree:
    """Change exactly one time-series window parameter."""

    paths = _paths_matching(
        tree,
        lambda node: (
            node.kind == "function"
            and FUNCTION_BY_NAME[str(node.value)].parameter_kind == "window"
        ),
    )
    if not paths:
        return tree
    path = paths[int(rng.integers(0, len(paths)))]
    node = tree.subtree(path)
    alternatives = tuple(value for value in windows if int(value) != int(node.parameter))
    if not alternatives:
        return tree
    parameter = int(alternatives[int(rng.integers(0, len(alternatives)))])
    replacement = ExpressionTree(
        "function",
        node.value,
        node.children,
        parameter,
    )
    return tree.replace(path, replacement)


def mutate_constant(
    tree: ExpressionTree,
    rng: np.random.Generator,
    *,
    constant_range: tuple[float, float] | None = (-1.0, 1.0),
    perturbation_std: float = 0.1,
) -> ExpressionTree:
    """Perturb exactly one existing scalar constant."""

    if constant_range is None:
        return tree
    paths = _paths_matching(tree, lambda node: node.kind == "constant")
    if not paths:
        return tree
    path = paths[int(rng.integers(0, len(paths)))]
    node = tree.subtree(path)
    lower, upper = constant_range
    value = float(
        np.clip(float(node.value) + rng.normal(0.0, perturbation_std), lower, upper)
    )
    return tree.replace(path, ExpressionTree("constant", value))


def delete_node(
    tree: ExpressionTree,
    rng: np.random.Generator,
) -> ExpressionTree:
    """Delete one function node by promoting one of its children."""

    paths = _paths_matching(tree, lambda node: node.kind == "function")
    if not paths:
        return tree
    path = paths[int(rng.integers(0, len(paths)))]
    node = tree.subtree(path)
    child = node.children[int(rng.integers(0, len(node.children)))]
    return tree.replace(path, child)


def insert_node(
    tree: ExpressionTree,
    rng: np.random.Generator,
    *,
    functions: tuple[FunctionSpec, ...] = FUNCTION_SPECS,
    windows: tuple[int, ...] = DEFAULT_WINDOWS,
    exponents: tuple[float, ...] = DEFAULT_EXPONENTS,
) -> ExpressionTree:
    """Wrap one subtree in a randomly selected unary function."""

    unary = tuple(spec for spec in functions if spec.arity == 1)
    if not unary:
        return tree
    path = tree.paths()[int(rng.integers(0, len(tree.paths())))]
    subtree = tree.subtree(path)
    spec = unary[int(rng.integers(0, len(unary)))]
    replacement = ExpressionTree(
        "function",
        spec.name,
        (subtree,),
        _random_parameter(
            rng,
            spec.parameter_kind,
            windows=windows,
            exponents=exponents,
        ),
    )
    return tree.replace(path, replacement)


def point_mutation(
    tree: ExpressionTree,
    rng: np.random.Generator,
    *,
    replacement_probability: float,
    terminals: tuple[str, ...] = DEFAULT_TERMINALS,
    functions: tuple[FunctionSpec, ...] = FUNCTION_SPECS,
    windows: tuple[int, ...] = DEFAULT_WINDOWS,
    exponents: tuple[float, ...] = DEFAULT_EXPONENTS,
    constant_range: tuple[float, float] | None = (-1.0, 1.0),
) -> ExpressionTree:
    """Replace exactly one operator or leaf while preserving tree arity."""

    if not 0 <= replacement_probability <= 1:
        raise ValueError("replacement_probability must be between zero and one")
    if rng.random() < replacement_probability:
        path = tree.paths()[int(rng.integers(0, len(tree.paths())))]
        node = tree.subtree(path)
        if node.kind == "function":
            current = FUNCTION_BY_NAME[str(node.value)]
            alternatives = tuple(
                spec
                for spec in functions
                if spec.arity == current.arity and spec.name != current.name
            )
            if not alternatives:
                return tree
            replacement_spec = alternatives[
                int(rng.integers(0, len(alternatives)))
            ]
            replacement = ExpressionTree(
                "function",
                replacement_spec.name,
                node.children,
                _random_parameter(
                    rng,
                    replacement_spec.parameter_kind,
                    windows=windows,
                    exponents=exponents,
                ),
            )
        else:
            replacement = random_leaf(
                rng,
                terminals=terminals,
                constant_range=constant_range,
            )
        return tree.replace(path, replacement)
    return tree


def unique_expressions(trees: Iterable[ExpressionTree]) -> tuple[ExpressionTree, ...]:
    result: list[ExpressionTree] = []
    seen: set[str] = set()
    for raw_tree in trees:
        tree = canonicalize_tree(raw_tree)
        expression = tree.to_expression()
        if expression not in seen:
            seen.add(expression)
            result.append(tree)
    return tuple(result)
