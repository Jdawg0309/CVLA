"""
Operation metadata shared by the CVLA LaTeX inspector.

Each registered operation provides a canonical LaTeX template, domain/codomain
information, and a mapping that allows the inspector to render both the symbolic
formula and numeric substitution for that operation.
"""

from dataclasses import dataclass, field
from typing import Callable, Dict, Optional, TYPE_CHECKING

import numpy as np

from state.models.operation_record import OperationRecord
from state.models.tensor_model import TensorData

if TYPE_CHECKING:
    from state.app_state import AppState


def _format_scalar(value: Optional[float]) -> str:
    if value is None:
        return r"\text{?}"
    try:
        num = float(value)
    except (TypeError, ValueError):
        return str(value)
    if abs(num) < 1e-6 and num != 0:
        return f"{num:.6g}"
    if abs(num) >= 1e5 or abs(num) < 1e-2:
        return f"{num:.3g}"
    return f"{num:.3f}"


def _format_vector(arr: np.ndarray) -> str:
    flat = arr.flatten()
    entries = flat[:4]
    rows = " \\\\ ".join(_format_scalar(float(x)) for x in entries)
    if flat.size > len(entries):
        rows += " \\\\ \\dots"
    return rf"\begin{{bmatrix}} {rows} \end{{bmatrix}}"


def _format_matrix(arr: np.ndarray) -> str:
    rows = []
    for row in arr[:3]:
        entries = row[:4]
        row_text = " & ".join(_format_scalar(float(x)) for x in entries)
        if row.size > len(entries):
            row_text += " & \\dots"
        rows.append(row_text)
    if arr.shape[0] > 3:
        rows.append("\\dots")
    body = r" \\ ".join(rows)
    return rf"\begin{{bmatrix}} {body} \end{{bmatrix}}"


def format_tensor_value(tensor: Optional[TensorData]) -> str:
    if tensor is None:
        return r"\text{missing}"
    arr = tensor.to_numpy()
    if arr.size == 1:
        return _format_scalar(float(arr.flat[0]))
    if arr.ndim == 1:
        return _format_vector(arr)
    if arr.ndim == 2:
        return _format_matrix(arr)
    return r"\text{tensor}"


def _get_tensor_by_index(state: "AppState", record: Optional[OperationRecord], index: Optional[int]) -> Optional[TensorData]:
    if record is None or index is None:
        return None
    if index < 0 or index >= len(record.target_ids):
        return None
    tensor_id = record.target_ids[index]
    return next((t for t in state.tensors if t.id == tensor_id), None)


@dataclass(frozen=True)
class PlaceholderBinding:
    kind: str  # 'target', 'parameter', 'result', 'literal'
    index: Optional[int] = None
    param_name: Optional[str] = None
    literal: Optional[str] = None

    def format(self, state: "AppState", record: Optional[OperationRecord], result_tensor: Optional[TensorData]) -> str:
        if self.kind == "target":
            tensor = _get_tensor_by_index(state, record, self.index)
            return format_tensor_value(tensor)
        if self.kind == "parameter":
            if record is None or self.param_name is None:
                return r"\text{?}"
            params = dict(record.parameters)
            return _format_scalar(params.get(self.param_name))
        if self.kind == "result":
            return format_tensor_value(result_tensor)
        if self.kind == "literal":
            return self.literal or ""
        return r"\text{?}"


NumericFormatter = Callable[[Optional[OperationRecord], "AppState", Optional[TensorData], "OperationMetadata"], str]


@dataclass(frozen=True)
class OperationMetadata:
    name: str
    domain: str
    codomain: str
    latex_template: str
    placeholder_bindings: Dict[str, PlaceholderBinding] = field(default_factory=dict)
    symbol_map: Dict[str, str] = field(default_factory=dict)
    numeric_formatter: Optional[NumericFormatter] = None

    def render_latex(self) -> str:
        if not self.placeholder_bindings:
            return self.latex_template
        mapping = {
            key: self.symbol_map.get(key, key)
            for key in self.placeholder_bindings
        }
        return self.latex_template.format(**mapping)

    def render_numeric(
        self,
        record: Optional[OperationRecord],
        state: "AppState",
        result_tensor: Optional[TensorData],
    ) -> str:
        if self.numeric_formatter:
            return self.numeric_formatter(record, state, result_tensor, self)
        if not self.placeholder_bindings:
            return self.latex_template
        mapping = {}
        for key, binding in self.placeholder_bindings.items():
            mapping[key] = binding.format(state, record, result_tensor)
        return self.latex_template.format(**mapping)


def _target(idx: int) -> PlaceholderBinding:
    return PlaceholderBinding(kind="target", index=idx)


def _parameter(name: str) -> PlaceholderBinding:
    return PlaceholderBinding(kind="parameter", param_name=name)


def _result_binding() -> PlaceholderBinding:
    return PlaceholderBinding(kind="result")


def _literal(value: str) -> PlaceholderBinding:
    return PlaceholderBinding(kind="literal", literal=value)


def _make_metadata(
    name: str,
    domain: str,
    codomain: str,
    latex_template: str,
    placeholders: Dict[str, PlaceholderBinding],
    symbol_map: Optional[Dict[str, str]] = None,
    numeric_formatter: Optional[NumericFormatter] = None,
) -> OperationMetadata:
    return OperationMetadata(
        name=name,
        domain=domain,
        codomain=codomain,
        latex_template=latex_template,
        placeholder_bindings=placeholders,
        symbol_map=symbol_map or {},
        numeric_formatter=numeric_formatter,
    )


def _default_symbol(ph: str) -> str:
    default_map = {
        "A": r"\mathbf{A}",
        "B": r"\mathbf{B}",
        "P": r"\mathbf{P}",
        "V": r"\mathbf{V}",
        "U": r"\mathbf{U}",
        "L": r"\mathbf{L}",
        "R": r"\mathbf{R}",
        "Z": r"\mathbf{Z}",
        "T": r"\mathbf{T}",
        "Sigma": r"\Sigma",
        "Lambda": r"\Lambda",
        "a": r"\mathbf{a}",
        "b": r"\mathbf{b}",
        "v": r"\mathbf{v}",
        "w": r"\mathbf{w}",
        "x": r"\mathbf{x}",
    }
    return default_map.get(ph, ph)


def _with_default_symbols(placeholders: Dict[str, PlaceholderBinding]) -> Dict[str, str]:
    return {key: _default_symbol(key) for key in placeholders}


OPERATION_METADATA: Dict[str, OperationMetadata] = {
    "add": _make_metadata(
        name="Add",
        domain="Tensor × Tensor",
        codomain="Tensor",
        latex_template="{A} + {B}",
        placeholders={"A": _target(0), "B": _target(1)},
        symbol_map={"A": r"\mathbf{A}", "B": r"\mathbf{B}"},
    ),
    "subtract": _make_metadata(
        name="Subtract",
        domain="Tensor × Tensor",
        codomain="Tensor",
        latex_template="{A} - {B}",
        placeholders={"A": _target(0), "B": _target(1)},
        symbol_map={"A": r"\mathbf{A}", "B": r"\mathbf{B}"},
    ),
    "scale": _make_metadata(
        name="Scale",
        domain="Tensor × ℝ",
        codomain="Tensor",
        latex_template="{k} {A}",
        placeholders={"k": _parameter("factor"), "A": _target(0)},
        symbol_map={"A": r"\mathbf{A}", "k": "k"},
    ),
    "negate": _make_metadata(
        name="Negate",
        domain="Tensor",
        codomain="Tensor",
        latex_template="-{A}",
        placeholders={"A": _target(0)},
        symbol_map={"A": r"\mathbf{A}"},
    ),
    "hadamard": _make_metadata(
        name="Hadamard",
        domain="Tensor × Tensor",
        codomain="Tensor",
        latex_template="{A} \\odot {B}",
        placeholders={"A": _target(0), "B": _target(1)},
        symbol_map={"A": r"\mathbf{A}", "B": r"\mathbf{B}"},
    ),
    "matrix_multiply": _make_metadata(
        name="Matrix Multiply",
        domain="Matrix × Matrix",
        codomain="Matrix",
        latex_template="{A} \\cdot {B}",
        placeholders={"A": _target(0), "B": _target(1)},
        symbol_map={"A": r"\mathbf{A}", "B": r"\mathbf{B}"},
    ),
    "outer_product": _make_metadata(
        name="Outer Product",
        domain="Vector × Vector",
        codomain="Matrix",
        latex_template="{a} \\otimes {b}",
        placeholders={"a": _target(0), "b": _target(1)},
        symbol_map={"a": r"\mathbf{a}", "b": r"\mathbf{b}"},
    ),
    "kronecker": _make_metadata(
        name="Kronecker Product",
        domain="Matrix × Matrix",
        codomain="Matrix",
        latex_template="{A} \\otimes {B}",
        placeholders={"A": _target(0), "B": _target(1)},
        symbol_map={"A": r"\mathbf{A}", "B": r"\mathbf{B}"},
    ),
    "determinant": _make_metadata(
        name="Determinant",
        domain="Matrix",
        codomain="Scalar",
        latex_template="\\det\\left({A}\\right)",
        placeholders={"A": _target(0)},
        symbol_map={"A": r"\mathbf{A}"},
    ),
    "trace": _make_metadata(
        name="Trace",
        domain="Matrix",
        codomain="Scalar",
        latex_template="\\operatorname{tr}\\left({A}\\right)",
        placeholders={"A": _target(0)},
        symbol_map={"A": r"\mathbf{A}"},
    ),
    "rank": _make_metadata(
        name="Rank",
        domain="Matrix",
        codomain="Scalar",
        latex_template="\\operatorname{rank}\\left({A}\\right)",
        placeholders={"A": _target(0)},
        symbol_map={"A": r"\mathbf{A}"},
    ),
    "condition_number": _make_metadata(
        name="Condition Number",
        domain="Matrix",
        codomain="Scalar",
        latex_template="\\kappa\\left({A}\\right)",
        placeholders={"A": _target(0)},
        symbol_map={"A": r"\mathbf{A}"},
    ),
    "nullity": _make_metadata(
        name="Nullity",
        domain="Matrix",
        codomain="Scalar",
        latex_template="\\operatorname{null}\\left({A}\\right)",
        placeholders={"A": _target(0)},
        symbol_map={"A": r"\mathbf{A}"},
    ),
    "transpose": _make_metadata(
        name="Transpose",
        domain="Matrix",
        codomain="Matrix",
        latex_template="{A}^T",
        placeholders={"A": _target(0)},
        symbol_map={"A": r"\mathbf{A}"},
    ),
    "inverse": _make_metadata(
        name="Inverse",
        domain="Matrix",
        codomain="Matrix",
        latex_template="{A}^{-1}",
        placeholders={"A": _target(0)},
        symbol_map={"A": r"\mathbf{A}"},
    ),
    "pseudoinverse": _make_metadata(
        name="Pseudoinverse",
        domain="Matrix",
        codomain="Matrix",
        latex_template="{A}^+",
        placeholders={"A": _target(0)},
        symbol_map={"A": r"\mathbf{A}"},
    ),
    "adjoint": _make_metadata(
        name="Adjoint",
        domain="Matrix",
        codomain="Matrix",
        latex_template="{A}^*",
        placeholders={"A": _target(0)},
        symbol_map={"A": r"\mathbf{A}"},
    ),
    "cofactor": _make_metadata(
        name="Cofactor Matrix",
        domain="Matrix",
        codomain="Matrix",
        latex_template="\\operatorname{cof}\\left({A}\\right)",
        placeholders={"A": _target(0)},
        symbol_map={"A": r"\mathbf{A}"},
    ),
    "adjugate": _make_metadata(
        name="Adjugate Matrix",
        domain="Matrix",
        codomain="Matrix",
        latex_template="\\operatorname{adj}\\left({A}\\right)",
        placeholders={"A": _target(0)},
        symbol_map={"A": r"\mathbf{A}"},
    ),
    "eigen": _make_metadata(
        name="Eigendecomposition",
        domain="Matrix",
        codomain="Tuple",
        latex_template="{A} = V \\Lambda V^{-1}",
        placeholders={"A": _target(0)},
        symbol_map={"A": r"\mathbf{A}"},
    ),
    "svd": _make_metadata(
        name="SVD",
        domain="Matrix",
        codomain="Tuple",
        latex_template="{A} = U \\Sigma V^T",
        placeholders={"A": _target(0)},
        symbol_map={"A": r"\mathbf{A}"},
    ),
    "qr": _make_metadata(
        name="QR Decomposition",
        domain="Matrix",
        codomain="Tuple",
        latex_template="{A} = Q R",
        placeholders={"A": _target(0)},
        symbol_map={"A": r"\mathbf{A}"},
    ),
    "lu": _make_metadata(
        name="LU Decomposition",
        domain="Matrix",
        codomain="Tuple",
        latex_template="{A} = L U",
        placeholders={"A": _target(0)},
        symbol_map={"A": r"\mathbf{A}", "L": r"\mathbf{L}", "U": r"\mathbf{U}"},
    ),
    "cholesky": _make_metadata(
        name="Cholesky Decomposition",
        domain="Matrix",
        codomain="Tuple",
        latex_template="{A} = L L^T",
        placeholders={"A": _target(0)},
        symbol_map={"A": r"\mathbf{A}", "L": r"\mathbf{L}"},
    ),
    "schur": _make_metadata(
        name="Schur Decomposition",
        domain="Matrix",
        codomain="Tuple",
        latex_template="{A} = Z T Z^*",
        placeholders={"A": _target(0)},
        symbol_map={"A": r"\mathbf{A}", "Z": r"\mathbf{Z}", "T": r"\mathbf{T}"},
    ),
    "eigenvalues": _make_metadata(
        name="Eigenvalues",
        domain="Matrix",
        codomain="Tuple",
        latex_template="\\Lambda\\left({A}\\right)",
        placeholders={"A": _target(0)},
        symbol_map={"A": r"\mathbf{A}", "Lambda": r"\Lambda"},
    ),
    "eigenvectors": _make_metadata(
        name="Eigenvectors",
        domain="Matrix",
        codomain="Matrix",
        latex_template="{A} v = \\lambda v",
        placeholders={"A": _target(0)},
        symbol_map={"A": r"\mathbf{A}", "v": r"\mathbf{v}", "lambda": r"\lambda"},
    ),
    "spectral_radius": _make_metadata(
        name="Spectral Radius",
        domain="Matrix",
        codomain="Scalar",
        latex_template="\\rho\\left({A}\\right)",
        placeholders={"A": _target(0)},
        symbol_map={"A": r"\mathbf{A}"},
    ),
    "power_iteration": _make_metadata(
        name="Power Iteration",
        domain="Matrix",
        codomain="Vector",
        latex_template="{A}^n {v}",
        placeholders={"A": _target(0), "v": _target(0)},
        symbol_map={"A": r"\mathbf{A}", "v": r"\mathbf{v}"},
    ),
    "change_basis": _make_metadata(
        name="Change of Basis",
        domain="Matrix × Matrix",
        codomain="Matrix",
        latex_template="{P}^{-1} {A} {P}",
        placeholders={"P": _target(1), "A": _target(0)},
        symbol_map={"P": r"\mathbf{P}", "A": r"\mathbf{A}"},
    ),
    "similarity_transform": _make_metadata(
        name="Similarity Transform",
        domain="Matrix × Matrix",
        codomain="Matrix",
        latex_template="{P}^{-1} {A} {P}",
        placeholders={"P": _target(1), "A": _target(0)},
        symbol_map={"P": r"\mathbf{P}", "A": r"\mathbf{A}"},
    ),
    "orthogonalize": _make_metadata(
        name="Orthogonalize",
        domain="Matrix",
        codomain="Matrix",
        latex_template="\\mathrm{{orth}}\\left({A}\\right)",
        placeholders={"A": _target(0)},
        symbol_map={"A": r"\mathbf{A}"},
    ),
    "project_subspace": _make_metadata(
        name="Project Subspace",
        domain="Matrix × Matrix",
        codomain="Matrix",
        latex_template="\\mathrm{{proj}}_{ {B} }\\left({A}\\right)",
        placeholders={"A": _target(0), "B": _target(1)},
        symbol_map={"A": r"\mathbf{A}", "B": r"\mathbf{B}"},
    ),
    "frobenius_norm": _make_metadata(
        name="Frobenius Norm",
        domain="Matrix",
        codomain="Scalar",
        latex_template="\\left\\|{A}\\right\\|_F",
        placeholders={"A": _target(0)},
        symbol_map={"A": r"\mathbf{A}"},
    ),
    "l1_norm": _make_metadata(
        name="L1 Norm",
        domain="Matrix",
        codomain="Scalar",
        latex_template="\\left\\|{A}\\right\\|_1",
        placeholders={"A": _target(0)},
        symbol_map={"A": r"\mathbf{A}"},
    ),
    "l2_norm": _make_metadata(
        name="L2 Norm",
        domain="Matrix",
        codomain="Scalar",
        latex_template="\\left\\|{A}\\right\\|_2",
        placeholders={"A": _target(0)},
        symbol_map={"A": r"\mathbf{A}"},
    ),
    "inf_norm": _make_metadata(
        name="Infinity Norm",
        domain="Matrix",
        codomain="Scalar",
        latex_template="\\left\\|{A}\\right\\|_\\infty",
        placeholders={"A": _target(0)},
        symbol_map={"A": r"\mathbf{A}"},
    ),
    "nuclear_norm": _make_metadata(
        name="Nuclear Norm",
        domain="Matrix",
        codomain="Scalar",
        latex_template="\\left\\|{A}\\right\\|_*",
        placeholders={"A": _target(0)},
        symbol_map={"A": r"\mathbf{A}"},
    ),
    "normalize": _make_metadata(
        name="Normalize",
        domain="Vector",
        codomain="Vector",
        latex_template="\\hat{{v}} = \\frac{{{v}}}{{\\|{v}\\|}}",
        placeholders={"v": _target(0)},
        symbol_map={"v": r"\mathbf{v}"},
    ),
    "dot": _make_metadata(
        name="Dot Product",
        domain="Vector × Vector",
        codomain="Scalar",
        latex_template="{a} \\cdot {b}",
        placeholders={"a": _target(0), "b": _target(1)},
        symbol_map={"a": r"\mathbf{a}", "b": r"\mathbf{b}"},
    ),
    "cross": _make_metadata(
        name="Cross Product",
        domain="Vector × Vector",
        codomain="Vector",
        latex_template="{a} \\times {b}",
        placeholders={"a": _target(0), "b": _target(1)},
        symbol_map={"a": r"\mathbf{a}", "b": r"\mathbf{b}"},
    ),
    "project": _make_metadata(
        name="Project",
        domain="Vector × Vector",
        codomain="Vector",
        latex_template="\\mathrm{{proj}}_{ {b} }({a})",
        placeholders={"a": _target(0), "b": _target(1)},
        symbol_map={"a": r"\mathbf{a}", "b": r"\mathbf{b}"},
    ),
    "reject": _make_metadata(
        name="Reject",
        domain="Vector × Vector",
        codomain="Vector",
        latex_template="\\mathrm{{rej}}_{ {b} }({a})",
        placeholders={"a": _target(0), "b": _target(1)},
        symbol_map={"a": r"\mathbf{a}", "b": r"\mathbf{b}"},
    ),
    "angle_between": _make_metadata(
        name="Angle Between",
        domain="Vector × Vector",
        codomain="Scalar",
        latex_template="\\theta = \\cos^{-1}\\left(\\frac{{a} \\cdot {b}}{\\|{a}\\| \\|{b}\\|}\\right)",
        placeholders={"a": _target(0), "b": _target(1)},
        symbol_map={"a": r"\mathbf{a}", "b": r"\mathbf{b}"},
    ),
    "gaussian_elimination": _make_metadata(
        name="Gaussian Elimination",
        domain="Matrix",
        codomain="Matrix",
        latex_template="\\mathrm{{ref}}({A})",
        placeholders={"A": _target(0)},
        symbol_map={"A": r"\mathbf{A}"},
    ),
    "rref": _make_metadata(
        name="RREF",
        domain="Matrix",
        codomain="Matrix",
        latex_template="\\mathrm{{rref}}({A})",
        placeholders={"A": _target(0)},
        symbol_map={"A": r"\mathbf{A}"},
    ),
    "back_substitution": _make_metadata(
        name="Back Substitution",
        domain="Matrix",
        codomain="Vector",
        latex_template="\\text{BackSub}({A})",
        placeholders={"A": _target(0)},
        symbol_map={"A": r"\mathbf{A}"},
    ),
    "solve_linear": _make_metadata(
        name="Solve Ax=b",
        domain="Matrix × Vector",
        codomain="Vector",
        latex_template="{A}^{-1} {b}",
        placeholders={"A": _target(0), "b": _target(1)},
        symbol_map={"A": r"\mathbf{A}", "b": r"\mathbf{b}"},
    ),
    "least_squares": _make_metadata(
        name="Least Squares",
        domain="Matrix × Vector",
        codomain="Vector",
        latex_template="(A^T A)^{-1} A^T {b}",
        placeholders={"b": _target(1)},
        symbol_map={"b": r"\mathbf{b}"},
    ),
    "symmetrize": _make_metadata(
        name="Symmetrize",
        domain="Matrix",
        codomain="Matrix",
        latex_template="\\frac{1}{2}({A} + {A}^T)",
        placeholders={"A": _target(0)},
        symbol_map={"A": r"\mathbf{A}"},
    ),
    "skew_symmetrize": _make_metadata(
        name="Skew-Symmetrize",
        domain="Matrix",
        codomain="Matrix",
        latex_template="\\frac{1}{2}({A} - {A}^T)",
        placeholders={"A": _target(0)},
        symbol_map={"A": r"\mathbf{A}"},
    ),
    "diagonalize": _make_metadata(
        name="Diagonalize",
        domain="Matrix",
        codomain="Matrix",
        latex_template="\\mathrm{{diag}}({A})",
        placeholders={"A": _target(0)},
        symbol_map={"A": r"\mathbf{A}"},
    ),
    "triangular_upper": _make_metadata(
        name="Upper Triangular",
        domain="Matrix",
        codomain="Matrix",
        latex_template="\\mathrm{{triu}}({A})",
        placeholders={"A": _target(0)},
        symbol_map={"A": r"\mathbf{A}"},
    ),
    "triangular_lower": _make_metadata(
        name="Lower Triangular",
        domain="Matrix",
        codomain="Matrix",
        latex_template="\\mathrm{{tril}}({A})",
        placeholders={"A": _target(0)},
        symbol_map={"A": r"\mathbf{A}"},
    ),
    "matrix_vector_multiply": _make_metadata(
        name="Matrix-Vector Multiply",
        domain="Matrix × Vector",
        codomain="Vector",
        latex_template="{A} {b}",
        placeholders={"A": _target(0), "b": _target(1)},
        symbol_map={"A": r"\mathbf{A}", "b": r"\mathbf{b}"},
    ),
}


def get_operation_metadata(operation_id: str) -> OperationMetadata:
    """Return metadata for the given operation, with a conservative fallback."""
    meta = OPERATION_METADATA.get(operation_id)
    if meta:
        if not meta.symbol_map:
            meta = OperationMetadata(
                name=meta.name,
                domain=meta.domain,
                codomain=meta.codomain,
                latex_template=meta.latex_template,
                placeholder_bindings=meta.placeholder_bindings,
                symbol_map=_with_default_symbols(meta.placeholder_bindings),
                numeric_formatter=meta.numeric_formatter,
            )
        return meta
    return OperationMetadata(
        name=operation_id,
        domain="Tensor",
        codomain="Tensor",
        latex_template=operation_id,
        placeholder_bindings={},
    )
