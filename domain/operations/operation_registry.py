"""
Operation registry for CVLA.

Contains 30+ linear algebra operations organized by category:
- Basic Arithmetic (add, subtract, scale, negate, hadamard)
- Matrix Multiplication (matrix multiply, outer product, kronecker)
- Matrix Properties (determinant, trace, rank, condition number, nullity)
- Transformations (transpose, inverse, pseudoinverse, adjoint, cofactor, adjugate)
- Decompositions (eigen, SVD, QR, LU, Cholesky, Schur)
- Eigenvalues & Vectors (eigenvalues, eigenvectors, spectral radius, power iteration)
- Change of Basis (change basis, similarity, orthogonalize, project subspace)
- Norms & Distances (frobenius, L1, L2, infinity, nuclear)
- Vector Operations (normalize, dot, cross, project, reject, angle)
- Linear Systems (gaussian elimination, RREF, back substitution, solve, least squares)
- Special Matrices (symmetrize, skew-symmetrize, diagonalize, triangular)
"""

from typing import Dict, List, Optional, Any, Sequence, Tuple
import numpy as np

from domain.operations.operation_spec import OperationSpec
from state.models.operation_step import OperationStep, RenderVectorHint


class OperationRegistry:
    """Registry for all operations."""

    def __init__(self):
        self._operations: Dict[str, OperationSpec] = {}

    def register(self, operation: OperationSpec) -> None:
        self._operations[operation.id] = operation

    def get(self, operation_id: str) -> Optional[OperationSpec]:
        return self._operations.get(operation_id)

    def list_all(self) -> List[OperationSpec]:
        return list(self._operations.values())


# ============================================================================
# BASIC ARITHMETIC OPERATIONS
# ============================================================================

class AddOperation(OperationSpec):
    id = "add"
    inputs = ("tensor_a", "tensor_b")
    outputs = ("tensor",)
    assumptions = ("same shape",)

    def compute(self, inputs: Dict[str, Any], params: Dict[str, Any]) -> Any:
        tensors = inputs.get("tensors") or ()
        if len(tensors) < 2:
            raise ValueError("Add requires two tensors.")
        a, b = tensors[0].to_numpy(), tensors[1].to_numpy()
        if a.shape != b.shape:
            raise ValueError("Tensor shapes must match.")
        return a + b

    def steps(self, inputs, params, result):
        return (OperationStep(title="Add", description="Element-wise addition", values=()),)


class SubtractOperation(OperationSpec):
    id = "subtract"
    inputs = ("tensor_a", "tensor_b")
    outputs = ("tensor",)
    assumptions = ("same shape",)

    def compute(self, inputs: Dict[str, Any], params: Dict[str, Any]) -> Any:
        tensors = inputs.get("tensors") or ()
        if len(tensors) < 2:
            raise ValueError("Subtract requires two tensors.")
        a, b = tensors[0].to_numpy(), tensors[1].to_numpy()
        if a.shape != b.shape:
            raise ValueError("Tensor shapes must match.")
        return a - b

    def steps(self, inputs, params, result):
        return (OperationStep(title="Subtract", description="Element-wise subtraction", values=()),)


class ScaleOperation(OperationSpec):
    id = "scale"
    inputs = ("tensor",)
    outputs = ("tensor",)
    assumptions = ()

    def compute(self, inputs: Dict[str, Any], params: Dict[str, Any]) -> Any:
        tensor = inputs.get("tensor") or (inputs.get("tensors") or [None])[0]
        if tensor is None:
            raise ValueError("Scale requires a tensor.")
        factor = float(params.get("factor", 1.0))
        return tensor.to_numpy() * factor

    def steps(self, inputs, params, result):
        factor = params.get("factor", 1.0)
        return (OperationStep(title="Scale", description=f"Multiply by {factor}", values=(("factor", factor),)),)


class NegateOperation(OperationSpec):
    id = "negate"
    inputs = ("tensor",)
    outputs = ("tensor",)
    assumptions = ()

    def compute(self, inputs: Dict[str, Any], params: Dict[str, Any]) -> Any:
        tensor = inputs.get("tensor") or (inputs.get("tensors") or [None])[0]
        if tensor is None:
            raise ValueError("Negate requires a tensor.")
        return -tensor.to_numpy()

    def steps(self, inputs, params, result):
        return (OperationStep(title="Negate", description="Multiply by -1", values=()),)


class HadamardOperation(OperationSpec):
    id = "hadamard"
    inputs = ("tensor_a", "tensor_b")
    outputs = ("tensor",)
    assumptions = ("same shape",)

    def compute(self, inputs: Dict[str, Any], params: Dict[str, Any]) -> Any:
        tensors = inputs.get("tensors") or ()
        if len(tensors) < 2:
            raise ValueError("Hadamard requires two tensors.")
        a, b = tensors[0].to_numpy(), tensors[1].to_numpy()
        if a.shape != b.shape:
            raise ValueError("Tensor shapes must match.")
        return a * b

    def steps(self, inputs, params, result):
        return (OperationStep(title="Hadamard", description="Element-wise multiplication", values=()),)


# ============================================================================
# MATRIX MULTIPLICATION OPERATIONS
# ============================================================================

class MatrixMultiplyOperation(OperationSpec):
    id = "matrix_multiply"
    inputs = ("matrix_a", "matrix_b")
    outputs = ("matrix",)
    assumptions = ("A.cols == B.rows",)

    def compute(self, inputs: Dict[str, Any], params: Dict[str, Any]) -> Any:
        tensors = inputs.get("tensors") or ()
        if len(tensors) < 2:
            raise ValueError("Matrix multiply requires two tensors.")
        a, b = tensors[0].to_numpy(), tensors[1].to_numpy()
        return a @ b

    def steps(self, inputs, params, result):
        return (OperationStep(title="Matrix Multiply", description="A × B", values=()),)


class OuterProductOperation(OperationSpec):
    id = "outer_product"
    inputs = ("vector_a", "vector_b")
    outputs = ("matrix",)
    assumptions = ()

    def compute(self, inputs: Dict[str, Any], params: Dict[str, Any]) -> Any:
        tensors = inputs.get("tensors") or ()
        if len(tensors) < 2:
            raise ValueError("Outer product requires two vectors.")
        a, b = tensors[0].to_numpy(), tensors[1].to_numpy()
        return np.outer(a, b)

    def steps(self, inputs, params, result):
        return (OperationStep(title="Outer Product", description="a ⊗ b", values=()),)


class KroneckerOperation(OperationSpec):
    id = "kronecker"
    inputs = ("matrix_a", "matrix_b")
    outputs = ("matrix",)
    assumptions = ()

    def compute(self, inputs: Dict[str, Any], params: Dict[str, Any]) -> Any:
        tensors = inputs.get("tensors") or ()
        if len(tensors) < 2:
            raise ValueError("Kronecker product requires two tensors.")
        a, b = tensors[0].to_numpy(), tensors[1].to_numpy()
        return np.kron(a, b)

    def steps(self, inputs, params, result):
        return (OperationStep(title="Kronecker Product", description="A ⊗ B", values=()),)


# ============================================================================
# MATRIX PROPERTIES
# ============================================================================

class DeterminantOperation(OperationSpec):
    id = "determinant"
    inputs = ("matrix",)
    outputs = ("scalar",)
    assumptions = ("square matrix",)

    def compute(self, inputs: Dict[str, Any], params: Dict[str, Any]) -> float:
        tensor = inputs.get("tensor") or inputs.get("matrix") or (inputs.get("tensors") or [None])[0]
        if tensor is None:
            raise ValueError("Determinant requires a matrix.")
        mat = tensor.to_numpy()
        return float(np.linalg.det(mat))

    def steps(self, inputs, params, result):
        return (OperationStep(title="Determinant", description=f"det(A) = {result:.6g}", values=(("det", result),)),)


class TraceOperation(OperationSpec):
    id = "trace"
    inputs = ("matrix",)
    outputs = ("scalar",)
    assumptions = ("square matrix",)

    def compute(self, inputs: Dict[str, Any], params: Dict[str, Any]) -> float:
        tensor = inputs.get("tensor") or inputs.get("matrix") or (inputs.get("tensors") or [None])[0]
        if tensor is None:
            raise ValueError("Trace requires a matrix.")
        mat = tensor.to_numpy()
        return float(np.trace(mat))

    def steps(self, inputs, params, result):
        return (OperationStep(title="Trace", description=f"tr(A) = {result:.6g}", values=(("trace", result),)),)


class RankOperation(OperationSpec):
    id = "rank"
    inputs = ("matrix",)
    outputs = ("scalar",)
    assumptions = ()

    def compute(self, inputs: Dict[str, Any], params: Dict[str, Any]) -> int:
        tensor = inputs.get("tensor") or inputs.get("matrix") or (inputs.get("tensors") or [None])[0]
        if tensor is None:
            raise ValueError("Rank requires a matrix.")
        mat = tensor.to_numpy()
        return int(np.linalg.matrix_rank(mat))

    def steps(self, inputs, params, result):
        return (OperationStep(title="Rank", description=f"rank(A) = {result}", values=(("rank", result),)),)


class ConditionNumberOperation(OperationSpec):
    id = "condition_number"
    inputs = ("matrix",)
    outputs = ("scalar",)
    assumptions = ()

    def compute(self, inputs: Dict[str, Any], params: Dict[str, Any]) -> float:
        tensor = inputs.get("tensor") or inputs.get("matrix") or (inputs.get("tensors") or [None])[0]
        if tensor is None:
            raise ValueError("Condition number requires a matrix.")
        mat = tensor.to_numpy()
        return float(np.linalg.cond(mat))

    def steps(self, inputs, params, result):
        return (OperationStep(title="Condition Number", description=f"κ(A) = {result:.6g}", values=(("cond", result),)),)


class NullityOperation(OperationSpec):
    id = "nullity"
    inputs = ("matrix",)
    outputs = ("scalar",)
    assumptions = ()

    def compute(self, inputs: Dict[str, Any], params: Dict[str, Any]) -> int:
        tensor = inputs.get("tensor") or inputs.get("matrix") or (inputs.get("tensors") or [None])[0]
        if tensor is None:
            raise ValueError("Nullity requires a matrix.")
        mat = tensor.to_numpy()
        rank = np.linalg.matrix_rank(mat)
        return mat.shape[1] - rank  # nullity = cols - rank

    def steps(self, inputs, params, result):
        return (OperationStep(title="Nullity", description=f"null(A) = {result}", values=(("nullity", result),)),)


# ============================================================================
# TRANSFORMATIONS
# ============================================================================

class TransposeOperation(OperationSpec):
    id = "transpose"
    inputs = ("matrix",)
    outputs = ("matrix",)
    assumptions = ()

    def compute(self, inputs: Dict[str, Any], params: Dict[str, Any]) -> Any:
        tensor = inputs.get("tensor") or inputs.get("matrix") or (inputs.get("tensors") or [None])[0]
        if tensor is None:
            raise ValueError("Transpose requires a matrix.")
        return tensor.to_numpy().T

    def steps(self, inputs, params, result):
        return (OperationStep(title="Transpose", description="Aᵀ", values=()),)


class InverseOperation(OperationSpec):
    id = "inverse"
    inputs = ("matrix",)
    outputs = ("matrix",)
    assumptions = ("square matrix", "non-singular")

    def compute(self, inputs: Dict[str, Any], params: Dict[str, Any]) -> Any:
        tensor = inputs.get("tensor") or inputs.get("matrix") or (inputs.get("tensors") or [None])[0]
        if tensor is None:
            raise ValueError("Inverse requires a matrix.")
        mat = tensor.to_numpy()
        return np.linalg.inv(mat)

    def steps(self, inputs, params, result):
        return (OperationStep(title="Inverse", description="A⁻¹", values=()),)


class PseudoinverseOperation(OperationSpec):
    id = "pseudoinverse"
    inputs = ("matrix",)
    outputs = ("matrix",)
    assumptions = ()

    def compute(self, inputs: Dict[str, Any], params: Dict[str, Any]) -> Any:
        tensor = inputs.get("tensor") or inputs.get("matrix") or (inputs.get("tensors") or [None])[0]
        if tensor is None:
            raise ValueError("Pseudoinverse requires a matrix.")
        mat = tensor.to_numpy()
        return np.linalg.pinv(mat)

    def steps(self, inputs, params, result):
        return (OperationStep(title="Pseudoinverse", description="A⁺ (Moore-Penrose)", values=()),)


class AdjointOperation(OperationSpec):
    id = "adjoint"
    inputs = ("matrix",)
    outputs = ("matrix",)
    assumptions = ()

    def compute(self, inputs: Dict[str, Any], params: Dict[str, Any]) -> Any:
        tensor = inputs.get("tensor") or inputs.get("matrix") or (inputs.get("tensors") or [None])[0]
        if tensor is None:
            raise ValueError("Adjoint requires a matrix.")
        mat = tensor.to_numpy()
        return np.conj(mat.T)

    def steps(self, inputs, params, result):
        return (OperationStep(title="Adjoint", description="A* (conjugate transpose)", values=()),)


class CofactorOperation(OperationSpec):
    id = "cofactor"
    inputs = ("matrix",)
    outputs = ("matrix",)
    assumptions = ("square matrix",)

    def compute(self, inputs: Dict[str, Any], params: Dict[str, Any]) -> Any:
        tensor = inputs.get("tensor") or inputs.get("matrix") or (inputs.get("tensors") or [None])[0]
        if tensor is None:
            raise ValueError("Cofactor requires a matrix.")
        mat = tensor.to_numpy()
        n = mat.shape[0]
        cofactor = np.zeros_like(mat)
        for i in range(n):
            for j in range(n):
                minor = np.delete(np.delete(mat, i, axis=0), j, axis=1)
                cofactor[i, j] = ((-1) ** (i + j)) * np.linalg.det(minor)
        return cofactor

    def steps(self, inputs, params, result):
        return (OperationStep(title="Cofactor Matrix", description="cof(A)", values=()),)


class AdjugateOperation(OperationSpec):
    id = "adjugate"
    inputs = ("matrix",)
    outputs = ("matrix",)
    assumptions = ("square matrix",)

    def compute(self, inputs: Dict[str, Any], params: Dict[str, Any]) -> Any:
        tensor = inputs.get("tensor") or inputs.get("matrix") or (inputs.get("tensors") or [None])[0]
        if tensor is None:
            raise ValueError("Adjugate requires a matrix.")
        mat = tensor.to_numpy()
        n = mat.shape[0]
        cofactor = np.zeros_like(mat)
        for i in range(n):
            for j in range(n):
                minor = np.delete(np.delete(mat, i, axis=0), j, axis=1)
                cofactor[i, j] = ((-1) ** (i + j)) * np.linalg.det(minor)
        return cofactor.T  # Adjugate is transpose of cofactor

    def steps(self, inputs, params, result):
        return (OperationStep(title="Adjugate Matrix", description="adj(A) = cof(A)ᵀ", values=()),)


# ============================================================================
# DECOMPOSITIONS
# ============================================================================

class EigenOperation(OperationSpec):
    id = "eigen"
    inputs = ("matrix",)
    outputs = ("eigenvalues", "eigenvectors")
    assumptions = ("square matrix",)

    def compute(self, inputs: Dict[str, Any], params: Dict[str, Any]) -> Tuple[Any, Any]:
        tensor = inputs.get("tensor") or inputs.get("matrix") or (inputs.get("tensors") or [None])[0]
        if tensor is None:
            raise ValueError("Eigendecomposition requires a matrix.")
        mat = tensor.to_numpy()
        eigenvalues, eigenvectors = np.linalg.eig(mat)
        return (eigenvalues.real, eigenvectors.real)

    def steps(self, inputs, params, result):
        return (OperationStep(title="Eigendecomposition", description="A = VΛV⁻¹", values=()),)


class SVDOperation(OperationSpec):
    id = "svd"
    inputs = ("matrix",)
    outputs = ("U", "S", "Vt")
    assumptions = ()

    def compute(self, inputs: Dict[str, Any], params: Dict[str, Any]) -> Tuple[Any, Any, Any]:
        tensor = inputs.get("tensor") or inputs.get("matrix") or (inputs.get("tensors") or [None])[0]
        if tensor is None:
            raise ValueError("SVD requires a matrix.")
        mat = tensor.to_numpy()
        U, S, Vt = np.linalg.svd(mat)
        return (U, S, Vt)

    def steps(self, inputs, params, result):
        return (OperationStep(title="SVD", description="A = UΣVᵀ", values=()),)


class QROperation(OperationSpec):
    id = "qr"
    inputs = ("matrix",)
    outputs = ("Q", "R")
    assumptions = ()

    def compute(self, inputs: Dict[str, Any], params: Dict[str, Any]) -> Tuple[Any, Any]:
        tensor = inputs.get("tensor") or inputs.get("matrix") or (inputs.get("tensors") or [None])[0]
        if tensor is None:
            raise ValueError("QR decomposition requires a matrix.")
        mat = tensor.to_numpy()
        Q, R = np.linalg.qr(mat)
        return (Q, R)

    def steps(self, inputs, params, result):
        return (OperationStep(title="QR Decomposition", description="A = QR", values=()),)


class LUOperation(OperationSpec):
    id = "lu"
    inputs = ("matrix",)
    outputs = ("L", "U")
    assumptions = ("square matrix",)

    def compute(self, inputs: Dict[str, Any], params: Dict[str, Any]) -> Tuple[Any, Any]:
        tensor = inputs.get("tensor") or inputs.get("matrix") or (inputs.get("tensors") or [None])[0]
        if tensor is None:
            raise ValueError("LU decomposition requires a matrix.")
        mat = tensor.to_numpy()
        try:
            from scipy.linalg import lu
            P, L, U = lu(mat)
            return (L, U)
        except ImportError:
            # Fallback: simple LU without pivoting
            n = mat.shape[0]
            L = np.eye(n)
            U = mat.copy().astype(float)
            for i in range(n):
                for j in range(i + 1, n):
                    if U[i, i] != 0:
                        L[j, i] = U[j, i] / U[i, i]
                        U[j, :] -= L[j, i] * U[i, :]
            return (L, U)

    def steps(self, inputs, params, result):
        return (OperationStep(title="LU Decomposition", description="A = LU", values=()),)


class CholeskyOperation(OperationSpec):
    id = "cholesky"
    inputs = ("matrix",)
    outputs = ("L",)
    assumptions = ("square matrix", "positive definite")

    def compute(self, inputs: Dict[str, Any], params: Dict[str, Any]) -> Any:
        tensor = inputs.get("tensor") or inputs.get("matrix") or (inputs.get("tensors") or [None])[0]
        if tensor is None:
            raise ValueError("Cholesky decomposition requires a matrix.")
        mat = tensor.to_numpy()
        return np.linalg.cholesky(mat)

    def steps(self, inputs, params, result):
        return (OperationStep(title="Cholesky", description="A = LLᵀ", values=()),)


class SchurOperation(OperationSpec):
    id = "schur"
    inputs = ("matrix",)
    outputs = ("T", "Z")
    assumptions = ("square matrix",)

    def compute(self, inputs: Dict[str, Any], params: Dict[str, Any]) -> Tuple[Any, Any]:
        tensor = inputs.get("tensor") or inputs.get("matrix") or (inputs.get("tensors") or [None])[0]
        if tensor is None:
            raise ValueError("Schur decomposition requires a matrix.")
        mat = tensor.to_numpy()
        try:
            from scipy.linalg import schur
            T, Z = schur(mat)
            return (T.real, Z.real)
        except ImportError:
            # Fallback: return eigendecomposition-based approximation
            eigenvalues, eigenvectors = np.linalg.eig(mat)
            return (np.diag(eigenvalues.real), eigenvectors.real)

    def steps(self, inputs, params, result):
        return (OperationStep(title="Schur Decomposition", description="A = ZTZ*", values=()),)


# ============================================================================
# EIGENVALUES & VECTORS
# ============================================================================

class EigenvaluesOperation(OperationSpec):
    id = "eigenvalues"
    inputs = ("matrix",)
    outputs = ("eigenvalues",)
    assumptions = ("square matrix",)

    def compute(self, inputs: Dict[str, Any], params: Dict[str, Any]) -> Any:
        tensor = inputs.get("tensor") or inputs.get("matrix") or (inputs.get("tensors") or [None])[0]
        if tensor is None:
            raise ValueError("Eigenvalues requires a matrix.")
        mat = tensor.to_numpy()
        eigenvalues = np.linalg.eigvals(mat)
        return eigenvalues.real

    def steps(self, inputs, params, result):
        return (OperationStep(title="Eigenvalues", description="λ values", values=()),)


class EigenvectorsOperation(OperationSpec):
    id = "eigenvectors"
    inputs = ("matrix",)
    outputs = ("eigenvectors",)
    assumptions = ("square matrix",)

    def compute(self, inputs: Dict[str, Any], params: Dict[str, Any]) -> Any:
        tensor = inputs.get("tensor") or inputs.get("matrix") or (inputs.get("tensors") or [None])[0]
        if tensor is None:
            raise ValueError("Eigenvectors requires a matrix.")
        mat = tensor.to_numpy()
        _, eigenvectors = np.linalg.eig(mat)
        return eigenvectors.real

    def steps(self, inputs, params, result):
        return (OperationStep(title="Eigenvectors", description="v vectors", values=()),)


class SpectralRadiusOperation(OperationSpec):
    id = "spectral_radius"
    inputs = ("matrix",)
    outputs = ("scalar",)
    assumptions = ("square matrix",)

    def compute(self, inputs: Dict[str, Any], params: Dict[str, Any]) -> float:
        tensor = inputs.get("tensor") or inputs.get("matrix") or (inputs.get("tensors") or [None])[0]
        if tensor is None:
            raise ValueError("Spectral radius requires a matrix.")
        mat = tensor.to_numpy()
        eigenvalues = np.linalg.eigvals(mat)
        return float(np.max(np.abs(eigenvalues)))

    def steps(self, inputs, params, result):
        return (OperationStep(title="Spectral Radius", description=f"ρ(A) = {result:.6g}", values=(("rho", result),)),)


class PowerIterationOperation(OperationSpec):
    id = "power_iteration"
    inputs = ("matrix",)
    outputs = ("vector",)
    assumptions = ("square matrix",)

    def compute(self, inputs: Dict[str, Any], params: Dict[str, Any]) -> Any:
        tensor = inputs.get("tensor") or inputs.get("matrix") or (inputs.get("tensors") or [None])[0]
        if tensor is None:
            raise ValueError("Power iteration requires a matrix.")
        mat = tensor.to_numpy()
        n = mat.shape[0]
        v = np.ones(n) / np.sqrt(n)
        for _ in range(100):  # Fixed iterations
            v_new = mat @ v
            v_new = v_new / np.linalg.norm(v_new)
            if np.allclose(v, v_new):
                break
            v = v_new
        return v

    def steps(self, inputs, params, result):
        return (OperationStep(title="Power Iteration", description="Dominant eigenvector", values=()),)


# ============================================================================
# CHANGE OF BASIS
# ============================================================================

class ChangeBasisOperation(OperationSpec):
    id = "change_basis"
    inputs = ("matrix_A", "matrix_P")
    outputs = ("matrix",)
    assumptions = ("P is invertible",)

    def compute(self, inputs: Dict[str, Any], params: Dict[str, Any]) -> Any:
        tensors = inputs.get("tensors") or ()
        if len(tensors) < 2:
            raise ValueError("Change of basis requires A and P matrices.")
        A, P = tensors[0].to_numpy(), tensors[1].to_numpy()
        P_inv = np.linalg.inv(P)
        return P_inv @ A @ P

    def steps(self, inputs, params, result):
        return (OperationStep(title="Change of Basis", description="P⁻¹AP", values=()),)


class SimilarityTransformOperation(OperationSpec):
    id = "similarity_transform"
    inputs = ("matrix_A", "matrix_S")
    outputs = ("matrix",)
    assumptions = ("S is invertible",)

    def compute(self, inputs: Dict[str, Any], params: Dict[str, Any]) -> Any:
        tensors = inputs.get("tensors") or ()
        if len(tensors) < 2:
            raise ValueError("Similarity transform requires A and S matrices.")
        A, S = tensors[0].to_numpy(), tensors[1].to_numpy()
        S_inv = np.linalg.inv(S)
        return S_inv @ A @ S

    def steps(self, inputs, params, result):
        return (OperationStep(title="Similarity Transform", description="S⁻¹AS", values=()),)


class OrthogonalizeOperation(OperationSpec):
    id = "orthogonalize"
    inputs = ("matrix",)
    outputs = ("matrix",)
    assumptions = ()

    def compute(self, inputs: Dict[str, Any], params: Dict[str, Any]) -> Any:
        tensor = inputs.get("tensor") or inputs.get("matrix") or (inputs.get("tensors") or [None])[0]
        if tensor is None:
            raise ValueError("Orthogonalize requires a matrix.")
        mat = tensor.to_numpy()
        Q, _ = np.linalg.qr(mat)
        return Q

    def steps(self, inputs, params, result):
        return (OperationStep(title="Orthogonalize", description="Gram-Schmidt (via QR)", values=()),)


class ProjectSubspaceOperation(OperationSpec):
    id = "project_subspace"
    inputs = ("matrix_A", "matrix_B")
    outputs = ("matrix",)
    assumptions = ()

    def compute(self, inputs: Dict[str, Any], params: Dict[str, Any]) -> Any:
        tensors = inputs.get("tensors") or ()
        if len(tensors) < 2:
            raise ValueError("Project subspace requires two matrices.")
        A, B = tensors[0].to_numpy(), tensors[1].to_numpy()
        # Project A onto column space of B: B(B'B)^(-1)B'A
        proj_matrix = B @ np.linalg.pinv(B.T @ B) @ B.T
        return proj_matrix @ A

    def steps(self, inputs, params, result):
        return (OperationStep(title="Project Subspace", description="Project onto col(B)", values=()),)


# ============================================================================
# NORMS & DISTANCES
# ============================================================================

class FrobeniusNormOperation(OperationSpec):
    id = "frobenius_norm"
    inputs = ("matrix",)
    outputs = ("scalar",)
    assumptions = ()

    def compute(self, inputs: Dict[str, Any], params: Dict[str, Any]) -> float:
        tensor = inputs.get("tensor") or inputs.get("matrix") or (inputs.get("tensors") or [None])[0]
        if tensor is None:
            raise ValueError("Frobenius norm requires a matrix.")
        mat = tensor.to_numpy()
        return float(np.linalg.norm(mat, 'fro'))

    def steps(self, inputs, params, result):
        return (OperationStep(title="Frobenius Norm", description=f"‖A‖F = {result:.6g}", values=(("norm", result),)),)


class L1NormOperation(OperationSpec):
    id = "l1_norm"
    inputs = ("matrix",)
    outputs = ("scalar",)
    assumptions = ()

    def compute(self, inputs: Dict[str, Any], params: Dict[str, Any]) -> float:
        tensor = inputs.get("tensor") or inputs.get("matrix") or (inputs.get("tensors") or [None])[0]
        if tensor is None:
            raise ValueError("L1 norm requires a matrix.")
        mat = tensor.to_numpy()
        return float(np.linalg.norm(mat, 1))

    def steps(self, inputs, params, result):
        return (OperationStep(title="L1 Norm", description=f"‖A‖₁ = {result:.6g}", values=(("norm", result),)),)


class L2NormOperation(OperationSpec):
    id = "l2_norm"
    inputs = ("matrix",)
    outputs = ("scalar",)
    assumptions = ()

    def compute(self, inputs: Dict[str, Any], params: Dict[str, Any]) -> float:
        tensor = inputs.get("tensor") or inputs.get("matrix") or (inputs.get("tensors") or [None])[0]
        if tensor is None:
            raise ValueError("L2 norm requires a matrix.")
        mat = tensor.to_numpy()
        return float(np.linalg.norm(mat, 2))

    def steps(self, inputs, params, result):
        return (OperationStep(title="L2 Norm", description=f"‖A‖₂ = {result:.6g}", values=(("norm", result),)),)


class InfNormOperation(OperationSpec):
    id = "inf_norm"
    inputs = ("matrix",)
    outputs = ("scalar",)
    assumptions = ()

    def compute(self, inputs: Dict[str, Any], params: Dict[str, Any]) -> float:
        tensor = inputs.get("tensor") or inputs.get("matrix") or (inputs.get("tensors") or [None])[0]
        if tensor is None:
            raise ValueError("Infinity norm requires a matrix.")
        mat = tensor.to_numpy()
        return float(np.linalg.norm(mat, np.inf))

    def steps(self, inputs, params, result):
        return (OperationStep(title="Infinity Norm", description=f"‖A‖∞ = {result:.6g}", values=(("norm", result),)),)


class NuclearNormOperation(OperationSpec):
    id = "nuclear_norm"
    inputs = ("matrix",)
    outputs = ("scalar",)
    assumptions = ()

    def compute(self, inputs: Dict[str, Any], params: Dict[str, Any]) -> float:
        tensor = inputs.get("tensor") or inputs.get("matrix") or (inputs.get("tensors") or [None])[0]
        if tensor is None:
            raise ValueError("Nuclear norm requires a matrix.")
        mat = tensor.to_numpy()
        return float(np.linalg.norm(mat, 'nuc'))

    def steps(self, inputs, params, result):
        return (OperationStep(title="Nuclear Norm", description=f"‖A‖* = {result:.6g}", values=(("norm", result),)),)


# ============================================================================
# VECTOR OPERATIONS
# ============================================================================

class NormalizeOperation(OperationSpec):
    id = "normalize"
    inputs = ("vector",)
    outputs = ("vector",)
    assumptions = ()

    def compute(self, inputs: Dict[str, Any], params: Dict[str, Any]) -> Any:
        tensor = inputs.get("tensor") or inputs.get("vector") or (inputs.get("tensors") or [None])[0]
        if tensor is None:
            raise ValueError("Normalize requires a vector.")
        vec = tensor.to_numpy()
        norm = np.linalg.norm(vec)
        if norm == 0:
            return vec
        return vec / norm

    def steps(self, inputs, params, result):
        return (OperationStep(title="Normalize", description="v̂ = v / ‖v‖", values=()),)


class DotOperation(OperationSpec):
    id = "dot"
    inputs = ("vector_a", "vector_b")
    outputs = ("scalar",)
    assumptions = ("vectors are same length",)

    def compute(self, inputs: Dict[str, Any], params: Dict[str, Any]) -> float:
        vectors = inputs.get("vectors") or ()
        if len(vectors) < 2:
            raise ValueError("Dot requires two vectors.")
        v1, v2 = vectors[0], vectors[1]
        arr1 = np.array(v1.coords, dtype="f4")
        arr2 = np.array(v2.coords, dtype="f4")
        if arr1.shape != arr2.shape:
            raise ValueError("Vector dimensions must match.")
        return float(arr1.dot(arr2))

    def steps(self, inputs: Dict[str, Any], params: Dict[str, Any], result: Any) -> Sequence[OperationStep]:
        vectors = inputs.get("vectors") or ()
        if len(vectors) < 2:
            return ()
        v1, v2 = vectors[0], vectors[1]
        a = np.array(v1.coords, dtype="f4")
        b = np.array(v2.coords, dtype="f4")
        if a.shape != b.shape:
            return ()

        steps: List[OperationStep] = []
        running = 0.0
        for idx in range(len(a)):
            product = float(a[idx] * b[idx])
            running += product
            steps.append(OperationStep(
                title=f"Multiply component {idx + 1}",
                description=f"{a[idx]:.4g} × {b[idx]:.4g} = {product:.4g}",
                values=(
                    ("a", float(a[idx])),
                    ("b", float(b[idx])),
                    ("product", product),
                    ("sum", float(running)),
                ),
                render_vectors=(
                    RenderVectorHint(
                        coords=(float(running), 0.0, 0.0),
                        color=(0.95, 0.9, 0.2),
                        label="dot sum",
                    ),
                ),
            ))

        steps.append(OperationStep(
            title="Dot product",
            description=f"Result = {float(result):.4g}",
            values=(("dot", float(result)),),
            render_vectors=(
                RenderVectorHint(
                    coords=(float(result), 0.0, 0.0),
                    color=(0.95, 0.9, 0.2),
                    label="dot result",
                ),
            ),
        ))
        return steps


class CrossOperation(OperationSpec):
    id = "cross"
    inputs = ("vector_a", "vector_b")
    outputs = ("vector",)
    assumptions = ("3D vectors",)

    def compute(self, inputs: Dict[str, Any], params: Dict[str, Any]) -> Any:
        tensors = inputs.get("tensors") or inputs.get("vectors") or ()
        if len(tensors) < 2:
            raise ValueError("Cross product requires two vectors.")
        a, b = tensors[0].to_numpy(), tensors[1].to_numpy()
        return np.cross(a, b)

    def steps(self, inputs, params, result):
        return (OperationStep(title="Cross Product", description="a × b", values=()),)


class ProjectOperation(OperationSpec):
    id = "project"
    inputs = ("vector_a", "vector_b")
    outputs = ("vector",)
    assumptions = ()

    def compute(self, inputs: Dict[str, Any], params: Dict[str, Any]) -> Any:
        tensors = inputs.get("tensors") or inputs.get("vectors") or ()
        if len(tensors) < 2:
            raise ValueError("Projection requires two vectors.")
        a, b = tensors[0].to_numpy(), tensors[1].to_numpy()
        # Project a onto b: (a·b / b·b) * b
        scalar = np.dot(a, b) / np.dot(b, b)
        return scalar * b

    def steps(self, inputs, params, result):
        return (OperationStep(title="Project", description="proj_b(a)", values=()),)


class RejectOperation(OperationSpec):
    id = "reject"
    inputs = ("vector_a", "vector_b")
    outputs = ("vector",)
    assumptions = ()

    def compute(self, inputs: Dict[str, Any], params: Dict[str, Any]) -> Any:
        tensors = inputs.get("tensors") or inputs.get("vectors") or ()
        if len(tensors) < 2:
            raise ValueError("Rejection requires two vectors.")
        a, b = tensors[0].to_numpy(), tensors[1].to_numpy()
        # Rejection is a - projection
        scalar = np.dot(a, b) / np.dot(b, b)
        projection = scalar * b
        return a - projection

    def steps(self, inputs, params, result):
        return (OperationStep(title="Reject", description="rej_b(a) = a - proj_b(a)", values=()),)


class AngleBetweenOperation(OperationSpec):
    id = "angle_between"
    inputs = ("vector_a", "vector_b")
    outputs = ("scalar",)
    assumptions = ()

    def compute(self, inputs: Dict[str, Any], params: Dict[str, Any]) -> float:
        tensors = inputs.get("tensors") or inputs.get("vectors") or ()
        if len(tensors) < 2:
            raise ValueError("Angle requires two vectors.")
        a, b = tensors[0].to_numpy(), tensors[1].to_numpy()
        cos_angle = np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))
        cos_angle = np.clip(cos_angle, -1, 1)  # Handle numerical errors
        return float(np.arccos(cos_angle))  # Returns radians

    def steps(self, inputs, params, result):
        degrees = np.degrees(result)
        return (OperationStep(title="Angle Between", description=f"θ = {degrees:.2f}°", values=(("angle_rad", result), ("angle_deg", degrees))),)


# ============================================================================
# LINEAR SYSTEMS
# ============================================================================

class GaussianEliminationOperation(OperationSpec):
    id = "gaussian_elimination"
    inputs = ("matrix",)
    outputs = ("matrix",)
    assumptions = ("augmented matrix [A|b]",)

    def compute(self, inputs: Dict[str, Any], params: Dict[str, Any]) -> Any:
        tensor = inputs.get("tensor") or inputs.get("matrix") or (inputs.get("tensors") or [None])[0]
        if tensor is None:
            raise ValueError("Gaussian elimination requires a matrix.")
        mat = tensor.to_numpy().astype(float).copy()
        rows, cols = mat.shape

        # Forward elimination
        for i in range(min(rows, cols)):
            # Find pivot
            max_row = i
            for k in range(i + 1, rows):
                if abs(mat[k, i]) > abs(mat[max_row, i]):
                    max_row = k
            mat[[i, max_row]] = mat[[max_row, i]]

            # Eliminate below
            if mat[i, i] != 0:
                for k in range(i + 1, rows):
                    factor = mat[k, i] / mat[i, i]
                    mat[k, i:] -= factor * mat[i, i:]

        return mat

    def steps(self, inputs: Dict[str, Any], params: Dict[str, Any], result: Any) -> Sequence[OperationStep]:
        return (
            OperationStep(
                title="Gaussian elimination",
                description="Row echelon form via forward elimination",
                values=(),
            ),
        )


class RREFOperation(OperationSpec):
    id = "rref"
    inputs = ("matrix",)
    outputs = ("matrix",)
    assumptions = ()

    def compute(self, inputs: Dict[str, Any], params: Dict[str, Any]) -> Any:
        tensor = inputs.get("tensor") or inputs.get("matrix") or (inputs.get("tensors") or [None])[0]
        if tensor is None:
            raise ValueError("RREF requires a matrix.")
        mat = tensor.to_numpy().astype(float).copy()
        rows, cols = mat.shape

        pivot_row = 0
        for col in range(cols):
            if pivot_row >= rows:
                break

            # Find pivot
            max_row = pivot_row
            for k in range(pivot_row + 1, rows):
                if abs(mat[k, col]) > abs(mat[max_row, col]):
                    max_row = k

            if abs(mat[max_row, col]) < 1e-10:
                continue

            mat[[pivot_row, max_row]] = mat[[max_row, pivot_row]]

            # Scale pivot row
            mat[pivot_row] = mat[pivot_row] / mat[pivot_row, col]

            # Eliminate above and below
            for k in range(rows):
                if k != pivot_row:
                    mat[k] -= mat[k, col] * mat[pivot_row]

            pivot_row += 1

        return mat

    def steps(self, inputs, params, result):
        return (OperationStep(title="RREF", description="Reduced Row Echelon Form", values=()),)


class BackSubstitutionOperation(OperationSpec):
    id = "back_substitution"
    inputs = ("matrix",)
    outputs = ("vector",)
    assumptions = ("upper triangular augmented matrix",)

    def compute(self, inputs: Dict[str, Any], params: Dict[str, Any]) -> Any:
        tensor = inputs.get("tensor") or inputs.get("matrix") or (inputs.get("tensors") or [None])[0]
        if tensor is None:
            raise ValueError("Back substitution requires a matrix.")
        mat = tensor.to_numpy().astype(float)
        n = mat.shape[0]
        x = np.zeros(n)

        for i in range(n - 1, -1, -1):
            if abs(mat[i, i]) < 1e-10:
                continue
            x[i] = (mat[i, -1] - np.dot(mat[i, i+1:n], x[i+1:n])) / mat[i, i]

        return x

    def steps(self, inputs, params, result):
        return (OperationStep(title="Back Substitution", description="Solve from bottom up", values=()),)


class SolveLinearOperation(OperationSpec):
    id = "solve_linear"
    inputs = ("matrix_A", "vector_b")
    outputs = ("vector",)
    assumptions = ("Ax = b",)

    def compute(self, inputs: Dict[str, Any], params: Dict[str, Any]) -> Any:
        tensors = inputs.get("tensors") or ()
        if len(tensors) < 2:
            raise ValueError("Solve requires A and b.")
        A, b = tensors[0].to_numpy(), tensors[1].to_numpy()
        return np.linalg.solve(A, b)

    def steps(self, inputs, params, result):
        return (OperationStep(title="Solve Ax=b", description="x = A⁻¹b", values=()),)


class LeastSquaresOperation(OperationSpec):
    id = "least_squares"
    inputs = ("matrix_A", "vector_b")
    outputs = ("vector",)
    assumptions = ()

    def compute(self, inputs: Dict[str, Any], params: Dict[str, Any]) -> Any:
        tensors = inputs.get("tensors") or ()
        if len(tensors) < 2:
            raise ValueError("Least squares requires A and b.")
        A, b = tensors[0].to_numpy(), tensors[1].to_numpy()
        result, _, _, _ = np.linalg.lstsq(A, b, rcond=None)
        return result

    def steps(self, inputs, params, result):
        return (OperationStep(title="Least Squares", description="x̂ = (AᵀA)⁻¹Aᵀb", values=()),)


# ============================================================================
# SPECIAL MATRICES
# ============================================================================

class SymmetrizeOperation(OperationSpec):
    id = "symmetrize"
    inputs = ("matrix",)
    outputs = ("matrix",)
    assumptions = ()

    def compute(self, inputs: Dict[str, Any], params: Dict[str, Any]) -> Any:
        tensor = inputs.get("tensor") or inputs.get("matrix") or (inputs.get("tensors") or [None])[0]
        if tensor is None:
            raise ValueError("Symmetrize requires a matrix.")
        mat = tensor.to_numpy()
        return (mat + mat.T) / 2

    def steps(self, inputs, params, result):
        return (OperationStep(title="Symmetrize", description="(A + Aᵀ)/2", values=()),)


class SkewSymmetrizeOperation(OperationSpec):
    id = "skew_symmetrize"
    inputs = ("matrix",)
    outputs = ("matrix",)
    assumptions = ()

    def compute(self, inputs: Dict[str, Any], params: Dict[str, Any]) -> Any:
        tensor = inputs.get("tensor") or inputs.get("matrix") or (inputs.get("tensors") or [None])[0]
        if tensor is None:
            raise ValueError("Skew-symmetrize requires a matrix.")
        mat = tensor.to_numpy()
        return (mat - mat.T) / 2

    def steps(self, inputs, params, result):
        return (OperationStep(title="Skew-Symmetrize", description="(A - Aᵀ)/2", values=()),)


class DiagonalizeOperation(OperationSpec):
    id = "diagonalize"
    inputs = ("matrix",)
    outputs = ("matrix",)
    assumptions = ("square matrix",)

    def compute(self, inputs: Dict[str, Any], params: Dict[str, Any]) -> Any:
        tensor = inputs.get("tensor") or inputs.get("matrix") or (inputs.get("tensors") or [None])[0]
        if tensor is None:
            raise ValueError("Diagonalize requires a matrix.")
        mat = tensor.to_numpy()
        return np.diag(np.diag(mat))

    def steps(self, inputs, params, result):
        return (OperationStep(title="Diagonalize", description="Extract diagonal", values=()),)


class UpperTriangularOperation(OperationSpec):
    id = "triangular_upper"
    inputs = ("matrix",)
    outputs = ("matrix",)
    assumptions = ()

    def compute(self, inputs: Dict[str, Any], params: Dict[str, Any]) -> Any:
        tensor = inputs.get("tensor") or inputs.get("matrix") or (inputs.get("tensors") or [None])[0]
        if tensor is None:
            raise ValueError("Upper triangular requires a matrix.")
        mat = tensor.to_numpy()
        return np.triu(mat)

    def steps(self, inputs, params, result):
        return (OperationStep(title="Upper Triangular", description="triu(A)", values=()),)


class LowerTriangularOperation(OperationSpec):
    id = "triangular_lower"
    inputs = ("matrix",)
    outputs = ("matrix",)
    assumptions = ()

    def compute(self, inputs: Dict[str, Any], params: Dict[str, Any]) -> Any:
        tensor = inputs.get("tensor") or inputs.get("matrix") or (inputs.get("tensors") or [None])[0]
        if tensor is None:
            raise ValueError("Lower triangular requires a matrix.")
        mat = tensor.to_numpy()
        return np.tril(mat)

    def steps(self, inputs, params, result):
        return (OperationStep(title="Lower Triangular", description="tril(A)", values=()),)


# ============================================================================
# LEGACY OPERATIONS (for backwards compatibility)
# ============================================================================

class MatrixVectorMultiplyOperation(OperationSpec):
    id = "matrix_vector_multiply"
    inputs = ("matrix", "vector")
    outputs = ("vector",)
    assumptions = ("matrix columns match vector length",)

    def compute(self, inputs: Dict[str, Any], params: Dict[str, Any]) -> Tuple[float, ...]:
        matrix = inputs.get("matrix")
        vector = inputs.get("vector")
        if matrix is None or vector is None:
            raise ValueError("Matrix-vector multiply requires matrix and vector.")
        mat_np = matrix.to_numpy()
        vec_np = vector.to_numpy()
        if mat_np.shape[1] != vec_np.shape[0]:
            raise ValueError("Matrix columns must match vector length.")
        result = mat_np @ vec_np
        return tuple(float(x) for x in result)

    def steps(self, inputs: Dict[str, Any], params: Dict[str, Any], result: Any) -> Sequence[OperationStep]:
        matrix = inputs.get("matrix")
        vector = inputs.get("vector")
        if matrix is None or vector is None:
            return ()
        mat_np = matrix.to_numpy()
        vec_np = vector.to_numpy()
        if mat_np.shape[1] != vec_np.shape[0]:
            return ()

        row_color = getattr(matrix, "color", (0.6, 0.8, 1.0))
        input_color = getattr(vector, "color", (0.8, 0.8, 0.8))
        output_color = (0.95, 0.9, 0.2)

        def to_vec3(values):
            vals = list(values)
            if len(vals) < 3:
                vals = vals + [0.0] * (3 - len(vals))
            return tuple(float(v) for v in vals[:3])

        steps: List[OperationStep] = []
        running = [0.0 for _ in range(mat_np.shape[0])]
        for idx in range(mat_np.shape[0]):
            row = mat_np[idx]
            dot_val = float(np.dot(row, vec_np))
            running[idx] = dot_val
            partial = running.copy()
            steps.append(OperationStep(
                title=f"Row {idx + 1} · input",
                description=f"Row {idx + 1} dot input = {dot_val:.4g}",
                values=(("row_dot", dot_val),),
                render_vectors=(
                    RenderVectorHint(
                        coords=to_vec3(row),
                        color=row_color,
                        label=f"row {idx + 1}",
                    ),
                    RenderVectorHint(
                        coords=to_vec3(vec_np),
                        color=input_color,
                        label="input v",
                    ),
                    RenderVectorHint(
                        coords=to_vec3(partial),
                        color=output_color,
                        label="partial output",
                    ),
                ),
            ))

        return steps


# ============================================================================
# REGISTRY INITIALIZATION
# ============================================================================

registry = OperationRegistry()

# Basic Arithmetic
registry.register(AddOperation())
registry.register(SubtractOperation())
registry.register(ScaleOperation())
registry.register(NegateOperation())
registry.register(HadamardOperation())

# Matrix Multiplication
registry.register(MatrixMultiplyOperation())
registry.register(OuterProductOperation())
registry.register(KroneckerOperation())

# Matrix Properties
registry.register(DeterminantOperation())
registry.register(TraceOperation())
registry.register(RankOperation())
registry.register(ConditionNumberOperation())
registry.register(NullityOperation())

# Transformations
registry.register(TransposeOperation())
registry.register(InverseOperation())
registry.register(PseudoinverseOperation())
registry.register(AdjointOperation())
registry.register(CofactorOperation())
registry.register(AdjugateOperation())

# Decompositions
registry.register(EigenOperation())
registry.register(SVDOperation())
registry.register(QROperation())
registry.register(LUOperation())
registry.register(CholeskyOperation())
registry.register(SchurOperation())

# Eigenvalues & Vectors
registry.register(EigenvaluesOperation())
registry.register(EigenvectorsOperation())
registry.register(SpectralRadiusOperation())
registry.register(PowerIterationOperation())

# Change of Basis
registry.register(ChangeBasisOperation())
registry.register(SimilarityTransformOperation())
registry.register(OrthogonalizeOperation())
registry.register(ProjectSubspaceOperation())

# Norms & Distances
registry.register(FrobeniusNormOperation())
registry.register(L1NormOperation())
registry.register(L2NormOperation())
registry.register(InfNormOperation())
registry.register(NuclearNormOperation())

# Vector Operations
registry.register(NormalizeOperation())
registry.register(DotOperation())
registry.register(CrossOperation())
registry.register(ProjectOperation())
registry.register(RejectOperation())
registry.register(AngleBetweenOperation())

# Linear Systems
registry.register(GaussianEliminationOperation())
registry.register(RREFOperation())
registry.register(BackSubstitutionOperation())
registry.register(SolveLinearOperation())
registry.register(LeastSquaresOperation())

# Special Matrices
registry.register(SymmetrizeOperation())
registry.register(SkewSymmetrizeOperation())
registry.register(DiagonalizeOperation())
registry.register(UpperTriangularOperation())
registry.register(LowerTriangularOperation())

# Legacy
registry.register(MatrixVectorMultiplyOperation())

# Print registered operations count
print(f"[CVLA] Registered {len(registry.list_all())} operations")
