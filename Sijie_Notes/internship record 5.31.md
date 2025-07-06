# internship record 5.31

## Symmetry

requirement R：Given an array x, return any one of its positive elements; if none exist, return 0

P：

```
def P (x : List Int) (z : Int) : Bool :=
  match list_max (x.filter (· > 0)) with
  | some m => z = m
  | none   => z = 0 
```

R^-1:For a given output value z, derive a valid input x such that the output f(x) would yield z. If z = 0, then x must contain no positive numbers.

P^-1:

```
def P_inv (z : Int) (x : List Int) : Bool :=
  match list_max (x.filter (· > 0)) with
  | some m => z = m
  | none   => z = 0
```

In this example, given an input *x*, we obtain an output *y* = *f*(*x*) through a certain function (or model). The predicate P(*x*, *y*) is used to judge whether 'this input-output pair is reasonable'. Conversely, when deriving *x* from *y* (i.e., *f*⁻¹(*y*) = *x*), the inverse predicate essentially also judges whether (*x*, *y*) is reasonable. The predicate and the inverse predicate are fundamentally checking the same relationship.

Goal: Verify whether function *f* correctly implements requirement R

Forward Verification:

For input *x*, compute *y* = *f*(*x*)

Use predicate *p*(*x*, *y*) to verify whether (*x*, *y*) satisfies requirement R

If *p*(*x*, *y*) = false → immediately conclude *f* is incorrect

If *p*(*x*, *y*) = true → only indicates current input passes; further verification required

Inverse Verification:

For output *y*, compute x' = *f*⁻¹(*y*) (where *f*⁻¹ is the inverse function generated from the inverse requirement)

Use inverse predicate *p*⁻¹(*y*, x') to verify whether (*y*, x') satisfies the inverse requirement

Resonance Verification:

If *p*(*x*, *y*) = true and *p*⁻¹(*y*, x') = true
→ indicates *f* and *f*⁻¹ exhibit behavioral consistency
→ *f* correctly implements the requirement

Further verify the resonance property (logical implication relationship) between *p* and *p*⁻¹

Current issue: When the verification logic for p and p⁻¹ is identical, resonance naturally occurs, resulting in loss of verification capability.

**We need to identify requirement scenarios where the predicate and inverse predicate both verify the same input-output pair (x, y) but use different verification logic.**

## Asymmetric 

1. 

R：for a rotation matrix A, verify that it is an orthogonal matrix.

```
Write a verification predicate based on the following requirements:
Requirement: For rotation matrix A, verify it is an orthogonal matrix.
Specification: The generated predicate should:

Accept a matrix as input

Check if it satisfies the definition of an orthogonal matrix: AᵀA = I
Provide the predicate in mathematical notation or pseudocode.
```



response：

```
                                         True ⟺ Aᵀ A = I
```

R^-1：for an orthogonal matrix A, verify it is a rotation matrix.

```
Write a verification predicate based on the following inverse requirement:
Inverse requirement: For orthogonal matrix A, verify it is a rotation matrix.

```



response:

```
               IsRotationMatrix(A) ⟺ Aᵀ A = I ∧ det(A)=1
```

2. 

 R：Given an orthogonal matrix A, determine if it is invertible.

P：det(A) is not zero

R^-1：Given an invertible matrix A, determine if it is orthogonal.

p^-1：A^-1 = A ^T, and det(A) is 1 or -1.



When I found requirements where the way to check the "forward predicate" and the "backward predicate" is different，I used to divide requirements into many types，After thinking about it many times，I think if a requirement has a "sufficient but not necessary" condition，then the way to check them can be different.，that is, the condition always leads to the conclusion，.but the conclusion leading back to the condition is not always true，and needs extra checking conditions.

If this requirement is reversible, has a "necessary and sufficient" condition, is one-to-one, symmetric，(meaning the condition always leads to the conclusion, and the conclusion always leads back to the condition)，then the core logic for checking its forward predicate and backward predicate is the same，but we can maybe use different methods for the forward check，(and finding the inverse is also really just making sure we have a different method for the same requirement)，For example, Requirement R: "Please create a matrix A that is invertible"，You could write two rules (predicates):The first rule is to directly check that A's determinant is not zero，The second one is to start from the definition，use A to find another matrix B，such that A times B equals the Identity matrix. These are two different ways to check，But they both come from the same forward requirement，And I think the "inverse requirement" for this one is also quite hard to come up with.



