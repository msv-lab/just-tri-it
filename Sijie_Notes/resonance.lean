import Mathlib.Data.Set.Basic
import Mathlib.Data.Set.Function
import Mathlib.Data.Fintype.Basic
import Mathlib.Logic.Function.Basic

open Classical
noncomputable section

variable {A B : Type} (R : Set (A × B))  -- Specification relation

/-
## 1. FOR-INV (Forward + Inverse)
 g ∘ f ≡ id_A ∧ f ∘ g ≡ id_B
Liberal : g ∘ f ≡ id_A
-/
def FOR_INV (f : A → B) (g : B → A) : Prop :=
  (∀ a, g (f a) = a) ∧ (∀ b, f (g b) = b)

def FOR_INV_Liberal (f : A → B) (g : B → A) : Prop :=
  ∀ a, g (f a) = a

/-
## 1. FOR-INV with Partial Inverse
-/
def PartialInverse {A1 A2 B : Type} (f : A1 × A2 → B) (g : B × A2 → A1) : Prop :=
  ∀ (x : A1) (y : A2), g (f (x, y), y) = x

def FOR_INV_Partial {A1 A2 B : Type} (R : Set (A1 × A2 × B))
  (f : A1 → A2 → B) (g : B → A2 → A1) : Prop :=
  ∀ (x : A1) (y : A2), (x, y, f x y) ∈ R →
    g (f x y) y = x

/-
## 2. FOR-FIB (Forward + Fiber)
  ∀ b, f(g(b)) = {b} ∧ ∀ a, a ∈ g(f(a)) (Page 17)
Liberal : ∀ a, a ∈ g(f(a)) (Page 17)
-/
-- Partial fiber function (liberal FOR-FIB)
def PartialFiber {A1 A2 B : Type}
  (f : A1 × A2 → B) (g : B × A2 → Set A1): Prop :=
  ∀ (x : A1) (y : A2), x ∈ g (f (x, y), y) ∧ ∀ x' ∈ g (f (x, y), y), f (x', y) = f (x, y)

/-
## 3. COR-FIB (Correspondence + Fiber)
  ∀ b,
    g(b) = g(f(g(b))) ∧
    b ∈ f(g(b)) ∧
    ∀ a' a'' ∈ g(b), f(a') = f(a'')
  ∧ ∀ a,
    f(a) = f(g(f(a))) ∧
    a ∈ g(f(a)) ∧
    ∀ b' b'' ∈ f(a), g(b') = g(b'')
-/

def COR_FIB {A B : Type} (f : A → Set B) (g : B → Set A) : Prop :=
  (∀ b : B,
    (∃ a : A, a ∈ g b ∧ b ∈ f a) ∧
    (∀ a1 a2 : A, a1 ∈ g b → a2 ∈ g b → f a1 = f a2) ∧
    (∃ a : A, a ∈ g b ∧ ∀ a', a' ∈ g b → a' ∈ g b → a' ∈ g b)  -- 避免写 f (g b)
  ) ∧
  (∀ a : A,
    (∃ b : B, b ∈ f a ∧ a ∈ g b) ∧
    (∀ b1 b2 : B, b1 ∈ f a → b2 ∈ f a → g b1 = g b2) ∧
    (∃ b : B, b ∈ f a ∧ ∀ b', b' ∈ f a → b' ∈ f a → b' ∈ f a)  -- 避免写 g (f a)
  )

/-the second way to formalize the COR-FIB property-/
noncomputable def COR_FIB2 {A B : Type} (f : A → Set B) (g : B → Set A) : Prop :=
(∀ b : B,
   (∃ a : A, a ∈ g b ∧ b ∈ f a) ∧
   (∀ a1 a2 : A, a1 ∈ g b → a2 ∈ g b → f a1 = f a2) ∧
   (∃ a0 ∈ g b, ∃ b' ∈ f a0, g b = g b')
) ∧
(∀ a : A,
   (∃ b : B, b ∈ f a ∧ a ∈ g b) ∧
   (∀ b1 b2 : B, b1 ∈ f a → b2 ∈ f a → g b1 = g b2) ∧
   (∃ b0 ∈ f a, ∃ a' ∈ g b0, f a = f a')
)
/-
## 4. VAL-FIB (Validator + Fiber)
  VAL-FIB for R = FOR-FIB for validator specification
-/
-- VAL-FIB: Validator version of FOR-FIB
-- f : A → B, p : A × B → Bool (i.e. validator)
-- g : B → Set A

def VAL_FIB {A B : Type} (f : A → B) (p : A → B → Bool) (g : B → Set A) : Prop :=
  ∀ (a : A), p a (f a) ∧ a ∈ g (f a) ∧ ∀ a' ∈ g (f a), p a' (f a)


 /-
 Regarding the cor-fib function, there have been errors in the formalised definition with lean4.
 In fact, I think the current writing method does not fully conform to the definition of properties in the presentations.
 I tried to use choose(a model in lean4) to express it, but there were some errors.
 This part may take time to solve. There may also be a problem with the definition of properties in this part of the presentation(just i think).
 -/
