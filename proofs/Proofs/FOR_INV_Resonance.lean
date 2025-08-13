import Mathlib.Data.Set.Basic  --Basic operations and definitions about sets
import Mathlib.Data.Set.Function  --About functions
import Mathlib.Logic.Function.Basic --Provides the logic of functions, such as identity, injection or surjection.

variable {A B : Type} (R : Set (A × B))  -- Specification relation
/-
## 1. FOR-INV (Forward + Inverse)
 g ∘ f ≡ id_A ∧ f ∘ g ≡ id_B
Liberal : g ∘ f ≡ id_A
-/
def FOR_INV (f : A → B) (g : B → A) : Prop :=
  (∀ a, g (f a) = a) ∧ (∀ b, f (g b) = b)
-- The definition also indicates that f and g are mutually inverse functions

def correct_R (f : A → B) : Prop := ∀ a, (a, f a) ∈ R
def correct_inv (g : B → A) : Prop := ∀ b, (g b, b) ∈ R

theorem resonance (f : A → B) (g : B → A)  :
    FOR_INV f g → (correct_R R f ∧ correct_inv R g) ∨ (¬ correct_R R f ∧ ¬ correct_inv R g):= by
    intro h_inv  -- assume the property holds
    by_cases h_f_correct : correct_R R f --branch according to whether f meet R
    case pos =>
      -- case1: f corect
      left   -- the left of ∨
      constructor      ---- A ∧ B generate two subgoals: one is to prove A, the other is to prove B.
      · exact h_f_correct
      · -- prove g is also correct: ∀ b, (g b, b) ∈ R
        intro b  --  (g b, b) = (g b, f(g b)) ∈ R
        have h_eq : b = f (g b) := (h_inv.2 b).symm -- f ∘ g = id
        convert h_f_correct (g b)  --  (g b, f(g b)) ∈ R
    case neg =>
      -- case2: f incorrect
      right      -- the right of ∨
      constructor
      · exact h_f_correct
      · -- prove g is also incorrect
        intro h_g_correct    -- ¬ Q's definition is Q → False  
        apply h_f_correct
        intro a   -- (a, f a) ∈ R
        have h_eq : a = g (f a) := (h_inv.1 a).symm  --  g ∘ f = id
        convert h_g_correct (f a) --  (g(f a), f a) ∈ R
