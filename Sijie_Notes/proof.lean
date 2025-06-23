import Mathlib.Data.Set.Basic
import Mathlib.Data.Set.Function
import Mathlib.Data.Fintype.Basic
import Mathlib.Logic.Function.Basic

variable {A B : Type} (R : Set (A × B))  -- Specification relation

/-
## 1. FOR-INV (Forward + Inverse)
 g ∘ f ≡ id_A ∧ f ∘ g ≡ id_B
Liberal : g ∘ f ≡ id_A
-/
def FOR_INV (f : A → B) (g : B → A) : Prop :=
  (∀ a, g (f a) = a) ∧ (∀ b, f (g b) = b)

-- f 满足需求 R
def correct_R (f : A → B) : Prop := ∀ a, (a, f a) ∈ R
def correct_inv (g : B → A) : Prop := ∀ b, (g b, b) ∈ R

theorem resonance_equivalence [Nonempty A](f0 : A → B) (hf0 : Function.Bijective f0)
    (hR : ∀ a b, (a, b) ∈ R ↔ b = f0 a) (f : A → B) (g : B → A) (h_inv : FOR_INV f g) :
    FOR_INV f g ↔
    (correct_R R f ∧ correct_inv R g) ∨ (¬ correct_R R f ∧ ¬ correct_inv R g) := by
  constructor
  -- forward  (⇒): FOR_INV → all correct or all incorrect
  · intro h_inv
    by_cases h_eq : f = f0
    · left
      constructor
      · intro a
        rw [h_eq]
        rw [hR]
      · intro b
        rw [hR]
        rw [h_eq] at h_inv
        exact (h_inv.2 b).symm
    -- case 2: incorrect
    · right
      constructor
      · intro h_correct
        apply h_eq; ext a
        rw [←hR a (f a)]
        apply h_correct
      · intro h_correct
        apply h_eq; ext a
        have := h_correct (f a)
        rw [hR] at this
        rw [this]
        congr 1
        exact h_inv.1 a
  -- backward  (⇐): all correct or all incorrect → FOR_INV
  · intro h_cases
    cases' h_cases with h_correct h_incorrect
    -- case 1: correct
    · constructor
      · intro a
        have h1 := h_correct.1 a
        have h2 := h_correct.2 (f a)
        rw [hR] at h1 h2
        exact (hf0.1 (h1.symm.trans h2)).symm
      · intro b
        have h1 := h_correct.2 b        -- (g b, b) ∈ R， b = f₀ (g b)
        have h2 := h_correct.1 (g b)    -- (g b, f (g b)) ∈ R， f (g b) = f₀ (g b)
        rw [hR] at h1 h2                -- h1: b = f₀ (g b), h2: f (g b) = f₀ (g b)
        exact h2.trans h1.symm
    -- case 2: incorrect
    · constructor
      · intro a
        by_cases h : g (f a) = a
        · exact h
        · exfalso
          exact h (h_inv.1 a)
      · intro b
        by_cases h : f (g b) = b
        · exact h
        · exfalso
          exact h (h_inv.2 b)
