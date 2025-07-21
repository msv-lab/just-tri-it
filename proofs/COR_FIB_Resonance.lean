import Mathlib.Data.Set.Basic
import Mathlib.Data.Set.Function
import Mathlib.Logic.Function.Basic
open Set

variable {A B : Type} (R : Set (A × B))  -- Specification relation

-- 沿用您之前的定义
def function_image {A B : Type} (f : A → Set B) (s : Set A) : Set B :=
  { b | ∃ a ∈ s, b ∈ f a }

def function_image' {A B : Type} (g : B → Set A) (t : Set B) : Set A :=
  { a | ∃ b ∈ t, a ∈ g b }

noncomputable def COR_FIB_simplified {A B : Type}
    (f : A → Set B) (g : B → Set A) : Prop :=
  (∀ b : B,
    (∃ a ∈ g b, b ∈ f a) ∧
    (∀ a1 a2, a1 ∈ g b → a2 ∈ g b → f a1 = f a2)
  )
  ∧
  (∀ a : A,
    (∃ b ∈ f a, a ∈ g b) ∧
    (∀ b1 b2, b1 ∈ f a → b2 ∈ f a → g b1 = g b2)
  )

-- 正确性定义：多值函数版本
def correct_R_multi (f : A → Set B) : Prop :=
  ∀ a, ∀ b ∈ f a, (a, b) ∈ R

def correct_inv_multi (g : B → Set A) : Prop :=
  ∀ b, ∀ a ∈ g b, (a, b) ∈ R

-- 主要定理：COR-FIB 保证共振性
theorem resonance_COR_FIB (f : A → Set B) (g : B → Set A) :
    COR_FIB_simplified f g →
    (correct_R_multi R f ∧ correct_inv_multi R g) ∨
    (¬ correct_R_multi R f ∧ ¬ correct_inv_multi R g) := by
  intro h_cor_fib
  by_cases h_f_correct : correct_R_multi R f

  case pos =>
    -- 情况1：f 正确
    left
    constructor
    · exact h_f_correct
    · -- 证明 g 也正确
      intro b a ha_in_gb
      -- 从 COR-FIB 的第一个条件获取可达性
      have ⟨a_witness, ha_witness_in_gb, hb_in_fa_witness⟩ := h_cor_fib.1 b |>.1

      -- 使用纤维内一致性
      have f_eq : f a = f a_witness := h_cor_fib.1 b |>.2 a a_witness ha_in_gb ha_witness_in_gb

      -- 由于 f 正确，(a_witness, b) ∈ R
      have witness_correct : (a_witness, b) ∈ R := h_f_correct a_witness b hb_in_fa_witness

      -- 由于 f a = f a_witness，且 b ∈ f a_witness，所以 b ∈ f a
      rw [←f_eq] at hb_in_fa_witness

      -- 因此 (a, b) ∈ R（由于 f 正确）
      exact h_f_correct a b hb_in_fa_witness

  case neg =>
    -- 情况2：f 不正确
    right
    constructor
    · exact h_f_correct
    · -- 证明 g 也不正确
      intro h_g_correct
      -- 反证法：假设 g 正确，推出矛盾
      apply h_f_correct
      intro a b hb_in_fa

      -- 从 COR-FIB 的第二个条件获取可覆盖性
      have ⟨b_witness, hb_witness_in_fa, ha_in_gb_witness⟩ := h_cor_fib.2 a |>.1

      -- 使用输出内一致性
      have g_eq : g b = g b_witness := h_cor_fib.2 a |>.2 b b_witness hb_in_fa hb_witness_in_fa

      -- 由于 g 正确，(a, b_witness) ∈ R
      have witness_correct : (a, b_witness) ∈ R := h_g_correct b_witness a ha_in_gb_witness

      -- 由于 g b = g b_witness，且 a ∈ g b_witness，所以 a ∈ g b
      rw [←g_eq] at ha_in_gb_witness

      -- 因此 (a, b) ∈ R（由于 g 正确）
      exact h_g_correct b a ha_in_gb_witness
