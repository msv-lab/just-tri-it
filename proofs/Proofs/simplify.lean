import Mathlib.Data.Set.Basic
import Mathlib.Data.Set.Function
import Mathlib.Logic.Function.Basic
open Set

def function_image {A B : Type} (f : A → Set B) (s : Set A) : Set B :=
  { b | ∃ a ∈ s, b ∈ f a }

def function_image' {A B : Type} (g : B → Set A) (t : Set B) : Set A :=
  { a | ∃ b ∈ t, a ∈ g b }

noncomputable def COR_FIB {A B : Type}
    (f : A → Set B)  (g : B → Set A) : Prop :=
  (∀ b : B,
    -- g(b) = g(f(g(b)))
    g b = function_image' g (function_image f (g b)) ∧

    -- b ∈ f(g(b))
    (∃ a ∈ g b, b ∈ f a) ∧

    (∀ a1 a2, a1 ∈ g b → a2 ∈ g b → f a1 = f a2)
  )
  ∧
  (∀ a : A,
    -- f(a) = f(g(f(a)))
    f a = function_image f (function_image' g (f a)) ∧

    -- a ∈ g(f(a))
    (∃ b ∈ f a, a ∈ g b) ∧

    (∀ b1 b2, b1 ∈ f a → b2 ∈ f a → g b1 = g b2)
  )

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

theorem stability_redundant
    (h : COR_FIB_simplified f g) :
    (∀ b, g b = function_image' g (function_image f (g b))) ∧
    (∀ a, f a = function_image f (function_image' g (f a))) := by
  constructor

  -- 第一部分：证明 ∀ b, g b = function_image' g (function_image f (g b))
  · intro b
    -- 从 COR_FIB_simplified 获取基本条件
    have h_reach : ∃ a ∈ g b, b ∈ f a := h.1 b |>.1
    have h_cons : ∀ a1 a2, a1 ∈ g b → a2 ∈ g b → f a1 = f a2 := h.1 b |>.2

    -- 选择特定的 a_h
    obtain ⟨a_h, ha_h_in_gb, hb_in_fa_h⟩ := h_reach

    -- 证明集合相等
    apply Set.ext
    intro x
    constructor

    -- 方向1: g b ⊆ function_image' g (function_image f (g b))
    · intro hx_in_gb
      -- 展开 function_image' 和 function_image 的定义
      use b
      constructor
      -- 需要证明 b ∈ function_image f (g b)
      · use a_h, ha_h_in_gb
      -- 需要证明 x ∈ g b
      · -- 由于 b ∈ f a_h，我们需要找到 b' ∈ f x 使得 x ∈ g b'

        -- 关键洞察：由于纤维内一致性，f x = f a_h
        have f_x_eq : f x = f a_h := h_cons x a_h hx_in_gb ha_h_in_gb

        -- 由于 b ∈ f a_h 且 f x = f a_h，所以 b ∈ f x
        rw [←f_x_eq] at hb_in_fa_h

        -- 现在使用第二个条件的可覆盖性
        have h_cover : ∃ b_temp ∈ f x, x ∈ g b_temp := h.2 x |>.1
        obtain ⟨b_temp, hb_temp_in_fx, hx_in_gb_temp⟩ := h_cover

        -- 使用输出内一致性
        have g_eq : g b = g b_temp := h.2 x |>.2 b b_temp hb_in_fa_h hb_temp_in_fx
        rw [g_eq]
        exact hx_in_gb_temp

    -- 方向2: function_image' g (function_image f (g b)) ⊆ g b
    · intro hx_in_img
      -- 展开定义
      obtain ⟨y, hy_in_img_f, hx_in_gy⟩ := hx_in_img
      obtain ⟨a, ha_in_gb, hy_in_fa⟩ := hy_in_img_f

      -- 使用纤维内一致性
      have f_a_eq : f a = f a_h := h_cons a a_h ha_in_gb ha_h_in_gb
      rw [f_a_eq] at hy_in_fa

      -- 使用输出内一致性
      have g_eq : g y = g b := h.2 a_h |>.2 y b hy_in_fa hb_in_fa_h
      rw [←g_eq]
      exact hx_in_gy

  -- 第二部分：证明 ∀ a, f a = function_image f (function_image' g (f a))
  · intro a
    -- 从 COR_FIB_simplified 获取基本条件
    have h_cover : ∃ b ∈ f a, a ∈ g b := h.2 a |>.1
    have h_cons : ∀ b1 b2, b1 ∈ f a → b2 ∈ f a → g b1 = g b2 := h.2 a |>.2

    -- 选择特定的 b_h
    obtain ⟨b_h, hb_h_in_fa, ha_in_gb_h⟩ := h_cover

    -- 证明集合相等
    apply Set.ext
    intro y
    constructor

    -- 方向1: f a ⊆ function_image f (function_image' g (f a))
    · intro hy_in_fa
      -- 展开 function_image 和 function_image' 的定义
      use a
      constructor
      -- 需要证明 a ∈ function_image' g (f a)
      · use b_h, hb_h_in_fa
      -- 需要证明 y ∈ f a
      · exact hy_in_fa

    -- 方向2: function_image f (function_image' g (f a)) ⊆ f a
    · intro hy_in_img
      -- 展开定义
      obtain ⟨a', ha'_in_img, hy_in_fa'⟩ := hy_in_img
      obtain ⟨b, hb_in_fa, ha'_in_gb⟩ := ha'_in_img

      -- 使用输出内一致性
      have g_eq : g b = g b_h := h_cons b b_h hb_in_fa hb_h_in_fa
      rw [g_eq] at ha'_in_gb

      -- 使用纤维内一致性
      have f_eq : f a' = f a := h.1 b_h |>.2 a' a ha'_in_gb ha_in_gb_h
      rw [f_eq] at hy_in_fa'
      exact hy_in_fa'
