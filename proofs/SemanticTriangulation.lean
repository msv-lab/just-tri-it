import Mathlib.Data.Set.Basic

/-definition 2.1-/


-- D: type of problem descriptions
-- P: type of "forward" programs (solve D)
-- Q: type of "transformed" programs (solve T D)
structure Triangulation (D D' P Q : Type) where
  T     : D → D'
  Phi   : P → Q → Prop
  /-semantic equivalence on the original side -/
  EP    : Setoid P

  /- Injectivity on classes: if the same q matches two p’s by Φ, then those two p’s are EP-equivalent. -/
  injective  :
    ∀ {p₁ p₂ q}, Phi p₁ q → Phi p₂ q → EP.r p₁ p₂

  /- Surjectivity on classes (onto the relevant image): for every q we can find a p with Φ -/
  surjective :
    ∀ q, ∃ p, Phi p q

  /-- Existence of a mate for every p. -/
  totality   :
    ∀ p, ∃ q, Phi p q

  /-- Correctness predicates (what it means to "solve" a task). -/
  CorrectP : D → P → Prop
  CorrectQ : D' → Q → Prop

  /- Correctness coupling : if q correctly solves T d and Φ(p,q) holds,
      then p correctly solves d. -/
  correctness_coupling :
    ∀ {d p q}, CorrectQ (T d) q → Phi p q → CorrectP d p










/-
   Proposition 4.5 - triangulation generalizes plurality
-/

-- We will need that correctness is stable under ≡ on P. if p ≡ q,both correct or incorrect
def CorrectRespects (D P : Type) (EP : Setoid P)
    (Correct : D → P → Prop) : Prop :=
  ∀ {d p q}, EP.r p q → (Correct d p ↔ Correct d q)

def pluralityTriangulation
  (D P : Type)(EP : Setoid P)(Correct : D → P → Prop)(resp : CorrectRespects D P EP Correct)
  : Triangulation D D P P :=
{ T := id , Phi := fun p q => EP.r p q , EP  := EP ,
    injective := by
    intro p₁ p₂ q h1 h2
    -- h1: Phi p₁ q (p₁ ≡ q) , h2: Phi p₂ q (p₂ ≡ q)
    exact EP.trans h1 (EP.symm h2),

    surjective := by
    intro q;
    exact ⟨q, EP.refl q⟩,

    totality := by
    intro p;
    exact ⟨p, EP.refl p⟩,

    CorrectP := Correct, CorrectQ := fun d' p => Correct d' p ,

    correctness_coupling := by
    intro d p q hQ hPhi
    have hiff := resp (d:=d) (p:=p) (q:=q) hPhi
    -- hiff : Correct d p ↔ Correct d q
    -- from Correct d q, conclude Correct d p
    exact Iff.mpr hiff hQ }











/-
  Proposition 5.2 — Full FWD_INV
-/


section Full_FWD_INV

variable {A B : Type}

abbrev Spec (A B : Type) := A → B → Prop
def T_inv (R : Spec A B) : Spec B A := fun b a => R a b


def CorrectF (R : Spec A B) (e : Equiv A B) : Prop := ∀ a, R a (e a)
def CorrectG (R' : Spec B A) (e' : Equiv B A) : Prop := ∀ b, R' b (e' b)


def tri_FULL_FWD_INV
  : Triangulation (Spec A B) (Spec B A) (Equiv A B) (Equiv B A) :=
{ T := T_inv, Phi := fun e e' => e' = e.symm
, EP :=
{ r := fun (e₁ e₂ : Equiv A B) => e₁ = e₂
, iseqv := by
    refine ⟨?refl, ?symm, ?trans⟩
    · intro x; rfl
    · intro x y h; simpa using h.symm
    · intro x y z h₁ h₂; simpa using h₁.trans h₂
}


, injective := by
    intro e1 e2 e' h1 h2
    have h : e2 = e1 := by
      simpa [h2] using congrArg Equiv.symm h1
    exact h.symm


, surjective := by
    intro e'
    exact ⟨e'.symm, rfl⟩

, totality := by
    intro e
    exact ⟨e.symm, rfl⟩

, CorrectP := fun R e => CorrectF R e
, CorrectQ := fun R' e' => CorrectG R' e'

, correctness_coupling := by
    intro R e e' hG hPhi a
    subst hPhi
    simpa using hG (e a)
}

end Full_FWD_INV










/-
  Proposition 5.4 — Partial FWD_INV
-/

section Partial_FWD_INV



variable {I₁ I₂ O : Type}


def T_inv1 (d : Spec (I₁ × I₂) O) : Spec (O × I₂) I₁ :=
  fun (oi : O × I₂) (i₁ : I₁) => d (i₁, oi.2) oi.1


abbrev PFwd (I₁ I₂ O : Type) := I₂ → Equiv I₁ O
abbrev PInv (I₁ I₂ O : Type) := I₂ → Equiv O I₁



def setoidPFwd {I₁ I₂ O : Type} : Setoid (PFwd I₁ I₂ O) where
  r p₁ p₂ := ∀ i₂ i₁, (p₁ i₂) i₁ = (p₂ i₂) i₁
  iseqv := {
    refl := fun _ _ _ => rfl
    symm := fun h i₂ i₁ => (h i₂ i₁).symm
    trans := fun h₁ h₂ i₂ i₁ => (h₁ i₂ i₁).trans (h₂ i₂ i₁)
  }



def Phi_partial (p : PFwd I₁ I₂ O) (q : PInv I₁ I₂ O) : Prop :=
  ∀ i₁ i₂, (q i₂) ((p i₂) i₁) = i₁


def CorrectP (d : Spec (I₁ × I₂) O) (p : PFwd I₁ I₂ O) : Prop :=
  ∀ i₁ i₂, d (i₁, i₂) ((p i₂) i₁)
def CorrectQ (d' : Spec (O × I₂) I₁) (q : PInv I₁ I₂ O) : Prop :=
  ∀ o i₂, d' (o, i₂) ((q i₂) o)


def tri_partial_FWD_INV
  : Triangulation (Spec (I₁ × I₂) O) (Spec (O × I₂) I₁) (PFwd I₁ I₂ O) (PInv I₁ I₂ O) :=
{ T := T_inv1, Phi := Phi_partial, EP := setoidPFwd


, injective := by
    intro p₁ p₂ q h1 h2 i₂ i₁
    have inj : Function.Injective (q i₂) := (q i₂).injective
    have e : (q i₂) ((p₁ i₂) i₁) = (q i₂) ((p₂ i₂) i₁) := by
      calc
        (q i₂) ((p₁ i₂) i₁) = i₁ := h1 i₁ i₂
        _ = (q i₂) ((p₂ i₂) i₁) := (h2 i₁ i₂).symm
    exact inj e


, surjective := by
    intro q
    refine ⟨(fun i₂ => (q i₂).symm), ?_⟩
    intro i₁ i₂
    exact Equiv.apply_symm_apply (q i₂) i₁



, totality := by
    intro p
    refine ⟨(fun i₂ => (p i₂).symm), ?_⟩
    intro i₁ i₂
    exact Equiv.symm_apply_apply (p i₂) i₁



, CorrectP := CorrectP, CorrectQ := CorrectQ
, correctness_coupling := by
    intro d p q hQ hPhi i₁ i₂
    have := hQ ((p i₂) i₁) i₂
    simpa [T_inv1, hPhi i₁ i₂] using this
}
end Partial_FWD_INV






/-
  Proposition 5.6 — Full FWD_SINV
-/


section Full_FWD_SINV


variable {I O : Type}

def T_swap (d : Spec I O) : Spec O I := fun o i => d i o


structure SINV (I O : Type) where
  q      : O → Set I
  cover  : ∀ i : I, ∃ o : O, i ∈ q o
  unique : ∀ i o₁ o₂, i ∈ q o₁ → i ∈ q o₂ → o₁ = o₂





def setoidFun (X Y : Type) : Setoid (X → Y) where
  r f g := ∀ x, f x = g x
  iseqv :=
    { refl  := by intro f x; rfl
      symm  := by intro f g h x; simpa using (h x).symm
      trans := by intro f g h h₁ h₂ x; exact (h₁ x).trans (h₂ x) }



-- φ：L1 ∧ L2
def Phi_SINV (p : I → O) (q : SINV I O) : Prop :=
  (∀ i, i ∈ q.q (p i)) ∧ (∀ i i', i' ∈ q.q (p i) → p i' = p i)




def CorrectPSINV (d : Spec I O) (p : I → O) : Prop :=
  ∀ i, d i (p i)
def CorrectQSINV (d' : Spec O I) (q : SINV I O) : Prop :=
  ∀ o i, i ∈ q.q o → d' o i


def sinvOfP (p : I → O) : SINV I O :=
{ q := fun o => { i | p i = o }
, cover := by
    intro i; exact ⟨p i, rfl⟩
, unique := by
    intro i o₁ o₂ hi₁ hi₂
    -- hi₁ : p i = o₁, hi₂ : p i = o₂
    exact hi₁.symm.trans hi₂ }


noncomputable def pOfSINV (q : SINV I O) : I → O :=
  fun i => Classical.choose (q.cover i)

lemma pOfSINV_L1 (q : SINV I O) :
  ∀ i, i ∈ q.q (pOfSINV q i) := by
  intro i
  exact Classical.choose_spec (q.cover i)
lemma pOfSINV_L2 (q : SINV I O) :
  ∀ i i', i' ∈ q.q (pOfSINV q i) → pOfSINV q i' = pOfSINV q i := by
  intro i i' hi'
  have hi'cov : i' ∈ q.q (pOfSINV q i') := pOfSINV_L1 q i'
  have := q.unique i' (pOfSINV q i') (pOfSINV q i) hi'cov hi'
  simp [this]


def tri_FWD_SINV
  : Triangulation (Spec I O) (Spec O I) (I → O) (SINV I O) :=
{ T := T_swap, Phi := Phi_SINV, EP := setoidFun I O

, injective := by
    intro p₁ p₂ q h1 h2 i
    have hi1 : i ∈ q.q (p₁ i) := h1.left i
    have hi2 : i ∈ q.q (p₂ i) := h2.left i
    exact q.unique i (p₁ i) (p₂ i) hi1 hi2

, surjective := by
    intro q
    refine ⟨pOfSINV q, ?_⟩
    refine ⟨?_, ?_⟩
    · exact pOfSINV_L1 q
    · exact pOfSINV_L2 q

, totality := by
    intro p
    refine ⟨sinvOfP p, ?_⟩
    constructor
    · intro i
      -- i ∈ { i | p i = p i }
      simp [sinvOfP]
    · intro i i' hi'
      -- hi' : p i' = p i
      simpa [sinvOfP] using hi'

, CorrectP := CorrectPSINV, CorrectQ := CorrectQSINV

, correctness_coupling := by
    intro d p q hQ hPhi i
    have hi : i ∈ q.q (p i) := hPhi.left i
    simpa [T_swap] using hQ (p i) i hi
}

end Full_FWD_SINV










/-
  Proposition 5.8 — Partial FWD_SINV
-/



namespace PARTIAL_FWD_SINV

variable {A Y B : Type}

abbrev Fwd  (A Y B : Type) := A → Y → B
abbrev SINV  (A Y B : Type) := B → Y → Set A
abbrev SpecPSINV (A Y B : Type) := B → Y → Set A → Prop
abbrev Spec (A Y B : Type) := A → Y → B → Prop


def T_pfib_partial (R : Spec A Y B) : SpecPSINV A Y B :=
  fun b y S => ∀ x, x ∈ S → R x y b


def Phi_pSINV (f : Fwd A Y B) (g : SINV A Y B) : Prop :=
  (∀ x y, x ∈ g (f x y) y) ∧
  (∀ x y x', x' ∈ g (f x y) y → f x' y = f x y)


structure Resonator (A Y B : Type) where
  g    : SINV A Y B
  spec : ∃ f : Fwd A Y B, Phi_pSINV f g
  unique : ∀ (y : Y) (x : A) (b₁ b₂ : B), x ∈ g b₁ y → x ∈ g b₂ y → b₁ = b₂

def EP_Fwd2 : Setoid (Fwd A Y B) :=
{ r := fun f₁ f₂ => ∀ x y, f₁ x y = f₂ x y,
  iseqv := ⟨(fun _ _ _ => rfl), (fun h x y => (h x y).symm),
            (fun h₁ h₂ x y => (h₁ x y).trans (h₂ x y))⟩ }



def CorrectF2 (R : Spec A Y B) (f : Fwd A Y B) : Prop :=
  ∀ x y, R x y (f x y)
def CorrectG2 (R' : SpecPSINV A Y B) (g : Resonator A Y B) : Prop :=
  ∀ b y, R' b y (g.g b y)



def triang_PARTIAL_FWD_SINV :
  Triangulation (Spec A Y B) (SpecPSINV A Y B)
                (Fwd A Y B) (Resonator A Y B) :=
{ T := T_pfib_partial
, Phi := fun f g => Phi_pSINV f g.g, EP := EP_Fwd2

, injective := by
    intro f₁ f₂ g h1 h2 x y
    -- L1: x ∈ g (f₁ x y) y
    have hx : x ∈ g.g (f₁ x y) y := h1.left x y
    have hx2 : x ∈ g.g (f₂ x y) y := h2.left x y
    exact g.unique y x (f₁ x y) (f₂ x y) hx hx2


, surjective := by
    intro g
    rcases g.spec with ⟨f, hf⟩
    exact ⟨f, hf⟩

, totality := by
    intro f
    let g : SINV A Y B := fun b y => {x | f x y = b}
    have hΦ : Phi_pSINV f g := by
      constructor
      · intro x y; simp [g]
      · intro x y x' hx'; simpa [g] using hx'
    have hunique : ∀ (y : Y) (x : A) (b₁ b₂ : B), x ∈ g b₁ y → x ∈ g b₂ y → b₁ = b₂ := by
      intro y x b₁ b₂ hb₁ hb₂
      exact hb₁.symm.trans hb₂
    exact ⟨⟨g, ⟨f, hΦ⟩, hunique⟩, hΦ⟩


, CorrectP := fun R f => CorrectF2 R f, CorrectQ := fun R' g => CorrectG2 R' g

, correctness_coupling := by
    intro d f g hG hPhi x y
    have hsound := hG (f x y) y
    have hx : x ∈ g.g (f x y) y := hPhi.left x y
    exact hsound x hx
}

end PARTIAL_FWD_SINV











/-
 Proposition 5.10 - Full ENUM_SINV
-/


namespace Full_ENUM_SINV


variable {I O : Type}


abbrev SpecRel   (I O : Type) := I → O → Prop
abbrev SpecEnum  (I O : Type) := I → Set O → Prop   -- d^≺
abbrev SpecSInv  (I O : Type) := O → Set I → Prop   -- d^≻

-- τ₁, τ₂
def tau1 (R : SpecRel I O) : SpecEnum I O :=
  fun i S => S = { o | R i o }
def tau2 (R : SpecRel I O) : SpecSInv I O :=
  fun o S => S = { i | R i o }




structure EnumDesc (I O : Type) where
  R : SpecRel I O
  d : SpecEnum I O
  repr : d = tau1 (I:=I) (O:=O) R
structure SInvDesc (I O : Type) where
  R : SpecRel I O
  d : SpecSInv I O
  repr : d = tau2 (I:=I) (O:=O) R




-- T = τ₂ ∘ τ₁⁻¹
def T_enum_sinv (de : EnumDesc I O) : SInvDesc I O :=
  { R := de.R, d := tau2 (I:=I) (O:=O) de.R, repr := rfl }



def Phi (p : I → Set O) (q : O → Set I) : Prop :=
  ∀ i o, o ∈ p i ↔ i ∈ q o



def EP : Setoid (I → Set O) :=
{ r := fun p₁ p₂ => ∀ i, p₁ i = p₂ i
, iseqv := ⟨(fun _ _ => rfl),
            (fun h i => (h i).symm),
            (fun h₁ h₂ i => (h₁ i).trans (h₂ i))⟩ }



def CorrectEnum (de : EnumDesc I O) (p : I → Set O) : Prop :=
  ∀ i, de.d i (p i)
def CorrectSInv (ds : SInvDesc I O) (q : O → Set I) : Prop :=
  ∀ o, ds.d o (q o)




def tri_Full_ENUM_SINV
  : Triangulation (EnumDesc I O) (SInvDesc I O) (I → Set O) (O → Set I) :=
{ T := T_enum_sinv, Phi := Phi, EP := EP

, injective := by
    intro p₁ p₂ q h1 h2 i
    apply Set.ext
    intro o
    exact (h1 i o).trans (Iff.symm (h2 i o))

, surjective := by
    intro q
    refine ⟨(fun i => { o | i ∈ q o }), ?_⟩
    intro i o
    rfl

, totality := by
    intro p
    refine ⟨(fun o => { i | o ∈ p i }), ?_⟩
    intro i o
    rfl

, CorrectP := CorrectEnum, CorrectQ := CorrectSInv

, correctness_coupling := by
    intro de p q hQ hPhi
    let R := de.R
    have hde : de.d = tau1 (I:=I) (O:=O) R := de.repr
    have hQ' : ∀ o, q o = { i | R i o } := by
      intro o
      have : (T_enum_sinv de).d o (q o) := hQ o
      simp [T_enum_sinv, tau2] at this
      exact this
    intro i
    have : p i = { o | R i o } := by
      apply Set.ext; intro o
      have φ := hPhi i o
      have iInQ : i ∈ q o ↔ R i o := by
        rw [hQ' o]
        rfl
      exact φ.trans iInQ
    rw [hde]
    simp [tau1, this]
}

end Full_ENUM_SINV









/-
 Proposition 5.11 - Partial ENUM_SINV
-/


namespace Partial_ENUM_SINV

variable {I₁ I₂ O : Type}

abbrev SpecRel   (I₁ I₂ O : Type) := I₁ → I₂ → O → Prop
abbrev SpecEnum  (I₁ I₂ O : Type) := I₁ → I₂ → Set O  → Prop
abbrev SpecSInv  (I₁ I₂ O : Type) := O   → I₂ → Set I₁ → Prop



-- τ₁, τ₂
def tau1 (R : SpecRel I₁ I₂ O) : SpecEnum I₁ I₂ O :=
  fun i₁ i₂ S => S = { o | R i₁ i₂ o }
def tau2 (R : SpecRel I₁ I₂ O) : SpecSInv I₁ I₂ O :=
  fun o i₂ S => S = { i₁ | R i₁ i₂ o }



structure EnumDesc (I₁ I₂ O : Type) where
  R : SpecRel I₁ I₂ O
  d : SpecEnum I₁ I₂ O
  repr : d = tau1 (I₁:=I₁) (I₂:=I₂) (O:=O) R
structure SInvDesc (I₁ I₂ O : Type) where
  R : SpecRel I₁ I₂ O
  d : SpecSInv I₁ I₂ O
  repr : d = tau2 (I₁:=I₁) (I₂:=I₂) (O:=O) R



-- T = τ₂ ∘ τ₁⁻¹
def T_enum_sinv (de : EnumDesc I₁ I₂ O) : SInvDesc I₁ I₂ O :=
  { R := de.R, d := tau2 (I₁:=I₁) (I₂:=I₂) (O:=O) de.R, repr := rfl }


def Phi (p : I₁ → I₂ → Set O) (q : O → I₂ → Set I₁) : Prop :=
  ∀ i₁ i₂ o, o ∈ p i₁ i₂ ↔ i₁ ∈ q o i₂




def EP : Setoid (I₁ → I₂ → Set O) :=
{ r := fun p₁ p₂ => ∀ i₁ i₂, p₁ i₁ i₂ = p₂ i₁ i₂
, iseqv := ⟨(fun _ _ _ => rfl),
            (fun h i₁ i₂ => (h i₁ i₂).symm),
            (fun h₁ h₂ i₁ i₂ => (h₁ i₁ i₂).trans (h₂ i₁ i₂))⟩ }




def CorrectEnum (de : EnumDesc I₁ I₂ O) (p : I₁ → I₂ → Set O) : Prop :=
  ∀ i₁ i₂, de.d i₁ i₂ (p i₁ i₂)
def CorrectSInv (ds : SInvDesc I₁ I₂ O) (q : O → I₂ → Set I₁) : Prop :=
  ∀ o i₂, ds.d o i₂ (q o i₂)




def tri_Partial_ENUM_SINV
  : Triangulation (EnumDesc I₁ I₂ O) (SInvDesc I₁ I₂ O) (I₁ → I₂ → Set O) (O → I₂ → Set I₁) :=
{ T := T_enum_sinv, Phi := Phi, EP := EP

, injective := by
    intro p₁ p₂ q h1 h2 i₁ i₂
    apply Set.ext; intro o
    exact (h1 i₁ i₂ o).trans (Iff.symm (h2 i₁ i₂ o))

, surjective := by
    intro q
    refine ⟨(fun i₁ i₂ => { o | i₁ ∈ q o i₂ }), ?_⟩
    intro i₁ i₂ o; rfl

, totality := by
    intro p
    refine ⟨(fun o i₂ => { i₁ | o ∈ p i₁ i₂ }), ?_⟩
    intro i₁ i₂ o; rfl

, CorrectP := CorrectEnum, CorrectQ := CorrectSInv

, correctness_coupling := by
    intro de p q hQ hPhi
    let R := de.R
    have hde : de.d = tau1 (I₁:=I₁) (I₂:=I₂) (O:=O) R := de.repr
    have hQ' : ∀ o i₂, q o i₂ = { i₁ | R i₁ i₂ o } := by
      intro o i₂
      have : (T_enum_sinv de).d o i₂ (q o i₂) := hQ o i₂
      simp [T_enum_sinv, tau2] at this
      exact this
    intro i₁ i₂
    have p_eq : p i₁ i₂ = { o | R i₁ i₂ o } := by
      apply Set.ext; intro o
      have φ := hPhi i₁ i₂ o
      have iInQ : i₁ ∈ q o i₂ ↔ R i₁ i₂ o := by
        rw [hQ' o i₂]
        rfl
      exact φ.trans iInQ
    rw [hde]
    simp [tau1, p_eq]
}

end Partial_ENUM_SINV







/-
 Proposition 5.12 - STREAM
-/


namespace STREAM

variable {I O : Type}


structure SpecDesc (I O : Type) where
  R : I → O → Prop


structure SpecSeqDesc (I O : Type) where
  R : List I → List O → Prop
  stateless : ∀ (p0 : I → O), (∀ i, R [i] [p0 i]) → ∀ l, R l (l.map p0)


structure StreamHom (I O : Type) where
  p : List I → List O
  p0   : I → O
  hom  : ∀ l, p l = l.map p0



def EP_STREAM {D' Q : Type}
  (base_tri : Triangulation (SpecDesc I O) D' (I → O) Q) :
  Setoid (StreamHom I O) where
  r h1 h2 := base_tri.EP.r h1.p0 h2.p0
  iseqv :=
  { refl  := by
      intro h
      exact base_tri.EP.refl h.p0
    symm  := by
      intro h1 h2 h
      exact base_tri.EP.symm h
    trans := by
      intro h1 h2 h3 h12 h23
      exact base_tri.EP.trans h12 h23 }




variable {D' Q : Type}
variable (tri_pt : Triangulation (SpecDesc I O) D' (I → O) Q)



def T_pointwise (d : SpecSeqDesc I O) : SpecDesc I O := { R := fun i o => d.R [i] [o] }
def T := fun d => tri_pt.T (T_pointwise d)
def Phi (h : StreamHom I O) (q : Q) := tri_pt.Phi h.p0 q ∧ ∀ l, h.p l = l.map h.p0


def CorrectP_STREAM (d : SpecSeqDesc I O) (h : StreamHom I O) : Prop := ∀ l, d.R l (h.p l)
def CorrectQ_STREAM (d' : D') (q : Q) : Prop := tri_pt.CorrectQ d' q



def triangulation_STREAM (base_tri : Triangulation (SpecDesc I O) D' (I → O) Q)

(h_CorrectP_def : ∀ (d : SpecDesc I O) (p : I → O), base_tri.CorrectP d p ↔ (∀ i, d.R i (p i)))

  : Triangulation (SpecSeqDesc I O) D' (StreamHom I O) Q :=
{ T  := fun d => base_tri.T (T_pointwise d)
, Phi := fun h q => base_tri.Phi h.p0 q ∧ ∀ l, h.p l = l.map h.p0
, EP  := EP_STREAM base_tri


, injective := by
    intro h1 h2 q h1q h2q
    exact base_tri.injective (p₁:=h1.p0) (p₂:=h2.p0) (q:=q) h1q.left h2q.left

, surjective := by
    intro q; rcases base_tri.surjective q with ⟨p0, hp⟩
    exact ⟨⟨fun l => l.map p0, p0, by intro l; rfl⟩, And.intro hp (by intro l; rfl)⟩

, totality := by
    intro h; rcases base_tri.totality h.p0 with ⟨q, hp⟩
    exact ⟨q, And.intro hp h.hom⟩

, CorrectP := CorrectP_STREAM
, CorrectQ := fun d' q => base_tri.CorrectQ d' q

, correctness_coupling := by
      intro d h q hQ hΦ l
      have maplaw : ∀ l, h.p l = l.map h.p0 := hΦ.right
      have pc0 : base_tri.CorrectP (T_pointwise d) h.p0 :=
        base_tri.correctness_coupling hQ hΦ.left
      rw [h_CorrectP_def] at pc0
      have singletons : ∀ i, d.R [i] [h.p0 i] := by
        intro i
        simpa [T_pointwise] using pc0 i
      have stream_ok : d.R l (l.map h.p0) :=
        d.stateless h.p0 singletons l
      simpa [maplaw l] using stream_ok

}


end STREAM
