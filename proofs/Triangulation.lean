import Mathlib.Data.Set.Basic

-- D: type of problem descriptions
-- P: type of "forward" programs (solve D)
-- Q: type of "transformed" programs (solve T D)
structure Triangulation (D D' P Q : Type) where
  T     : D → D'
  Phi   : P → Q → Prop
  /-semantic equivalence on the original side -/
  EP    : Setoid P
  /-coarser relation on the transformed side -/
  EQ    : Setoid Q

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
  Proposition 4.1 — plurality
-/

-- We will need that correctness is stable under ≡ on P. if p ≡ q,both correct or incorrect
def CorrectRespects (D P : Type) (EP : Setoid P)
    (Correct : D → P → Prop) : Prop :=
  ∀ {d p q}, EP.r p q → (Correct d p ↔ Correct d q)

def pluralityTriangulation
  (D P : Type)(EP : Setoid P)(Correct : D → P → Prop)(resp : CorrectRespects D P EP Correct)
  : Triangulation D D P P :=
{ T := id , Phi := fun p q => EP.r p q , EP  := EP , EQ  := EP,
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
  Proposition 5.2 — FOR_INV
-/

section FOR_INV

variable {A B : Type}

abbrev Spec (A B : Type) := A → B → Prop

def T_inv : Spec A B → Spec B A := fun R b a => R a b

/- phi ：g ∘ f = id_A , f ∘ g = id_B -/
structure IsoFG (f : A → B) (g : B → A) : Prop where
(left  : ∀ a, g (f a) = a)
(right : ∀ b, f (g b) = b)

/- EP,EQ -/
def extSetoid (X Y : Type) : Setoid (X → Y) where
  r f g := ∀ x, f x = g x
  iseqv := {
    refl := fun _ _ => rfl
    symm := fun h x => (h x).symm
    trans := fun h₁ h₂ x => (h₁ x).trans (h₂ x)
  }

variable (exists_g_for_f : ∀ f : A → B, ∃ g : B → A, IsoFG f g)
variable (exists_f_for_g : ∀ g : B → A, ∃ f : A → B, IsoFG f g)


def CorrectF_for_inv (R : Spec A B) (f : A → B) : Prop := ∀ a, R a (f a)
def CorrectG_for_inv (R : Spec A B) (g : B → A) : Prop := ∀ b, R (g b) b

def triang_FOR_INV
  : Triangulation (Spec A B) (Spec B A) (A → B) (B → A) :=
{ T  := T_inv , Phi := fun f g => IsoFG f g , EP  := extSetoid A B , EQ  := extSetoid B A

,  injective := by
    intro f₁ f₂ g h1 h2
    have inj_g : Function.Injective g := by
      intro b₁ b₂ h
      have := congrArg f₁ h
      simp [h1.right b₁, h1.right b₂] at this
      exact this
    intro a
    have g_eq : g (f₁ a) = g (f₂ a) := by simp [h1.left a, h2.left a]
    exact inj_g g_eq

, surjective := by
    intro g
    rcases exists_f_for_g g with ⟨f, hf⟩
    exact ⟨f, hf⟩

, totality := by
    intro f
    rcases exists_g_for_f f with ⟨g, hg⟩
    exact ⟨g, hg⟩,

    CorrectP := fun R f => CorrectF_for_inv R f, CorrectQ := fun R g => CorrectG_for_inv (T_inv R) g

, correctness_coupling := by
    intro R f g hG hIso a
    simpa [hIso.left a] using (hG (f a))
}

end FOR_INV






/-
  Proposition 5.5 — FOR-FIB (liberal)
-/

section FOR_FIB

variable {A B : Type}

abbrev Fwd (A B : Type) := A → B
abbrev SpecFib (A B : Type) := B → Set A → Prop

/-difine phi -/
def Phi_for_fib (f : Fwd A B) (g : B → Set A) : Prop :=
  (∀ b, ∀ a, a ∈ g b → f a = b) ∧ (∀ a, a ∈ g (f a))

/-- Resonator: only those fiber functions g that are compatible with some f. -/
structure Resonator (A B : Type) where
  g   : B → Set A
  spec : ∃ f : A → B, Phi_for_fib f g


/-- Extensional equality for forwards. -/
def EP_Fwd : Setoid (Fwd A B) :=
{ r := fun f₁ f₂ => ∀ a, f₁ a = f₂ a,
  iseqv := ⟨
    fun _ _ => rfl,
    fun h a => (h a).symm,
    fun h₁ h₂ a => (h₁ a).trans (h₂ a)
  ⟩ }
/-- Extensional equality for resonators. -/
def EQ_Res : Setoid (Resonator A B) :=
{ r := fun g₁ g₂ => ∀ b, g₁.g b = g₂.g b,
  iseqv := ⟨
    fun _ _ => rfl,
    fun h b => (h b).symm,
    fun h₁ h₂ b => (h₁ b).trans (h₂ b)
  ⟩ }

def T_fib (R : Spec A B) : SpecFib A B := fun b S => S = {a | R a b}


def CorrectF (R : Spec A B) (f : Fwd A B) : Prop := ∀ a, R a (f a)
def CorrectG (R' : SpecFib A B) (g : Resonator A B) : Prop := ∀ b, R' b (g.g b)



def triang_FOR_FIB :
  Triangulation (Spec A B) (SpecFib A B) (Fwd A B) (Resonator A B) :=
{ T := T_fib , Phi := fun f g => Phi_for_fib f g.g, EP := EP_Fwd, EQ := EQ_Res

, injective := by
    intro f₁ f₂ g h1 h2
    intro a
    have : a ∈ g.g (f₁ a) := (h1.right a)
    have eq2 : f₂ a = f₁ a := (h2.left (f₁ a) a this)
    exact eq2.symm

, surjective := by
    intro g
    rcases g.spec with ⟨f, hf⟩
    exact ⟨f, hf⟩

, totality := by
    intro f
    let g : B → Set A := fun b => {a | f a = b}
    have hΦ : Phi_for_fib f g := by
      constructor
      · intro b a ha; exact ha
      · intro a; simp [g]
    exact ⟨⟨g, ⟨f, hΦ⟩⟩, hΦ⟩,

  CorrectP := fun R f => CorrectF R f, CorrectQ := fun R' g => CorrectG R' g

, correctness_coupling := by
    intro d f g hG h0 a
    have ha : a ∈ g.g (f a) := h0.right a
    have eq : g.g (f a) = {a' | d a' (f a)} := hG (f a)
    rw [eq] at ha
    exact ha }

end FOR_FIB


lemma equiv
  {A B : Type} (f : A → B) (g : B → Set A) :
  ((∀ b ∈ Set.range f, f '' (g b) = {b}) ∧ (∀ a, a ∈ g (f a)))
  ↔
  (∀ a, f '' (g (f a)) = {f a} ∧ a ∈ g (f a)) := by
  classical
  constructor
  ·
    intro h a
    exact ⟨by simpa using h.1 (f a) ⟨a, rfl⟩, h.2 a⟩
  ·
    intro h
    refine ⟨?_, ?_⟩
    · intro b hb
      rcases hb with ⟨a, rfl⟩
      simpa using (h a).1
    · intro a
      simpa using (h a).2



/-
 Proposition 5.9 — Partial FOR_FIB
-/

section PARTIAL_FOR_FIB

variable {A Y B : Type}


abbrev Fwd2  (A Y B : Type) := A → Y → B
abbrev Fib2  (A Y B : Type) := B → Y → Set A
abbrev SpecPFib (A Y B : Type) := B → Y → Set A → Prop
abbrev Spec3 (A Y B : Type) := A → Y → B → Prop


def T_pfib (R : Spec3 A Y B) : SpecPFib A Y B :=
  fun b y S => S = {x | R x y b}


-- phi liberal partial-fiber
def Phi_pfib (f : Fwd2 A Y B) (g : Fib2 A Y B) : Prop :=
  (∀ y b x, x ∈ g b y → f x y = b) ∧
  (∀ x y, x ∈ g (f x y) y)


-- Resonator: only fiber functions g that admit some f with Phi_pfib
structure Resonator2 (A Y B : Type) where
  g    : Fib2 A Y B
  spec : ∃ f : Fwd2 A Y B, Phi_pfib f g


-- Setoids
def EP_Fwd2 : Setoid (Fwd2 A Y B) :=
{ r := fun f₁ f₂ => ∀ x y, f₁ x y = f₂ x y,
  iseqv := ⟨
    fun _ _ _ => rfl,
    fun h x y => (h x y).symm,
    fun h₁ h₂ x y => (h₁ x y).trans (h₂ x y)
  ⟩ }
def EQ_Res2 : Setoid (Resonator2 A Y B) :=
{ r := fun g₁ g₂ => ∀ b y, g₁.g b y = g₂.g b y,
  iseqv := ⟨
    fun _ _ _ => rfl,
    fun h b y => (h b y).symm,
    fun h₁ h₂ b y => (h₁ b y).trans (h₂ b y)
  ⟩ }



def CorrectF2 (R : Spec3 A Y B) (f : Fwd2 A Y B) : Prop := ∀ x y, R x y (f x y)
def CorrectG2 (R' : SpecPFib A Y B) (g : Resonator2 A Y B) : Prop := ∀ b y, R' b y (g.g b y)

def triang_PARTIAL_FOR_FIB :
  Triangulation (Spec3 A Y B) (SpecPFib A Y B) (Fwd2 A Y B) (Resonator2 A Y B) :=
{ T := T_pfib , Phi := fun f g => Phi_pfib f g.g , EP := EP_Fwd2 , EQ := EQ_Res2

, injective := by
    intro f₁ f₂ g h1 h2
    intro x y
    have hx : x ∈ g.g (f₁ x y) y := h1.right x y
    have eq2 : f₂ x y = f₁ x y := h2.left y (f₁ x y) x hx
    exact eq2.symm

, surjective := by
    intro g
    rcases g.spec with ⟨f, hf⟩
    exact ⟨f, hf⟩

, totality := by
    intro f
    let g : Fib2 A Y B := fun b y => {x | f x y = b}
    have hΦ : Phi_pfib f g := by
      constructor
      · intro y b x hx; exact hx
      · intro x y; simp [g]
    exact ⟨⟨g, ⟨f, hΦ⟩⟩, hΦ⟩,

  CorrectP := fun R f => CorrectF2 R f, CorrectQ := fun R' g => CorrectG2 R' g

, correctness_coupling := by
    intro d f g hG h0 x y
    have hx : x ∈ g.g (f x y) y := h0.right x y
    rw [hG (f x y) y] at hx
    exact hx }


end PARTIAL_FOR_FIB









/-
 Proposition 5.11 - COR-FIB
-/

section COR_FIB

variable {A B : Type}


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



def CorrectCorr (R : Spec A B) (f : A → Set B) : Prop := ∀ a, setOf (fun b => R a b) = f a
def CorrectFib  (R : Spec A B) (g : B → Set A) : Prop := ∀ b, setOf (fun a => R a b) = g b

lemma cor_fib_mem_iff
  (f : A → Set B) (g : B → Set A)
  (h : COR_FIB_simplified f g) :
  ∀ a b, b ∈ f a ↔ a ∈ g b := by
  classical
  rcases h with ⟨Hb, Ha⟩
  intro a b
  constructor
  ·
    intro hb
    rcases (Ha a).1 with ⟨b0, hb0f, ha_in_gb0⟩
    have g_const : ∀ b₁ ∈ f a, g b₁ = g b0 := by
      intro b1 hb1
      exact (Ha a).2 b1 b0 hb1 hb0f
    have : g b = g b0 := g_const b hb
    -- because a ∈ g b0 and g b = g b0, so a ∈ g b
    simpa [this] using ha_in_gb0
  ·
    intro ha_in_gb
    rcases (Hb b).1 with ⟨a0, ha0_in_gb, hb_in_fa0⟩
    have f_const : ∀ a1 a2, a1 ∈ g b → a2 ∈ g b → f a1 = f a2 := (Hb b).2
    have : f a = f a0 := f_const a a0 ha_in_gb ha0_in_gb
    simpa [this] using hb_in_fa0


lemma cor_fib_correctness_coupling
  (R : Spec A B) (f : A → Set B) (g : B → Set A)
  (hg : CorrectFib R g) (hφ : COR_FIB_simplified f g) :
  CorrectCorr R f := by
  classical
  intro a
  apply Set.ext
  intro b
  have memIff := cor_fib_mem_iff f g hφ a b
  constructor
  · -- R a b → b ∈ f a
    intro hR
    apply (memIff).2
    rwa [← hg b]
  · -- b ∈ f a → R a b
    intro hb
    have ha_mem_gb : a ∈ g b := (memIff).1 hb
    rw [← hg b] at ha_mem_gb
    exact ha_mem_gb

abbrev Spec_cor_fib (A B : Type) := A → B → Prop
abbrev SpecCorFib (A B : Type) := B → Set A → Prop


def T_cor_fib (R : Spec A B) : SpecFib A B := fun b S => S = {a | R a b}


structure ResCorr (A B : Type) where
  f    : A → Set B
  spec : ∃ g : B → Set A, COR_FIB_simplified f g
structure ResFib (A B : Type) where
  g    : B → Set A
  spec : ∃ f : A → Set B, COR_FIB_simplified f g


def EP_ResCorr : Setoid (ResCorr A B) :=
{ r := fun F₁ F₂ => ∀ a, F₁.f a = F₂.f a,
  iseqv := ⟨
    (by intro F a; rfl),
    (by intro F₁ F₂ h a; simpa using (h a).symm),
    (by intro F₁ F₂ F₃ h₁ h₂ a; simp [h₁ a, h₂ a])
  ⟩}
def EQ_ResFib : Setoid (ResFib A B) :=
{ r := fun G₁ G₂ => ∀ b, G₁.g b = G₂.g b,
  iseqv := ⟨
    (by intro G b; rfl),
    (by intro G₁ G₂ h b; simpa using (h b).symm),
    (by intro G₁ G₂ G₃ h₁ h₂ b; simp [h₁ b, h₂ b])
  ⟩}


def CorrectCorr' (R : Spec A B) (F : ResCorr A B) : Prop :=
  CorrectCorr R F.f
def CorrectFib'  (R : Spec A B) (G : ResFib  A B) : Prop :=
  CorrectFib  R G.g
def CorrectFib'' (R' : SpecFib A B) (G : ResFib A B) : Prop := ∀ b, R' b (G.g b)


def triang_COR_FIB  :
  Triangulation (Spec A B) (SpecCorFib A B) (ResCorr A B) (ResFib A B) :=
{ T  := T_cor_fib , Phi := fun F G => COR_FIB_simplified F.f G.g , EP := EP_ResCorr, EQ := EQ_ResFib

, injective := by
    intro F₁ F₂ G h1 h2
    intro a
    apply Set.ext
    intro b
    have iff1 := cor_fib_mem_iff (F₁.f) (G.g) h1
    have iff2 := cor_fib_mem_iff (F₂.f) (G.g) h2
    constructor
    · intro hb
      have := (iff1 a b).1 hb  -- b ∈ F₁.f a → a ∈ G.g b
      exact (iff2 a b).2 this  -- a ∈ G.g b → b ∈ F₂.f a
    · intro hb
      have := (iff2 a b).1 hb  -- b ∈ F₂.f a → a ∈ G.g b
      exact (iff1 a b).2 this  -- a ∈ G.g b → b ∈ F₁.f a


, surjective := by
    intro G
    rcases G.spec with ⟨f, hf⟩
    exact ⟨⟨f, ⟨G.g, hf⟩⟩, hf⟩


, totality := by
    intro F
    rcases F.spec with ⟨g, hg⟩
    exact ⟨⟨g, ⟨F.f, hg⟩⟩, hg⟩,


  CorrectP := fun R F => CorrectCorr' R F, CorrectQ := fun R' G => CorrectFib'' R' G


, correctness_coupling := by
    intro d F G hG hPhi a
    apply Set.ext
    intro b
    constructor
    · -- b ∈ {b | d a b} → b ∈ F.f a
      intro hdb
      have memiff := cor_fib_mem_iff F.f G.g hPhi a b
      apply memiff.2
      have eq : G.g b = {a' | d a' b} := hG b
      rw [eq]
      exact hdb

    · -- b ∈ F.f a → b ∈ {b | d a b}
      intro hb
      have memiff := cor_fib_mem_iff F.f G.g hPhi a b
      have ha_mem : a ∈ G.g b := memiff.1 hb
      have eq : G.g b = {a' | d a' b} := hG b
      rw [eq] at ha_mem
      exact ha_mem


}

end COR_FIB
