# internship record （English）5.30

1. ## abstract

   the idea：resonate and predicate

   Objective: Based on a requirement R (which may be precise or imprecise, where "imprecise" means multiple valid outputs are allowed for a single input, distinct from a non-injective function; e.g., for the input set {1, 3, -2}, the requirement is to output a number greater than 0, where valid outputs could be 1 or 3), input a prompt that is as accurate and complete as possible to a large language model (LLM) to generate a target program (essentially code implementing the requirement). We must then verify the correctness of this generated program.

   Previous Work: We proposed methods to verify correctness, including:

   1.Consistency-based verification (maximum equivalence class partitioning),
   2.LLM self-reported confidence scores (weak correlation between confidence and correctness),
   3.Assertion function generation (assertions likely to share homologous errors with the target program).
   However, these methods fail to address the following issues:

   1.Homologous errors: All candidate programs within the same equivalence class may share the same logical flaws (e.g., misdefining critical conditions like x∈{1,2,3}). Assertions generated using the same flawed logic will falsely validate incorrect programs as "correct."
   2.Infinite input sets: Verification becomes impossible for requirements with infinite input domains (e.g., x∈Z) due to the inability to exhaustively test all inputs.
   3.Imprecise requirements: For ambiguous requirements (e.g., "output must be sorted," "no duplicates," "concise," or "add items"), the lack of executable, strict criteria makes it impossible to determine whether the output truly satisfies the semantics of the requirement. Without formal specifications (e.g., predicates or invariants), we cannot define what constitutes a "correct" output.

2. ## what is resonate

   First, we have a requirement R (here focusing solely on addressing homologous errors, temporarily disregarding precision). We instruct a large language model (LLM) to generate a target program based on R. We aim to use this method to determine whether the target program is correct while avoiding misjudgment caused by homologous errors.

   The target program generates an output z for every input x and y (generalized here; in reality, there may be multiple parameters beyond x and y). Additionally, we generate a family of resonators (not a single resonator, but multiple) based on the inverse requirement R ^−1 . Each resonator employs distinct algorithms and alters the semantics for the inverse requirement (this step is key to resolving homologous errors).

   Specifically, for the output Z obtained from the previous step and a sample input y, the resonators generate the input x (achieving partial inversion of parameters). We then define a resonance property: if satisfied, it strongly indicates the correctness of the target program (since the likelihood of homologous errors is now minimal). If not satisfied, we can rapidly localize the error.

   During the process of concretization, we encountered the following issues:

   1. For requirement R：There exists a distinction between precise and imprecise requirements. For imprecise requirements, fundamentally, a single input may correspond to multiple valid outputs. For example, consider the resonance property:

      ```
                                    f(r(f(x, y), y),y) ≈ f(x, y)
      ```

      When an input (x,y) maps to multiple outputs z1 ,z2 ,z3 , this property may no longer hold. However, this does not necessarily indicate that the program is incorrect.

   2. For infinite input domains：There are two scenarios:
      First: The input set itself is inherently infinite (e.g., x∈Z).
      Second: When constructing the resonator family based on the inverse requirement R ^−1 , the inverse process yields infinite outputs.For the first case, the inability to exhaustively test all inputs makes the verification of the property incomplete, undermining the credibility of the validation.
      For the second case, the inverse function r cannot reconstruct all possible original inputs x. The large language model (LLM) may autonomously compromise—for instance, truncating values or applying rounding—to produce results that appear correct but distort the property’s validity.

3. ## what is predicate

   R：Each user type corresponds to a pricing rule. User types are infinite. For user types 1, 2, and 3, the price remains unchanged. For all other user types, the price is increased by 10.

   target programme：

   ```
def f(x, y):
       if x in [1, 2, 3]:
           return y
       else:
           return y + 10
   ```
   
   predicate:

   ```
                         P = (x∈{1,2,3} ∧ z=y) ∨ (x∉{1,2,3} ∧ z=y+10)
   ```
   
   Formalization in Lean4:

   ```
def P' (x y z : Int) : Prop :=
     (x = 1 ∨ x = 2 ∨ x = 3) ∧ z = y ∨ -- x ∈ {1,2,3} → z = y
     (x ≠ 1 ∧ x ≠ 2 ∧ x ≠ 3) ∧ z = y + 10 -- x ∉ {1,2,3} → z = y+10
   
   ```
   
   The predicate P is derived from requirement R using logical expressions (equations, inequalities, set relations). Its purpose is to formally define the input-output constraints of the program. These constraints can be written and proven in formal languages like Lean4.

4. ##  predicate solve infinite input domains

   The core reason why infinite input domains can’t be fully handled is that we can’t list every possible input, or the tool r can’t produce all inputs, making it impossible to check each one. By adding rules, we pull out the logic behind the requirement R and its reverse version R ^−1. These rules turn "checking every input one by one" into "just checking if certain logic conditions are met," so we don’t need to test every input anymore.

   example:

   R :Given two integers x and y, return the larger one. Here, the input is infinite.

   P :

   ```
   def maxSpec (x y z : Int) : Prop :=
     (z = x ∨ z = y) ∧ z ≥ x ∧ z ≥ y
   ```

   With predicates, we only need to check if a limited set of inputs and outputs satisfy those predicates to verify compliance with the underlying logic, avoiding the need to test every possible implementation.

5. ## predicate solve inexact requirements

   The core idea of inexact requirements is that a single input can have multiple valid outputs. Before adding predicates, we checked if all implementations gave the same result. After adding predicates, we instead check if the output satisfies the predicate. Even if one implementation gives  f₁(x,y) = z₁, and another gives f₂(x,y) = z₂, both are correct as long as p(x,y,zᵢ) holds. This avoids forcing strict equality  in the implementations.

8. ##  resonate + predicate

   We have a natural language requirement R, and let an LLM generate a target program f based on R. Our goal is to verify the correctness of f using this method. We also generate a formal predicate p from R, and we consider p to be fully correct (aligned with formal specifications and the mathematical logic/semantics of R).

   R：Given a list of integers, return the largest positive integer in the list.
   
   f：
   
   ```
   def f(x):
       positive_numbers = [n for n in x if n > 0]
       return max(positive_numbers) if positive_numbers else 0
   ```
   
   p(lean4)：
   
   ```
   def P (x : List Int) (z : Int) : Bool :=
     match list_max (x.filter (· > 0)) with
     | some m => z = m
     | none   => z = 0 
   ```
   
   We test the program f by feeding it inputs x and checking if its output z follows the rules in p. If it ever breaks the rules, f is wrong. But even if it passes all tests, it might still have hidden bugs. For example:
   
   R：Pick a positive number from an array.
   
   p:
   
   ```
                               P(x, _, z) := (z ∈ x) ∧ (z > 0)
   ```
   
   f(Hard-code a return value of 3):
   
   ```
   f2(x, _) = return 3
   ```
   
   No matter what input is provided, this predicate holds, but clearly f is incorrect.
   
   Therefore, we adopt resonance to generate the inverse requirement R ^−1 , the inverse implementation f ^−1 , and the inverse predicate p ^−1.
   
   R^-1:For a given output value z, derive a legal input x such that f(x) returns z.
   
   If z=0, then x contains no positive numbers.
   If z>0, then z is the largest positive integer in x.
   
   f^-1:
   
   ```
   def f_inv(z, x):
       if z == 0:
           return [n for n in x if n <= 0]
       else:
           return [n for n in x if n <= z] 
   ```
   
   p^-1:
   
   ```
   def P_inv (z : Int) (x : List Int) : Bool :=
     match list_max (x.filter (· > 0)) with
     | some m => z = m
     | none   => z = 0
   ```
   
   By using  f⁻¹(z, x) to derive the input x，we then check if the inverse predicate P⁻¹(z, x) passes the reverse verification.这里和上Key considerations for this step (and the earlier step of verifying with predicate p) include:
   
   For outputs with simple, well-defined mathematical logic (e.g., numerical results), we can directly use tools like Lean4 to run formal verification (as in the example above).
   
   ```
   def list_max (xs : List Int) : Option Int :=
     match xs with
     | []      => none
     | x :: xs => some (xs.foldl (λ acc n => if n > acc then n else acc) x)
     
   def f (x : List Int) : Int :=
     match list_max (x.filter (· > 0)) with
     | some m => m
     | none   => 0
     
    def P (x : List Int) (z : Int) : Bool :=
     match list_max (x.filter (· > 0)) with
     | some m => z = m
     | none   => z = 0
    
   def f_inv (z : Int) : List Int :=
     if z ≤ 0 then [-1, -2]  
     else [z, z - 1, -3]  
     
   def P_inv (z : Int) (x : List Int) : Bool :=
     match list_max (x.filter (· > 0)) with
     | some m => z = m
     | none   => z = 0
       
   #eval f [3, -1, 5, 2]  
   #eval P [3, -1, 5, 2] 5  
   #eval P [3, -1, 5, 2] 4  
   
   #eval f_inv 5           
   #eval f (f_inv 5)       
   ```
   
   ![image-20250529004042294](C:\Users\lenovo\AppData\Roaming\Typora\typora-user-images\image-20250529004042294.png)
   
   For outputs like natural language, which cannot be encoded in Lean4, we might rely on LLMs to generate forward-checking functions (e.g., in Python) based on predicate p. Similarly, p ^−1could generate reverse algorithms. This shows there are multiple methods to implement the predicates.
   
   Next, we implement the resonance property between p and p ^−1 :
   
   ```
                                            P(x,z)⟺P^−1(z,x)
   ```
   
   ```
   theorem resonance_equiv : ∀ (x : List Int) (z : Int), P x z = P_inv z x := by
     intros x z
     unfold P P_inv
     simp
   ```
   
   if all verification steps pass, we conclude that f is correct.
   
   

